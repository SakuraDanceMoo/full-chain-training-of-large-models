"""LoRA, genuine bitsandbytes NF4 QLoRA, and parameter-matched residual Adapter."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT/'05_posttraining'
sys.path.insert(0,str(ROOT/'02_gpt/scripts'))
from minigpt import GPT,GPTConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int):
        super().__init__()
        self.base=base
        self.scale=2.0
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))

    def forward(self,x):
        result=self.base(x)
        delta=F.linear(F.linear(x.float(),self.a),self.b)*self.scale
        return result+delta.to(result.dtype)


class AdapterBlock(nn.Module):
    def __init__(self,block: nn.Module,width:int,dim:int):
        super().__init__()
        self.base=block
        self.down=nn.Linear(dim,width,bias=False,dtype=torch.float32)
        self.up=nn.Linear(width,dim,bias=False,dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        x=self.base(x)
        return x+self.up(F.gelu(self.down(x.float()))).to(x.dtype)


def install(model,kind,config):
    model.requires_grad_(False)
    if kind in ['lora','qlora']:
        for block in model.blocks:
            for name in ['qkv','proj']:
                setattr(block.attn,name,LoRALinear(getattr(block.attn,name),config['rank']))
    elif kind=='adapter':
        model.blocks=nn.ModuleList([AdapterBlock(b,config['adapter_width'],model.config.n_embd) for b in model.blocks])
    else:
        raise ValueError(kind)
    return model


def load_pretrained(config):
    manifest_path=ROOT/config['base_checkpoint']
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    cfg=GPTConfig(**manifest['model_config'])
    cfg.dropout=0.0  # Same deterministic training/evaluation mode in every arm.
    model=GPT(cfg)
    state={}
    for part in manifest['parts']:
        state.update(torch.load(manifest_path.parent/part,map_location='cpu',weights_only=True)['model'])
    model.load_state_dict(state)
    return model.to(torch.bfloat16)


def trainable_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def apply_trainable(model,state):
    params=dict(model.named_parameters())
    for name,value in state.items():
        if name not in params or not params[name].requires_grad:
            raise ValueError(f'unexpected trainable key {name}')
        params[name].data.copy_(value.to(params[name]))


def merge_lora(model):
    for block in model.blocks:
        for name in ['qkv','proj']:
            layer=getattr(block.attn,name)
            if isinstance(layer,LoRALinear):
                layer.base.weight.data.add_((layer.b@layer.a*layer.scale).to(layer.base.weight.dtype))
                setattr(block.attn,name,layer.base)
    return model.requires_grad_(False)


def anchor_model(config):
    model=install(load_pretrained(config),'lora',config)
    payload=torch.load(STAGE/'models/common_sft_anchor.pt',map_location='cpu',weights_only=True)
    apply_trainable(model,payload['trainable'])
    return merge_lora(model)


def quantize_nf4(model):
    import bitsandbytes as bnb
    def replace(parent):
        for name,child in list(parent.named_children()):
            if name=='lm_head':
                continue  # Keep tied embedding/head in BF16, as in all PEFT arms.
            if isinstance(child,nn.Linear):
                new=bnb.nn.Linear4bit(child.in_features,child.out_features,bias=child.bias is not None,
                                     compute_dtype=torch.bfloat16,compress_statistics=True,
                                     quant_type='nf4',quant_storage=torch.uint8)
                new.weight=bnb.nn.Params4bit(child.weight.detach().cpu(),requires_grad=False,
                                            compress_statistics=True,quant_type='nf4')
                if child.bias is not None:
                    new.bias=nn.Parameter(child.bias.detach().cpu(),requires_grad=False)
                setattr(parent,name,new)
            else:
                replace(child)
    replace(model)
    return model.to('cuda')


def make_model(config,kind='lora',anchor=True):
    base=anchor_model(config) if anchor else load_pretrained(config)
    if kind=='qlora':
        base=quantize_nf4(base)
    return install(base,kind,config).to('cuda')


def quantization_metadata(model):
    modules=[m for m in model.modules() if m.__class__.__name__=='Linear4bit']
    return {'linear4bit_modules':len(modules),
            'packed_uint8_modules':sum(m.weight.dtype==torch.uint8 for m in modules),
            'nested_quant_modules':sum(bool(m.weight.quant_state.nested) for m in modules),
            'quant_types':sorted({m.weight.quant_state.quant_type for m in modules})}


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
