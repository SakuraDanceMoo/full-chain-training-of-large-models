"""Reload a small PEFT checkpoint against the pinned shared base and re-evaluate."""
import argparse
import json
from pathlib import Path

import torch

from data_pipeline import STAGE,load_data,tokenizer,write_json
from engine import seed_all,evaluate
from peft import make_model,apply_trainable


def reload_and_evaluate(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    meta=payload['metadata'];cfg=meta['config']
    seed_all(meta['seed'])
    model=make_model(cfg,meta['peft'],anchor=not meta['warmup'])
    apply_trainable(model,payload['trainable'])
    result=evaluate(model,load_data()['test'],tokenizer(cfg))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=reload_and_evaluate(args.checkpoint)
    if args.output: write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))
