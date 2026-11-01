"""Response-masked SFT, reference-correct DPO and on-policy verifiable GRPO."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from data_pipeline import STAGE, ROOT, load_data, tokenizer, tensors, write_json, source_sha256
from peft import (make_model, trainable_state, apply_trainable, count_trainable,
                  quantization_metadata)


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def amp():
    return torch.autocast('cuda',dtype=torch.bfloat16)


def sequence_logps(model, rows, tok, field='chosen', grad=True):
    x,y,mask=tensors(rows,tok,'cuda',field)
    with amp():
        logits,_=model(x,activation_checkpointing=grad)
    logps=F.log_softmax(logits.float(),-1).gather(-1,y.unsqueeze(-1)).squeeze(-1)
    return (logps*mask).sum(-1),logps,mask


@torch.no_grad()
def generate(model,rows,tok,sample=False,group=1,tokens=2):
    """Right pad per step; read each last valid position, never restrict vocabulary."""
    model.eval()
    prompts=[tok.encode(row['prompt'],add_bos=True) for row in rows for _ in range(group)]
    seqs=[p.copy() for p in prompts]
    responses=[[] for _ in seqs]
    stopped=[False]*len(seqs)
    eos=tok.token_to_id['<eos>']
    for _ in range(tokens):
        width=max(map(len,seqs))
        x=torch.full((len(seqs),width),tok.token_to_id['<pad>'],device='cuda',dtype=torch.long)
        for i,seq in enumerate(seqs):
            x[i,:len(seq)]=torch.tensor(seq,device='cuda')
        with amp():
            logits,_=model(x)
        last=logits[torch.arange(len(seqs),device='cuda'),torch.tensor([len(s)-1 for s in seqs],device='cuda')].float()
        ids=torch.multinomial(last.softmax(-1),1).flatten() if sample else last.argmax(-1)
        for i,value in enumerate(ids.tolist()):
            if not stopped[i]:
                responses[i].append(value)
                seqs[i].append(value)
                stopped[i]=(value==eos)
    return responses


def text_response(tok,ids):
    try:
        return tok.decode(ids).strip()
    except (UnicodeDecodeError, ValueError):
        # Full-vocabulary on-policy sampling can emit <unk> or invalid UTF-8 bytes.
        # An undecodable completion is an invalid answer with verifier reward zero.
        return '<invalid-completion>'


@torch.no_grad()
def evaluate(model,rows,tok):
    model.eval()
    predictions=[]
    nll=0.0
    count=0
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        total,logps,mask=sequence_logps(model,batch,tok,grad=False)
        nll-=float(total.sum())
        count+=int(mask.sum())
        responses=generate(model,batch,tok)
        for row,ids in zip(batch,responses):
            text=text_response(tok,ids)
            predictions.append({'id':row['id'],'prompt':row['prompt'],'gold':row['chosen'],
                                'prediction':text,'token_ids':ids,'correct':text==row['chosen'],
                                'valid':text in ['0','1']})
    accuracy=sum(p['correct'] for p in predictions)/len(predictions)
    return {'accuracy':accuracy,'format_rate':sum(p['valid'] for p in predictions)/len(predictions),
            'response_nll':nll/count,'response_ppl':math.exp(min(nll/count,30)),
            'n':len(rows),'predictions':predictions}


def dpo_loss(policy_chosen,policy_rejected,ref_chosen,ref_rejected,beta):
    return -F.logsigmoid(beta*((policy_chosen-policy_rejected)-(ref_chosen-ref_rejected))).mean()


def group_advantages(rewards,group):
    r=rewards.reshape(-1,group)
    return ((r-r.mean(-1,keepdim=True))/(r.std(-1,keepdim=True,unbiased=False)+1e-6)).flatten()


def save_delta(model,path,metadata):
    torch.save({'trainable':trainable_state(model),'metadata':metadata},path)


def train(config,method,kind,seed,warmup=False,steps_override=None,output_id=None):
    seed_all(seed)
    torch.set_num_threads(8)
    data=load_data()
    tok=tokenizer(config)
    run_id=output_id or ('warmup' if warmup else f'{method}_{kind}_s{seed}')
    logdir=STAGE/'logs/runs'/run_id
    logdir.mkdir(parents=True,exist_ok=True)
    (STAGE/'models').mkdir(exist_ok=True)
    start_setup=time.perf_counter()
    model=make_model(config,kind,anchor=not warmup)
    initial_state=trainable_state(model)
    params=[p for p in model.parameters() if p.requires_grad]
    lr=config['warmup_lr'] if warmup else config[f'{method}_lr']
    optim=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
    steps=steps_override or (config['warmup_steps'] if warmup else config['steps'])
    ref_scores={}
    reference=None
    if method=='dpo':
        model.eval()
        with torch.no_grad():
            for start in range(0,len(data['train']),32):
                rows=data['train'][start:start+32]
                c,_,_=sequence_logps(model,rows,tok,grad=False)
                r,_,_=sequence_logps(model,rows,tok,'rejected',grad=False)
                ref_scores.update({row['id']:(float(a),float(b)) for row,a,b in zip(rows,c,r)})
        write_json(logdir/'reference_logps.json',ref_scores)
    if method=='rlvr':
        reference=make_model(config,kind)
        reference.requires_grad_(False).eval()
    torch.cuda.synchronize()
    setup_seconds=time.perf_counter()-start_setup
    qmeta=quantization_metadata(model)
    val0=evaluate(model,data['validation'],tok)
    write_json(logdir/'initial_validation.json',val0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_seconds=0.0
    history=[]
    rng=random.Random(seed)
    with (logdir/'training.jsonl').open('w',encoding='utf-8') as handle:
        for step in range(1,steps+1):
            torch.cuda.synchronize()
            start=time.perf_counter()
            optim.zero_grad(set_to_none=True)
            diagnostics={}
            if method=='sft':
                rows=rng.choices(data['train'],k=config['batch_size'])
                model.train()
                total,logps,mask=sequence_logps(model,rows,tok)
                loss=-total.sum()/mask.sum()
            elif method=='dpo':
                rows=rng.choices(data['train'],k=config['batch_size'])
                model.train()
                c,_,_=sequence_logps(model,rows,tok)
                r,_,_=sequence_logps(model,rows,tok,'rejected')
                rc=torch.tensor([ref_scores[row['id']][0] for row in rows],device='cuda')
                rr=torch.tensor([ref_scores[row['id']][1] for row in rows],device='cuda')
                loss=dpo_loss(c,r,rc,rr,config['dpo_beta'])
                diagnostics={'preference_accuracy':float(((c-r)-(rc-rr)>0).float().mean().detach())}
            elif method=='rlvr':
                rows=rng.choices(data['train'],k=config['rlvr_prompt_batch'])
                group=config['rlvr_group_size']
                responses=generate(model,rows,tok,sample=True,group=group,tokens=config['rollout_tokens'])
                rollout=[]
                for i,row in enumerate(rows):
                    for j in range(group):
                        rollout.append({**row,'completion':responses[i*group+j]})
                rewards=torch.tensor([float(text_response(tok,r['completion'])==r['chosen']) for r in rollout],device='cuda')
                advantage=group_advantages(rewards,group)
                with torch.no_grad():
                    old,_,_=sequence_logps(model,rollout,tok,'completion',grad=False)
                    _,ref_logps,_=sequence_logps(reference,rollout,tok,'completion',grad=False)
                model.train()
                current,logps,mask=sequence_logps(model,rollout,tok,'completion')
                ratio=(current-old).exp()
                clipped=ratio.clamp(1-config['rlvr_clip'],1+config['rlvr_clip'])
                policy_loss=-torch.minimum(ratio*advantage,clipped*advantage).mean()
                diff=ref_logps-logps
                kl=((diff.clamp(-20,20).exp()-diff-1)*mask).sum()/mask.sum()
                loss=policy_loss+config['rlvr_kl_beta']*kl
                diagnostics={'rollout_reward':float(rewards.mean()),'nonzero_advantage_rate':float((advantage!=0).float().mean()),
                             'sampled_kl':float(kl.detach()),'policy_loss':float(policy_loss.detach())}
            else:
                raise ValueError(method)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss in {run_id} step {step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0)
            optim.step()
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            train_seconds+=elapsed
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),
                 'train_step_seconds':elapsed,**diagnostics}
            # Peak is sampled before validation; eval memory never enters training peak.
            row['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            row['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
            peak_alloc=row['peak_allocated_mib']; peak_reserved=row['peak_reserved_mib']
            if step%config['eval_interval']==0 or step==steps:
                val=evaluate(model,data['validation'],tok)
                row.update({'val_accuracy':val['accuracy'],'val_response_nll':val['response_nll']})
                print(f'{run_id} {step}/{steps} loss={row["loss"]:.4f} val={val["accuracy"]:.3f} sec={train_seconds:.1f}',flush=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            history.append(row)
            handle.write(json.dumps(row)+'\n');handle.flush()
    # Training peak is the maximum of per-step observations, not the last reset.
    peak_alloc=max(r['peak_allocated_mib'] for r in history)
    peak_reserved=max(r['peak_reserved_mib'] for r in history)
    changed=sum(float((p.detach().cpu()-initial_state[n]).square().sum()) for n,p in model.named_parameters() if p.requires_grad)
    result={'run_id':run_id,'method':method,'peft':kind,'seed':seed,'steps':steps,'learning_rate':lr,
            'trainable_parameters':count_trainable(model),'train_seconds':train_seconds,'setup_seconds':setup_seconds,
            'peak_allocated_mib':peak_alloc,'peak_reserved_mib':peak_reserved,'delta_l2':math.sqrt(changed),
            'quantization':qmeta,'initial_val_accuracy':val0['accuracy'],
            'val_accuracy':val['accuracy'],'val_response_nll':val['response_nll']}
    checkpoint=STAGE/'models'/('common_sft_anchor.pt' if warmup else run_id+'.pt')
    metadata={'config':config,'method':method,'peft':kind,'seed':seed,'warmup':warmup,
              'base_manifest_sha256':source_sha256(ROOT/config['base_checkpoint']),
              'tokenizer_sha256':source_sha256(ROOT/config['tokenizer']),
              'source_hash_mode':'UTF-8 JSON with LF line endings'}
    save_delta(model,checkpoint,metadata)
    if not warmup:
        test=evaluate(model,data['test'],tok)
        write_json(logdir/'test_predictions.json',test)
        result.update({'test_accuracy':test['accuracy'],'test_format_rate':test['format_rate'],
                       'test_response_nll':test['response_nll'],'test_response_ppl':test['response_ppl'],'test_n':test['n']})
    result['checkpoint']=checkpoint.relative_to(STAGE).as_posix()
    write_json(logdir/'result.json',result)
    return result


def cli(method):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=STAGE/'config.json')
    parser.add_argument('--peft',choices=['lora','qlora','adapter'],default='lora')
    parser.add_argument('--seed',type=int,default=1337)
    parser.add_argument('--steps',type=int)
    args=parser.parse_args()
    print(json.dumps(train(json.loads(args.config.read_text(encoding='utf-8')),method,args.peft,args.seed,steps_override=args.steps),indent=2))
