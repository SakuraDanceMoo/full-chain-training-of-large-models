"""Reproduce all arms in separate CUDA processes, then aggregate real results."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from data_pipeline import ROOT,STAGE,prepare,sha256,source_sha256,write_json


def child(args,log):
    env={**os.environ,'PYTHONUTF8':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',*args]
    with log.open('w',encoding='utf-8') as handle:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',env=env)
        for line in proc.stdout:
            handle.write(line);handle.flush();print(line,end='',flush=True)
        if proc.wait():
            raise RuntimeError(f'worker failed: {log}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--method',default='sft')
    p.add_argument('--peft',default='lora')
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--warmup',action='store_true')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    config=json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    if args.worker:
        import torch
        from engine import train,seed_all,evaluate
        from peft import make_model
        from data_pipeline import load_data,tokenizer
        if args.baseline:
            seed_all(1337)
            tok=tokenizer(config);data=load_data()
            model=make_model(config,'lora')
            test=evaluate(model,data['test'],tok)
            write_json(STAGE/'logs/anchor_test_predictions.json',test)
            print(json.dumps({k:v for k,v in test.items() if k!='predictions'},indent=2),flush=True)
            return
        train(config,args.method,args.peft,args.seed,warmup=args.warmup)
        return
    (STAGE/'logs').mkdir(exist_ok=True)
    (STAGE/'models').mkdir(exist_ok=True)
    prepare(config)
    import torch
    versions={m:importlib.metadata.version(m) for m in ['torch','numpy','matplotlib','bitsandbytes']}
    evidence={'timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),'timezone':'Asia/Hong_Kong','platform':platform.platform(),
              'python':sys.version,'packages':versions,'device':torch.cuda.get_device_name(0),
              'cuda_runtime':torch.version.cuda,'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory,
              'config':config,'config_sha256':sha256(STAGE/'config.json'),
              'tokenizer_sha256':source_sha256(ROOT/config['tokenizer']),
              'base_parts':{p.name:source_sha256(p) for p in (ROOT/config['base_checkpoint']).parent.glob('best_checkpoint*')},
              'text_hash_policy':'Source JSON normalized to LF; raw, generated LF JSON and binary checkpoint hashes are exact bytes',
              'note':'Single fixed warmup shared by all seeds and methods. BF16 base, FP32 PEFT parameters/AdamW; no multi-GPU.'}
    write_json(STAGE/'logs/protocol.json',evidence)
    if not (args.resume and (STAGE/'models/common_sft_anchor.pt').exists()):
        child(['--warmup','--seed',str(config['warmup_seed'])],STAGE/'logs/warmup_console.txt')
    if not (args.resume and (STAGE/'logs/anchor_test_predictions.json').exists()):
        child(['--baseline'],STAGE/'logs/baseline_console.txt')
    arms=[('sft','lora'),('dpo','lora'),('rlvr','lora'),('sft','qlora'),('sft','adapter')]
    results=[]
    for method,kind in arms:
        for seed in config['seeds']:
            name=f'{method}_{kind}_s{seed}'
            resultfile=STAGE/f'logs/runs/{name}/result.json'
            if not (args.resume and resultfile.exists()):
                child(['--method',method,'--peft',kind,'--seed',str(seed)],STAGE/f'logs/{name}_console.txt')
            results.append(json.loads(resultfile.read_text(encoding='utf-8')))
            flat=[{k:v for k,v in r.items() if k!='quantization'} for r in results]
            with (STAGE/'logs/evaluation_metrics.csv').open('w',newline='',encoding='utf-8') as h:
                w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    subprocess.run([sys.executable,str(STAGE/'scripts/plot_results.py')],check=True)


if __name__=='__main__':
    main()
