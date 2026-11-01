"""Verify sources, split isolation, measured run matrix, real quantization and artifacts."""
import argparse
import json
import math
from pathlib import Path

from data_pipeline import STAGE,ROOT,sha256,source_sha256,write_json,load_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--reload',action='store_true');args=p.parse_args()
    cfg=json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    protocol=json.loads((STAGE/'logs/protocol.json').read_text(encoding='utf-8'))
    assert sha256(STAGE/'config.json')==protocol['config_sha256']
    assert source_sha256(ROOT/cfg['tokenizer'])==protocol['tokenizer_sha256']
    for name,digest in protocol['base_parts'].items():
        assert source_sha256((ROOT/cfg['base_checkpoint']).parent/name)==digest
    manifest=json.loads((STAGE/'data/manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:assert sha256(STAGE/entry['path'])==entry['sha256']
    data=load_data();sets=[{r['id'] for r in rows} for rows in data.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    for name,info in manifest['splits'].items():
        assert sha256(STAGE/f'data/processed/{name}.json')==info['sha256']
    checked=[]
    reloads=[]
    checkpoints={}
    for method,kind in [('sft','lora'),('dpo','lora'),('rlvr','lora'),('sft','qlora'),('sft','adapter')]:
        for seed in cfg['seeds']:
            run=f'{method}_{kind}_s{seed}'
            base=STAGE/'logs/runs'/run
            result=json.loads((base/'result.json').read_text(encoding='utf-8'))
            assert result['steps']==cfg['steps'] and result['trainable_parameters']==442368
            assert result['delta_l2']>0
            hist=[json.loads(l) for l in (base/'training.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['step'] for r in hist]==list(range(1,cfg['steps']+1))
            assert all(math.isfinite(r['loss']) for r in hist)
            pred=json.loads((base/'test_predictions.json').read_text(encoding='utf-8'))
            assert [r['id'] for r in pred['predictions']]==[r['id'] for r in data['test']]
            assert pred['accuracy']==sum(r['correct'] for r in pred['predictions'])/len(data['test'])
            assert pred['accuracy']==result['test_accuracy']
            if kind=='qlora':
                q=result['quantization']
                assert q['linear4bit_modules']==q['packed_uint8_modules']==q['nested_quant_modules']==48
                assert q['quant_types']==['nf4']
            if method=='rlvr':
                assert any(r['nonzero_advantage_rate']>0 for r in hist)
            ck=STAGE/result['checkpoint'];assert ck.is_file()
            checkpoints[ck.name]={'bytes':ck.stat().st_size,'sha256':sha256(ck)}
            if args.reload and seed==cfg['seeds'][0]:
                from evaluate import reload_and_evaluate
                import torch,gc
                restored=reload_and_evaluate(ck)
                # Exact greedy IDs should survive save/reload, not just mean accuracy.
                assert restored['predictions']==pred['predictions'],run
                reloads.append(run)
                gc.collect();torch.cuda.empty_cache()
            checked.append(run)
    ck=STAGE/'models/common_sft_anchor.pt'
    checkpoints[ck.name]={'bytes':ck.stat().st_size,'sha256':sha256(ck)}
    write_json(STAGE/'models/manifest.json',{'format':'PEFT deltas only; base is repository stage three; shared anchor merged in BF16',
                'base_checkpoint':cfg['base_checkpoint'],'base_manifest_sha256':source_sha256(ROOT/cfg['base_checkpoint']),
                'checkpoint_metadata_note':'Checkpoints without source_hash_mode retain original runtime source bytes; portable source JSON hashes are in logs/protocol.json',
                'tokenizer':cfg['tokenizer'],'checkpoints':checkpoints})
    for img in ['training_loss','alignment_comparison','peft_resources','validation_accuracy']:
        assert (STAGE/f'curves/{img}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    result={'passed':True,'checked_runs':checked,'reload_predictions_identical':reloads,
            'run_count':len(checked),'splits':{k:len(v) for k,v in data.items()},
            'base_and_raw_hashes_verified':True,'quantized_storage_verified':True,
            'scope':'single CUDA, 3 seeds, fixed common warmup, synthetic comparison task'}
    write_json(STAGE/'logs/reproducibility_check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
