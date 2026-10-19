"""Single-GPU N x D ablations using stage-one BPE and stage-two GPT.

D means distinct positions in a contiguous training subset, not tokens seen.
Two epochs are used for every run, so compute and D covary. No convergence claim.
Validation is fixed before selecting any training subset or fitting a model.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
sys.path.insert(0, str(ROOT / '01_tokenizer/scripts'))
sys.path.insert(0, str(ROOT / '02_gpt/scripts'))
from tokenizer import BPETokenizer
from minigpt import GPT, GPTConfig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config):
    raw = STAGE / 'data/raw/tinyshakespeare.txt'
    if not raw.exists():
        shutil.copyfile(ROOT / '02_gpt/data/raw/tinyshakespeare.txt', raw)
    original = ROOT / '02_gpt/data/raw/tinyshakespeare.txt'
    if sha(raw) != sha(original):
        raise ValueError('raw corpus does not match the archived original')
    tokpath = ROOT / '03_pretraining/models/tokenizer_bpe_512.json'
    tok = BPETokenizer.load(tokpath)
    processed = STAGE / 'data/processed'
    cache = processed / 'tokens.npz'
    fingerprint = {'raw_sha256': sha(raw), 'tokenizer_sha256': sha(tokpath)}
    meta = processed / 'split.json'
    if cache.exists() and meta.exists():
        previous = json.loads(meta.read_text(encoding='utf-8'))
        if previous.get('fingerprint') != fingerprint:
            raise ValueError('changed corpus/tokenizer: remove stale processed cache explicitly')
        ids = np.load(cache)['ids']
    else:
        ids = np.asarray(tok.encode(raw.read_text(encoding='utf-8')), dtype=np.int32)
        np.savez_compressed(cache, ids=ids)
    val_count = config['validation_tokens']
    # Leave a whole context gap between train and validation slices.
    available = len(ids) - val_count - config['block_size']
    if available <= max(config['data_tokens']):
        raise ValueError('not enough independent train/validation tokens')
    manifest = {'files': [{'path': 'data/raw/tinyshakespeare.txt', 'encoding': 'UTF-8',
                          'sha256': sha(raw), 'source': 'Archived stage-two tinyShakespeare corpus'}]}
    (STAGE / 'data/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    split = {'fingerprint': fingerprint, 'total_tokens': len(ids), 'train_start': 0,
             'maximum_training_tokens': max(config['data_tokens']),
             'validation_start': len(ids)-val_count, 'validation_tokens': val_count,
             'context_gap_tokens': available-max(config['data_tokens']),
             'tokenizer': '03_pretraining/models/tokenizer_bpe_512.json',
             'vocab_size': tok.vocab_size}
    meta.write_text(json.dumps(split, indent=2), encoding='utf-8')
    return torch.tensor(ids[:available], dtype=torch.long, device='cuda'), torch.tensor(ids[-val_count:], dtype=torch.long, device='cuda'), tok.vocab_size


@torch.no_grad()
def evaluate(model, validation, block_size, batch_size):
    model.eval()
    losses = []
    # Non-overlapping fixed contexts: identical validation tokens for all runs.
    starts = list(range(0, validation.numel()-block_size, block_size))
    for i in range(0, len(starts), batch_size):
        chosen = starts[i:i+batch_size]
        x = torch.stack([validation[s:s+block_size] for s in chosen])
        y = torch.stack([validation[s+1:s+block_size+1] for s in chosen])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append((float(loss), len(chosen)*block_size))
    return sum(l*w for l,w in losses)/sum(w for _,w in losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='skip completed CSV rows with identical config')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    record = STAGE/'logs/scaling_config.json'
    if args.resume and record.exists() and json.loads(record.read_text(encoding='utf-8')) != config:
        raise ValueError('config differs from the logged run; cannot resume')
    record.write_text(json.dumps(config, indent=2), encoding='utf-8')
    train, validation, vocab = prepare(config)
    target = STAGE/'logs/scaling_runs.csv'
    rows = []
    if args.resume and target.exists():
        with target.open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
    completed = {(r['model'], int(r['data_tokens']), int(r['seed'])) for r in rows}
    history = STAGE/'logs/scaling_training.jsonl'
    if not args.resume:
        history.write_text('', encoding='utf-8')
    for specification in config['models']:
        name = specification['name']
        for data_tokens in config['data_tokens']:
            for seed in config['seed_list']:
                if (name, data_tokens, seed) in completed:
                    continue
                torch.manual_seed(seed)
                gptconf = GPTConfig(vocab_size=vocab, block_size=config['block_size'], dropout=0.,
                                    **{k:v for k,v in specification.items() if k != 'name'})
                model = GPT(gptconf).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                              weight_decay=config['weight_decay'])
                parameters = sum(p.numel() for p in model.parameters())
                block, batch = config['block_size'], config['batch_size']
                # Adjacent non-overlapping contexts, shuffled afresh each epoch.
                starts = torch.arange(0, data_tokens-block, block, device='cuda')
                steps = config['epochs'] * math.ceil(len(starts)/batch)
                step, seen = 0, 0
                initial = evaluate(model, validation, block, batch)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for epoch in range(config['epochs']):
                    model.train()
                    order = starts[torch.randperm(len(starts), device='cuda')]
                    for j in range(0, len(order), batch):
                        selection = order[j:j+batch]
                        offsets = selection[:, None] + torch.arange(block, device='cuda')[None, :]
                        x, y = train[offsets], train[offsets+1]
                        progress = step / max(steps-1, 1)
                        warmup = min(1., (step+1)/max(steps*.1, 1))
                        lr = config['learning_rate'] * warmup * (.1+.9*.5*(1+math.cos(math.pi*progress)))
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            _, loss = model(x, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        seen += x.numel()
                        if step % max(steps//4, 1) == 0:
                            event = dict(model=name, data_tokens=data_tokens, seed=seed, step=step,
                                         loss=float(loss.detach()), learning_rate=lr)
                            with history.open('a', encoding='utf-8') as file:
                                file.write(json.dumps(event)+'\n')
                        step += 1
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                final = evaluate(model, validation, block, batch)
                row = dict(model=name, parameters=parameters, data_tokens=data_tokens, seed=seed,
                           initial_loss=initial, validation_loss=final, steps=steps, tokens_seen=seen,
                           train_seconds=seconds, tokens_per_second=seen/seconds,
                           peak_memory_bytes=torch.cuda.max_memory_allocated())
                rows.append(row)
                with target.open('w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)
                del optimizer, model
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
