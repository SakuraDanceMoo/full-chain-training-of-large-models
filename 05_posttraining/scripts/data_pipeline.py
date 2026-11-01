"""Generate a fully enumerated, verifiable binary task with disjoint pair splits."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
sys.path.insert(0, str(ROOT / '01_tokenizer/scripts'))
from tokenizer import BPETokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256(path: Path) -> str:
    """Portable source JSON fingerprint; Git checkout may convert CRLF to LF."""
    raw=path.read_bytes()
    if path.suffix=='.json':
        raw=raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',newline='\n')


def prepare(config: dict) -> dict:
    raw = STAGE / 'data/raw/comparisons.jsonl'
    rows = []
    for a in range(32):
        for b in range(32):
            answer = str(int(a > b))
            rows.append({'id': f'{a:02d}_{b:02d}', 'a': a, 'b': b,
                         'prompt': f'Compare: {a:02d}>{b:02d}\nAnswer:',
                         'chosen': answer, 'rejected': str(1-int(answer))})
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists() and raw.read_bytes() != content.encode('utf-8'):
        raise ValueError('raw data changed: preserve it and choose a new dataset version')
    if not raw.exists():
        raw.write_bytes(content.encode('utf-8'))
    # Stratify by label, then split IDs, not individual chosen/rejected completions.
    rng = random.Random(config['data_seed'])
    splits = {'train': [], 'validation': [], 'test': []}
    for label in ['0', '1']:
        group = [r for r in rows if r['chosen'] == label]
        rng.shuffle(group)
        n = len(group)
        t, v = int(n*.75), int(n*.125)
        for name, values in zip(splits, [group[:t], group[t:t+v], group[t+v:]]):
            splits[name].extend(values)
    for name, values in splits.items():
        rng.shuffle(values)
        write_json(STAGE / f'data/processed/{name}.json', values)
    write_json(STAGE / 'data/manifest.json', {
        'source': 'Exhaustive locally generated integer comparisons 00..31; CC0-1.0',
        'generator': 'scripts/data_pipeline.py', 'encoding': 'UTF-8',
        'files': [{'path': 'data/raw/comparisons.jsonl', 'sha256': sha256(raw)}],
        'split_seed': config['data_seed'],
        'splits': {name: {'count': len(values), 'sha256': sha256(STAGE / f'data/processed/{name}.json'),
                          'labels': {s: sum(r['chosen']==s for r in values) for s in ['0','1']}}
                   for name, values in splits.items()},
        'deduplication': 'unique ordered (a,b) across splits; prompts shared across methods, never across splits',
        'protocol': 'train only for optimization and preference/reward construction; no validation/test feedback in updates'})
    return splits


def load_data() -> dict:
    return {s: json.loads((STAGE / f'data/processed/{s}.json').read_text(encoding='utf-8'))
            for s in ['train','validation','test']}


def tokenizer(config: dict) -> BPETokenizer:
    return BPETokenizer.load(ROOT / config['tokenizer'])


def completion_ids(tok, text: str) -> list[int]:
    return tok.encode(text, add_eos=True)


def tensors(rows: list[dict], tok, device, field='chosen'):
    import torch
    sequences, masks = [], []
    for row in rows:
        prompt = tok.encode(row['prompt'], add_bos=True)
        response = row['completion'] if field == 'completion' else completion_ids(tok, row[field])
        sequences.append(prompt + response)
        masks.append([0]*(len(prompt)-1) + [1]*len(response))
    width = max(map(len, sequences))-1
    pad = tok.token_to_id['<pad>']
    x = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    y = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, (seq, valid) in enumerate(zip(sequences,masks)):
        x[i,:len(seq)-1] = torch.tensor(seq[:-1],device=device)
        y[i,:len(seq)-1] = torch.tensor(seq[1:],device=device)
        mask[i,:len(valid)] = torch.tensor(valid,device=device,dtype=torch.bool)
    return x,y,mask


if __name__ == '__main__':
    cfg=json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    print({k:len(v) for k,v in prepare(cfg).items()})
