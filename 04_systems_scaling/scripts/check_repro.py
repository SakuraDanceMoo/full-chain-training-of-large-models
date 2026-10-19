"""Check archived experimental evidence without rerunning GPU workloads."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((STAGE/name).read_text(encoding='utf-8'))


def main():
    checks = {}
    config, split = read('config.json'), read('data/processed/split.json')
    for item in read('data/manifest.json')['files']:
        checks['raw_hash'] = hashlib.sha256((STAGE/item['path']).read_bytes()).hexdigest() == item['sha256']
    tokenizer = STAGE.parent/split['tokenizer']
    checks['tokenizer_hash'] = hashlib.sha256(tokenizer.read_bytes()).hexdigest() == split['fingerprint']['tokenizer_sha256']
    checks['train_validation_gap'] = split['maximum_training_tokens']+config['block_size'] < split['validation_start']
    with (STAGE/'logs/scaling_runs.csv').open(encoding='utf-8') as file:
        runs = list(csv.DictReader(file))
    expected = set(itertools.product([m['name'] for m in config['models']],config['data_tokens'],config['seed_list']))
    actual = {(r['model'],int(r['data_tokens']),int(r['seed'])) for r in runs}
    checks['all_75_runs_unique'] = expected == actual and len(runs) == 75
    checks['finite_losses_and_real_work'] = all(math.isfinite(float(r['validation_loss'])) and
        float(r['train_seconds']) > 0 and int(r['tokens_seen']) == config['epochs']*(int(r['data_tokens'])-config['block_size']) for r in runs)
    checks['loss_decreased_all_runs'] = all(float(r['validation_loss']) < float(r['initial_loss']) for r in runs)
    correctness = read('logs/attention_correctness.json')
    checks['attention_correctness'] = correctness['passed'] and len(correctness['cases']) == 108 and correctness['causal_future_invariance']
    with (STAGE/'logs/benchmark_metrics.csv').open(encoding='utf-8') as file:
        metrics = list(csv.DictReader(file))
    groups = {}
    for r in metrics:
        groups.setdefault((r['execution'],r['method'],r['sequence']),[]).append(r)
    checks['benchmark_samples'] = all(len(v)==25 for v in groups.values()) and {k[0] for k in groups} == {'launch','cuda_graph'}
    checks['benchmark_backends'] = {k[1] for k in groups} >= {'eager','sdpa_math','flashattention2_ref','triton_32','triton_64'}
    checks['finite_benchmark_latency'] = all(math.isfinite(float(r['latency_ms'])) and float(r['latency_ms'])>0 for r in metrics)
    trace = read('logs/attention_trace.json')
    kernels = [e for e in trace['traceEvents'] if e.get('cat')=='kernel']
    names = ' '.join(e['name'] for e in kernels)
    checks['real_profile_and_screenshot'] = len(kernels)>0 and '_attention' in names and '_attn_fwd' in names and (STAGE/'logs/profiling_screenshot.png').read_bytes().startswith(b'\x89PNG')
    fit = read('logs/scaling_fit.json')
    checks['whole_size_holdout'] = fit['training_cells']==16 and fit['held_out_cells']==9
    checks['mean_error_gate'] = fit['holdout_mape'] <= config['local_extrapolation_mape_limit'] and fit['local_mape_gate_passed']
    checks['distant_prediction_labeled_unvalidated'] = fit['speculative_0_1b']['validated'] is False
    checks['multi_gpu_cancelled'] = config['multi_gpu']=='cancelled_by_user'
    nb = read('notebooks/scaling_law.ipynb')
    code = [c for c in nb['cells'] if c['cell_type']=='code']
    checks['notebook_executed_without_errors'] = all(c.get('execution_count') is not None and not any(o['output_type']=='error' for o in c.get('outputs',[])) for c in code)
    for name in ['attention_launch','attention_cuda_graph','scaling_N','scaling_D','scaling_holdout','scaling_law']:
        checks[f'{name}_exports'] = all((STAGE/f'curves/{name}.{ext}').is_file() for ext in ('png','pdf','svg'))
    result = {'passed':all(checks.values()),'checks':checks,'note':'Evidence checks; not a rerun of GPU correctness or training. Far extrapolation remains unvalidated.'}
    (STAGE/'logs/reproducibility_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
