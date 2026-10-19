"""One-command single-GPU stage-four reproduction; no multi-GPU entry point."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-only', action='store_true', help='refit archived metrics without GPU training')
    parser.add_argument('--resume', action='store_true', help='skip completed scaling runs with matching config')
    parser.add_argument('--browser', help='Chromium/Edge executable for the profiler screenshot')
    args = parser.parse_args()
    cache = STAGE.parent/'.cache/stage4'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONUTF8'] = '1'
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache/'triton'))
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    jobs = []
    if not args.analysis_only:
        jobs += [('benchmark.py', []), ('run_scaling.py', ['--resume'] if args.resume else [])]
        jobs += [('capture_profile.py', ['--browser', args.browser] if args.browser else [])]
    jobs += [('fit_scaling.py', []), ('plot_results.py', []), ('build_notebook.py', []),
             ('report_results.py', []), ('check_repro.py', [])]
    for script, extra in jobs:
        print('Running', script, *extra, flush=True)
        with (STAGE/'logs'/f'{Path(script).stem}_reproduction.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, str(STAGE/'scripts'/script), *extra],
                                       cwd=STAGE.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.decode('utf-8', errors='replace').rstrip('\r\n'), flush=True)
            if process.wait():
                raise SystemExit(f'{script} failed; inspect logs/{Path(script).stem}_reproduction.log')


if __name__ == '__main__':
    main()
