"""Take an actual headless Edge screenshot of the saved profiler report."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', type=Path, default=None)
    args = parser.parse_args()
    browser = args.browser or next((p for p in [
        Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe',
        Path(os.environ.get('PROGRAMFILES', 'C:/Program Files'))/'Google/Chrome/Application/chrome.exe'
    ] if p.exists()), None)
    if browser is None:
        found = shutil.which('chromium') or shutil.which('google-chrome')
        browser = Path(found) if found else None
    if browser is None:
        raise SystemExit('Pass --browser pointing to a Chromium/Edge executable')
    profile = STAGE.parent.parent/'.cache/stage4-profile-browser'
    profile.mkdir(parents=True, exist_ok=True)
    image = STAGE/'logs/profiling_screenshot.png'
    command = [str(browser),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
               '--hide-scrollbars','--window-size=2600,1100',f'--user-data-dir={profile}',
               f'--screenshot={image}',(STAGE/'logs/profiling.html').as_uri()]
    try:
        subprocess.run(command,capture_output=True,timeout=45)
    except subprocess.TimeoutExpired:
        if not image.exists():
            raise
    if not image.is_file():
        raise RuntimeError('browser did not produce the requested screenshot')
    trace = json.loads((STAGE/'logs/attention_trace.json').read_text(encoding='utf-8'))
    kernels = [e for e in trace['traceEvents'] if e.get('cat')=='kernel']
    result = {'screenshot':'logs/profiling_screenshot.png','source':'logs/profiling.html',
              'raw_trace':'logs/attention_trace.json','trace_cuda_kernels':len(kernels),
              'kernel_names':sorted({e['name'] for e in kernels}),
              'capture':'actual Chromium headless screenshot of torch.profiler output; not an Nsight capture'}
    (STAGE/'logs/profiling_capture.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
