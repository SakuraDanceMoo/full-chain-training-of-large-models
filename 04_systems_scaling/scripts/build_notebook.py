"""Build and execute the scaling-law notebook with this Python interpreter."""
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

STAGE = Path(__file__).resolve().parents[1]


def main():
    output = STAGE/'notebooks'
    output.mkdir(exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = dict(name='stage4-local',display_name='Stage 4 Python',language='python')
    nb.cells = [
        nbf.v4.new_markdown_cell('# Stage 4 · Scaling law on measured single-GPU runs\n\n'
            '75 real runs: 5 model sizes × 5 data budgets × 3 seeds. Models reuse stage-two GPT; '
            'stage-one BPE is loaded from stage three. All multi-GPU tasks are cancelled.\n\n'
            'Fit on the smaller 4×4 cells only; hold out every largest-N/largest-D cell. '
            'D counts distinct positions, while training uses two epochs. Fixed validation tokens are separated before choosing training subsets.'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json, csv\nfrom IPython.display import display, Image\n"
            "STAGE = Path.cwd() if (Path.cwd()/'config.json').exists() else Path.cwd().parent\n"
            "sys.path.insert(0,str(STAGE/'scripts'))\nfrom fit_scaling import main as fit\nresult = fit()\n"),
        nbf.v4.new_markdown_cell('## Coefficients and normalization\n\n'
            'Joint model: `L=E + A*(N/1e6)^(-alpha) + B*(D/65536)^(-beta)`. '
            'One-dimensional fits hold D or N fixed. Coefficients and seed-only bootstrap intervals are printed below. '
            'The E boundary warning must be retained; it prevents interpreting the fitted floor physically.'),
        nbf.v4.new_code_cell("display(result['coefficients'])\ndisplay(result['coefficient_seed_bootstrap_95pct'])\n"
            "display(result['one_dimensional'])\nprint(result['coefficient_boundary_note'])"),
        nbf.v4.new_markdown_cell('## Local extrapolation check\n\n'
            'Maximum parameter count and maximum data budget were never used to estimate coefficients. '
            'The preregistered gate is **mean** absolute relative error ≤10%; maximum error is reported separately. '
            'Passing the mean gate is not a bound on every prediction.'),
        nbf.v4.new_code_cell("display({k:result[k] for k in ['training_cells','held_out_cells','fit_rmse','holdout_mape','holdout_max_relative_error','local_mape_gate_passed']})\n"
            "from plot_results import scaling_plots\nscaling_plots()\n"
            "for name in ['scaling_N','scaling_D','scaling_holdout','scaling_law']:\n    display(Image(filename=str(STAGE/'curves'/f'{name}.png')))"),
        nbf.v4.new_markdown_cell('## 0.1B extrapolation is a hypothesis\n\n'
            'This is a distant extrapolation on a single literary corpus after only two training epochs. '
            'No 0.1B model was trained with the same experiment protocol. The 95% bootstrap interval includes seed variation only; '
            'it excludes model misspecification and domain shift. It must not be used as a reliable performance forecast.'),
        nbf.v4.new_code_cell("display(result['speculative_0_1b'])\ndisplay(result['limitations'])"),
        nbf.v4.new_markdown_cell('## Reproduce\n\n'
            'From repository root: `python 04_systems_scaling/scripts/run_scaling.py`, '
            '`python 04_systems_scaling/scripts/fit_scaling.py`, '
            '`python 04_systems_scaling/scripts/plot_results.py`. '
            'This notebook refits archived measurements and redraws figures without retraining. Seeds and all hyperparameters are in `../config.json`.'),
    ]
    # A task-local kernelspec keeps execution on the exact environment, not a global Python.
    jupyter = STAGE.parent.parent/'.cache/stage4-jupyter'
    kernel = jupyter/'kernels/stage4-local'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Stage 4 Python','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    NotebookClient(nb,timeout=180,kernel_name='stage4-local',resources={'metadata':{'path':str(STAGE)}}).execute()
    # Deliver a portable default kernel name, while retaining executed outputs.
    nb.metadata.kernelspec = dict(name='python3',display_name='Python 3',language='python')
    nbf.write(nb,output/'scaling_law.ipynb')
    print('Executed notebook:',output/'scaling_law.ipynb')


if __name__=='__main__':
    main()
