"""Plot actual metrics only; each exported figure is a single evidence panel."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fit_scaling import power, joint, load_cells

STAGE = Path(__file__).resolve().parents[1]
COLORS = ['#738291', '#b89473', '#658e84', '#527aa3', '#a7778c']
LABELS = dict(eager='PyTorch eager', sdpa_math='SDPA math',
              flashattention2_ref='Official FlashAttention-v2 (Triton)',
              triton_32='Our Triton (BM=32)', triton_64='Our Triton (BM=64)')
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif',
                     'font.sans-serif':['Arial','DejaVu Sans'], 'pdf.fonttype':42,
                     'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False,
                     'legend.frameon':False})


def save(fig, name):
    base = STAGE/'curves'/name
    fig.set_size_inches(9, 4.8)
    for ax in fig.axes:
        ax.legend(fontsize=7.5,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    fig.tight_layout(rect=(0,.065,1,1))
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=str(base)+'.alignment.json')
    except ImportError:
        if len(fig.axes) != 1:
            raise RuntimeError('Multi-panel export requires the alignment auditor')
        Path(str(base)+'.alignment.json').write_text(json.dumps({'status':'NOT APPLICABLE', 'reason':'single panel'}),encoding='utf-8')
    fig.savefig(str(base)+'.png',dpi=300)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    plt.close(fig)


def benchmark_plots():
    with (STAGE/'logs/benchmark_metrics.csv').open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['execution'], row['method'], int(row['sequence']))].append(float(row['latency_ms']))
    for mode in ('launch', 'cuda_graph'):
        fig, ax = plt.subplots(figsize=(7.2,4.6))
        for color,(method,label) in zip(COLORS,LABELS.items()):
            sizes = sorted(n for execution,m,n in grouped if m==method and execution==mode)
            if not sizes:
                continue
            samples = [grouped[mode,method,n] for n in sizes]
            center = np.median(samples,axis=1)
            lower, upper = np.quantile(samples,[.1,.9],axis=1)
            ax.errorbar(sizes,center,yerr=[center-lower,upper-center],label=label,
                        color=color,marker='o',capsize=2,lw=1.5,ms=3)
        ax.set(xscale='log',yscale='log',xlabel='Sequence length N (tokens)',ylabel='Latency (ms)',
               title='Causal attention forward | '+('CUDA Graph replay' if mode=='cuda_graph' else 'Host launch included'))
        ax.set_xticks([64,128,256,512,1024,2048],labels=['64','128','256','512','1024','2048'])
        ax.legend(fontsize=7.5,loc='upper left')
        ax.grid(alpha=.18)
        fig.text(.14,.01,'Median; error bars: 10th–90th percentile of 25 timing samples. B=2, H=12, d=64, FP16.',fontsize=7)
        save(fig,'attention_'+mode)


def scaling_plots():
    fit = json.loads((STAGE/'logs/scaling_fit.json').read_text(encoding='utf-8'))
    cells = load_cells()
    for axis in ('N','D'):
        spec = fit['one_dimensional'][axis]
        fixed = spec['fixed_axis_value']
        subset = [(key,v) for key,v in sorted(cells.items()) if key[1 if axis=='N' else 0]==fixed]
        x = np.asarray([k[0 if axis=='N' else 1] for k,_ in subset])
        y = np.asarray([np.mean(v) for _,v in subset])
        sd = np.asarray([np.std(v,ddof=1) for _,v in subset])
        params = [spec['coefficients'][k] for k in ('E','A','alpha')]
        maximum = 1e8 if axis=='N' else 1e6
        grid = np.geomspace(x[0],maximum,250)
        inside = grid<=x[-2]
        fig,ax = plt.subplots(figsize=(7.2,4.6))
        ax.plot(grid[inside],power(grid[inside]/spec['x0'],params),color=COLORS[3],label='Fit (smaller sizes only)')
        ax.plot(grid[~inside],power(grid[~inside]/spec['x0'],params),'--',color=COLORS[3],label='Extrapolation (dashed)')
        ax.errorbar(x[:-1],y[:-1],yerr=sd[:-1],fmt='o',color=COLORS[0],capsize=3,label='Fit data, mean ± seed SD')
        ax.errorbar(x[-1:],y[-1:],yerr=sd[-1:],fmt='s',color=COLORS[4],capsize=3,label='Held-out size, mean ± seed SD')
        ax.set(xscale='log',xlabel='Parameters N' if axis=='N' else 'Distinct training positions D (tokens)',
               ylabel='Validation cross-entropy (nats/token)',title=f'L({axis}) | fixed '+('D' if axis=='N' else 'N')+f'={fixed:,}')
        ax.legend(fontsize=7.5,loc='upper right')
        ax.grid(alpha=.18)
        fig.text(.14,.01,f'3 seeds; 2 epochs. Held-out relative error: {spec["held_out_relative_error"]:.2%}. Far extrapolation is unvalidated.',fontsize=7)
        save(fig,'scaling_'+axis)
    p = [fit['coefficients'][name] for name in ('E','A','alpha','B','beta')]
    for x_axis in ('N','D'):
        fig,ax = plt.subplots(figsize=(7.2,4.6))
        if x_axis=='N':
            grid = np.geomspace(min(fit['model_parameters']),1e8,200)
            for color,d in zip(COLORS,fit['data_budgets']):
                pred = joint((grid/1e6,np.full_like(grid,d/65536)),p)
                measured = grid<=max(fit['model_parameters'])
                ax.plot(grid[measured],pred[measured],color=color,label=f'D={d:,}')
                ax.plot(grid[~measured],pred[~measured],'--',color=color)
            ax.set(xscale='log',xlabel='Parameters N',ylabel='Predicted validation loss',title='Joint power-law extrapolation (dashed = unmeasured N)')
            ax.legend(fontsize=7.5,loc='upper right')
            ax.grid(alpha=.18)
            fig.text(.14,.01,'Fit uses only the smaller 4 × 4 cells. E reaches zero boundary; distant predictions are speculative.',fontsize=7)
            save(fig,'scaling_law')
    fig,ax = plt.subplots(figsize=(7.2,4.6))
    for split,color,marker in [('fit',COLORS[0],'o'),('held_out',COLORS[4],'s')]:
        selected = [(k,v) for k,v in cells.items()
                    if ('fit' if k[0]<max(fit['model_parameters']) and k[1]<max(fit['data_budgets']) else 'held_out') == split]
        observed = np.asarray([np.mean(v) for k,v in selected])
        predicted = np.asarray([joint((np.asarray(k[0]/1e6),np.asarray(k[1]/65536)),p) for k,v in selected])
        ax.errorbar(observed,predicted,xerr=[np.std(v,ddof=1) for k,v in selected],fmt=marker,
                    color=color,capsize=2,label=split.replace('_',' ').capitalize())
    observed_all = [np.mean(v) for v in cells.values()]
    sd_all = [np.std(v,ddof=1) for v in cells.values()]
    predicted_all = [float(joint((np.asarray(k[0]/1e6),np.asarray(k[1]/65536)),p)) for k in cells]
    lower = min(min(np.asarray(observed_all)-sd_all),min(predicted_all))
    upper = max(max(np.asarray(observed_all)+sd_all),max(predicted_all))
    margin = (upper-lower)*.04
    lims=[lower-margin,upper+margin]
    ax.plot(lims,lims,'--',color='#b4b4b4',lw=1)
    ax.set(xlabel='Observed validation loss (mean ± seed SD)',ylabel='Predicted validation loss',
           title='Whole-size holdout validation',xlim=lims,ylim=lims)
    ax.legend(fontsize=8)
    fig.text(.14,.01,f'16 fitted cells; 9 held-out cells. MAPE={fit["holdout_mape"]:.2%}; max relative error={fit["holdout_max_relative_error"]:.2%}.',fontsize=7)
    save(fig,'scaling_holdout')


if __name__ == '__main__':
    benchmark_plots()
    scaling_plots()
