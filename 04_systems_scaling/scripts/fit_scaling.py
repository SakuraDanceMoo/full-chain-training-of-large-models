"""Fit learning-scale power laws with whole-size holdout and seed bootstrap.

Fit form: L=E+A*(x/x0)^(-alpha). Joint form adds the independent D term.
The finite-epoch protocol is empirical, not a compute-optimal scaling claim.
"""
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

STAGE = Path(__file__).resolve().parents[1]


def power(x, p):
    e, a, alpha = p
    return e+a*np.asarray(x)**(-alpha)


def joint(x, p):
    e, a, alpha, b, beta = p
    n, d = x
    return e+a*n**(-alpha)+b*d**(-beta)


def solve(x, y, combined=False):
    fn = joint if combined else power
    low = [0., 0., .01, 0., .01] if combined else [0., 0., .01]
    high = [float(min(y)), 20., 3., 20., 3.] if combined else [float(min(y)), 20., 3.]
    starts = ([2., .2, .3, 1., .3], [0.1, 1., .1, 1., .1]) if combined else ([2., 1., .3], [.1, 2., .1])
    fits = [least_squares(lambda p: fn(x, p)-y, start, bounds=(low, high),
                          max_nfev=20000, ftol=1e-10, xtol=1e-10, gtol=1e-10) for start in starts]
    best = min(fits, key=lambda result: np.sum(result.fun**2))
    if not best.success:
        raise RuntimeError('nonlinear fit did not converge: '+best.message)
    return best.x


def load_cells():
    with (STAGE/'logs/scaling_runs.csv').open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    cells = {}
    for r in rows:
        key = (int(r['parameters']), int(r['data_tokens']))
        cells.setdefault(key, []).append(float(r['validation_loss']))
    if any(len(values) != 3 for values in cells.values()):
        raise ValueError('expected all three seeds in every cell')
    return cells


def main():
    config = json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    cells = load_cells()
    sizes = sorted({n for n,d in cells})
    budgets = sorted({d for n,d in cells})
    keys = sorted(cells)
    n = np.asarray([x[0] for x in keys], dtype=float)
    d = np.asarray([x[1] for x in keys], dtype=float)
    y = np.asarray([np.mean(cells[k]) for k in keys])
    fit_mask = (n<max(sizes)) & (d<max(budgets))
    x = (n/1e6, d/65536.)
    fit_x = (x[0][fit_mask], x[1][fit_mask])
    p = solve(fit_x, y[fit_mask], combined=True)
    predicted = joint(x, p)
    test_mask = ~fit_mask
    mape = float(np.mean(np.abs((predicted[test_mask]-y[test_mask])/y[test_mask])))
    relative = np.abs((predicted[test_mask]-y[test_mask])/y[test_mask])
    rng = np.random.default_rng(20261019)
    draws = []
    for _ in range(100):
        boot = np.asarray([rng.choice(cells[k], size=3, replace=True).mean() for k in keys])
        draws.append(solve(fit_x, boot[fit_mask], combined=True))
    draws = np.asarray(draws)
    coefficient_names = ['E', 'A', 'alpha', 'B', 'beta']
    intervals = {name: [float(v) for v in np.quantile(draws[:,i], [.025,.975])]
                 for i,name in enumerate(coefficient_names)}
    extrap_n = config['large_model_parameters']
    extrap_d = 1_000_000
    speculative = float(joint((np.asarray(extrap_n/1e6), np.asarray(extrap_d/65536)), p))
    bootstrap_predictions = [float(joint((np.asarray(extrap_n/1e6), np.asarray(extrap_d/65536)), row)) for row in draws]

    axes = {}
    for axis in ('N', 'D'):
        fixed = budgets[-2] if axis=='N' else sizes[-2]
        subset = [(k, values) for k,values in sorted(cells.items()) if k[1 if axis=='N' else 0] == fixed]
        xv = np.asarray([k[0 if axis=='N' else 1] for k,_ in subset], dtype=float)
        center = np.asarray([np.mean(v) for _,v in subset])
        scale = 1e6 if axis=='N' else 65536.
        params = solve(xv[:-1]/scale, center[:-1])
        hold_prediction = float(power(xv[-1]/scale, params))
        axes[axis] = dict(form='E + A*(x/x0)^(-alpha)', x0=scale,
                          coefficients=dict(zip(['E','A','alpha'], map(float,params))),
                          fixed_axis_value=fixed, fitted_sizes=list(map(int,xv[:-1])),
                          held_out_size=int(xv[-1]), held_out_observed_loss=float(center[-1]),
                          held_out_predicted_loss=hold_prediction,
                          held_out_relative_error=abs(hold_prediction-float(center[-1]))/float(center[-1]))
    result = dict(protocol='2 epochs, fixed BF16 GPT training protocol; D and compute covary',
                  model_parameters=sizes, data_budgets=budgets, seeds=config['seed_list'],
                  sample_count=sum(map(len,cells.values())), training_cells=int(fit_mask.sum()), held_out_cells=int(test_mask.sum()),
                  normalization=dict(N0=1e6,D0=65536),
                  joint_form='L(N,D)=E + A*(N/1e6)^(-alpha) + B*(D/65536)^(-beta)',
                  coefficients=dict(zip(coefficient_names,map(float,p))), coefficient_seed_bootstrap_95pct=intervals,
                  coefficient_boundary_warning=bool(p[0] < 1e-6),
                  coefficient_boundary_note='E reached its nonnegative boundary; this data does not identify a physical irreducible loss floor.',
                  bootstrap_replicates=len(draws), bootstrap_unit='resample seeds within each N,D cell; no model misspecification coverage',
                  fit_rmse=float(np.sqrt(np.mean((predicted[fit_mask]-y[fit_mask])**2))),
                  holdout_mape=mape, holdout_max_relative_error=float(relative.max()),
                  local_mape_limit=config['local_extrapolation_mape_limit'],
                  local_mape_gate_passed=mape<=config['local_extrapolation_mape_limit'],
                  one_dimensional=axes,
                  speculative_0_1b=dict(N=extrap_n, D=extrap_d, predicted_loss=speculative,
                      seed_bootstrap_95pct=list(map(float,np.quantile(bootstrap_predictions,[.025,.975]))),
                      validated=False, reason='Far outside measured N,D; no 0.1B checkpoint was trained under this protocol. Local MAPE cannot bound this error.'),
                  limitations=['Only 2 epochs, not convergence/compute-optimal scaling',
                    'Nested contiguous subsets confound data amount with composition and optimizer steps',
                    'Single literary corpus and one fixed validation segment',
                    'Finite-size loss floor and exponents can be weakly identifiable',
                    'Bootstrap captures seed variation, not corpus or model-family uncertainty'])
    (STAGE/'logs/scaling_fit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (STAGE/'data/processed/scaling_cell_summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['parameters','data_tokens','loss_mean','loss_seed_sd','split','joint_prediction'])
        for i,k in enumerate(keys):
            writer.writerow([*k,y[i],np.std(cells[k],ddof=1),'fit' if fit_mask[i] else 'held_out',predicted[i]])
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    main()
