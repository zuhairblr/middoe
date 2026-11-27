from middoe.krnl_expera import expera
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
from middoe.des_pp import mbdoe_pp

# 1. The experiment: what you can change, what you can measure, and the limits
system = {
    't_s': [0, 10],    # each batch lasts 10 h
    't_r': 0.05,       # time grid step [h]
    't_d': 0.5,        # no sampling or switching in the first and last 0.5 h
    'tvi': {           # inputs that change during the batch
        'u1': {'min': 313.15, 'max': 343.15, 'stps': 3, 'cvp': 'CPF',
               'const': 'rel', 'offl': 1, 'offt': 1}},      # temperature [K]
    'tii': {           # inputs fixed for the whole batch
        'y10': {'min': 0.5, 'max': 2.0}},                     # initial A [mol/L]
    'tvo': {           # what you measure
        'y1': {'init': 0, 'meas': True, 'sp': 6, 'unc': 0.02,
               'offt': 1, 'samp_s': 1, 'samp_f': [0, 10]},    # A [mol/L]
        'y2': {'init': 0, 'meas': True, 'sp': 6, 'unc': 0.02,
               'offt': 1, 'samp_s': 1, 'samp_f': [0, 10]}},   # B [mol/L]
    'tio': {},
}

# 2. The model and its unknown parameters
models = {
    'can_m': ['M'],
    'krt': {'M': 'pys'},              # a Python file with solve_model(...)
    'src': {'M': 'model.py'},
    'creds': {'M': ''},
    'theta': {'M': [0.1, 40000]},     # first guess: k_ref [1/h], Ea [J/mol]
    't_l': {'M': [0.01, 20000]},      # lower bounds
    't_u': {'M': [2.0, 120000]},      # upper bounds
    'mutation': {'M': [True, True]},  # True = estimate this parameter
}

# 3. A "virtual lab": simulates experiments with the true parameters plus noise
insilicos = {
    'tr_m': 'M', 'theta': [0.3, 60000], 'errt': 'abs',
    'prels': {'1': {'u1': 323.15, 'y10': 1.0},     # two simple starting experiments
              '2': {'u1': 333.15, 'y10': 1.5}},
}

iden_opt = {'meth': 'LMBFGS', 'ob': 'WLS', 'var-cov': 'M', 'ms': False,
            'maxit': 500, 'tol': 1e-8, 'sens_m': 'central', 'eps': 1e-5}

des_opt = {'pp_ob': 'D', 'meth': 'PS', 'eps': 1e-5, 'plt': False,
           'itr': {'pps': 30, 'maxpp': 30, 'tolpp': 1e-3}}


def fit_and_report(label):
    est = parmest(system, models, iden_opt)            # fit the parameters to data.xlsx
    res = uncert(est, system, models, iden_opt)['results']['M']
    print(f"\n{label}")
    for name, val, ci, t in zip(['k_ref', 'Ea'], res['estimations'], res['CI'], res['t_values']):
        print(f"  {name:6s} = {val:10.4g}  +/- {ci:.3g}  (t = {t:.1f})")


if __name__ == '__main__':
    expera(system, models, insilicos, {}, expr=1)       # run starting experiment 1
    expera(system, models, insilicos, {}, expr=2)       # run starting experiment 2
    fit_and_report('After 2 starting experiments')

    design = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=1)  # plan experiment 3
    expera(system, models, insilicos, design, expr=3, swps=design['swps'])   # run it
    fit_and_report('After 1 designed experiment')
