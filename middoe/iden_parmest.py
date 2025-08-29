import os
import warnings
warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step, clipping to bounds"
)

# Limit BLAS oversubscription (optional but helps stability)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
from scipy.stats import truncnorm
import numpy as np
import copy
import multiprocessing as mp
from operator import attrgetter
from concurrent.futures import ProcessPoolExecutor, as_completed

from numdifftools import Hessian
from scipy.optimize import minimize, differential_evolution

# External project imports
from middoe.krnl_simula import simula
from middoe.iden_utils import _initialize_dictionaries


# ------------------------------------------------------------------
# Multiprocessing context (important on Windows / Jupyter)
# ------------------------------------------------------------------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # already set
    pass
CTX = mp.get_context("spawn")


def _in_child():
    """Return True if running inside a worker process."""
    return mp.current_process().name != "MainProcess"


# ============================================================
# Helpers
# ============================================================


def _initialise_theta_parameters(theta_min, theta_max, active_models,  mutation=None, theta_fixed=None):
    """
    Sample theta values within bounds for each active model,
    but keep masked (inactive) ones fixed at theta_fixed.

    Args:
        theta_min, theta_max: bound dictionaries
        active_models: list of solver keys
        mutation: dict of boolean masks per solver (True=active)
        theta_fixed: dict of fixed theta values per solver

    Returns:
        theta_parameters: dict of initial theta lists per solver
    """
    theta_parameters = {}
    for sv in active_models:
        if mutation is not None and theta_fixed is not None:
            mask = np.array(mutation[sv], dtype=bool)
            theta_parameters[sv] = [
                np.random.uniform(lo, hi) if m else theta_fixed[sv][i]
                for i, (lo, hi, m) in enumerate(zip(theta_min[sv], theta_max[sv], mask))
            ]
        else:
            theta_parameters[sv] = [
                np.random.uniform(lo, hi) for lo, hi in zip(theta_min[sv], theta_max[sv])
            ]
    return theta_parameters


def _count_observations(data, system):
    """
    Count total number of scalar measurements (N) in 'data' that will be used by _objective.
    Does NOT require simulation, just metadata.
    """
    import pandas as _pd

    tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
    ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]

    N = 0
    for sh in data.values():
        # time-varying outputs
        for v in tv_o:
            xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
            if xc in sh and yc in sh:
                mask = ~sh[xc].isna().values
                N += int(mask.sum())

        # time-invariant outputs
        for v in ti_o:
            yc = f"MES_Y:{v}"
            if yc in sh and not _pd.isna(sh[yc].iloc[0]):
                N += 1

    return N


# ============================================================
# Public entry point
# ============================================================

def parmest(system, models, iden_opt, data, case=None):
    """
    Parameter estimation main entry.
    """
    if case != 'freeze' and 'mutation' in models:
        for solver in models['mutation']:
            models['mutation'][solver] = [True] * len(models['mutation'][solver])
    _initialize_dictionaries(models, iden_opt)

    active_models = models['can_m']
    bound_max, bound_min = models['t_u'], models['t_l']
    mutation = models['mutation']

    method = iden_opt['meth']
    objf = iden_opt['ob']
    multi = iden_opt.get('ms', False)

    bootstrap = (iden_opt.get('var-cov') == 'B')
    varcov= iden_opt.get('var-cov','H')
    nboot = iden_opt.get('nboot', 100)
    logging = iden_opt.get('log', False)
    maxit = iden_opt.get('maxit', 1000)
    tol = iden_opt.get('tol', 1e-6)


    # theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models)
    if case == 'strov' and 'thetastart' in models:
        theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models, mutation=mutation, theta_fixed=models['thetastart'])
        theta_params_f = models['thetastart']
    else:
        theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models, mutation=mutation, theta_fixed=models['theta'])
        theta_params_f = models['theta']

    x0_dict = {sv: [1.0] * len(theta_params_r[sv]) for sv in active_models}

    if bootstrap:
        return _bootstrap_runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                                 objf, x0_dict, method, data, system, models,
                                 logging, nboot, multi, varcov, maxit, tol)

    if multi:
        nstarts = max(1, int(0.7 * mp.cpu_count()))
        return _multi_start_runner(active_models, theta_params_r, bound_max, bound_min, mutation,
                                   objf, method, data, system, models, logging, varcov, maxit, tol, nstarts)

    return _runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                   objf, x0_dict, method, data, system, models, logging, varcov, maxit, tol)


# ============================================================
# Multi-start machinery
# ============================================================

def _run_single_start(sv, thetac, thetamin, thetamax, thetas, objf,
                      method, data, system, models, logging, varcov, maxit, tol):
    try:
        if logging:
            print(f"[{sv}] Sampled x0: {[1.0] * len(thetac)}")
        res = _runner([sv], {sv: thetac}, {sv: thetamax}, {sv: thetamin}, {sv: thetas},
                      objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov, maxit, tol)
        if logging:
            print(f"[{sv}] Finished with result: {res}")
        return (sv, res)
    except Exception as e:
        import traceback, sys
        tb = ''.join(traceback.format_exception(*sys.exc_info()))
        return (sv, e, tb)


def _multi_start_runner(active_models, _unused, bmax, bmin, mutation, objf,
                        method, data, system, models, logging, varcov, maxit, tol, nstarts=None):
    del _unused
    nwork = max(1, int(0.7 * mp.cpu_count()))
    nstarts = nstarts or max(2 * nwork, 10)

    allr = {sv: [] for sv in active_models}
    tasks = []
    for sv in active_models:
        for _ in range(nstarts):
            # tasks.append((sv,
            #               [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])],
            #               bmin[sv], bmax[sv], mutation[sv], objf, method,
            #               data, system, models, logging, varcov, maxit, tol))
            thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=models['theta'])[sv]
            tasks.append((sv, thetac, bmin[sv], bmax[sv], mutation[sv], objf, method,
                          data, system, models, logging, varcov, maxit, tol))

    with ProcessPoolExecutor(max_workers=nwork, mp_context=CTX) as exe:
        for fut in as_completed([exe.submit(_run_single_start, *t) for t in tasks]):
            out = fut.result()
            # Crash in worker?
            if len(out) == 3 and isinstance(out[1], Exception):
                sv, e, tb = out
                print(f"[{sv}] worker crashed:\n{tb}")
                continue
            sv, res = out
            if res and res.get(sv):
                allr[sv].append(res[sv])

    return {
        sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
             if any(r.success for r in allr[sv]) else None)
        for sv in active_models
    }


def _multi_start_runner_serial(active_models, _unused, bmax, bmin,
                               mutation, objf, method, data, system, models,
                               logging, varcov, maxit, tol, nstarts=None):
    del _unused
    nstarts = nstarts or 20
    allr = {sv: [] for sv in active_models}

    for sv in active_models:
        for _ in range(nstarts):
            # thetac = [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])]
            thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=models['theta'])[sv]

            if logging:
                print(f"[{sv}][SERIAL] Trying theta: {thetac}")
            res = _runner([sv], {sv: thetac}, {sv: bmax[sv]}, {sv: bmin[sv]}, {sv: mutation[sv]},
                          objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov, maxit, tol)
            if res and res.get(sv):
                allr[sv].append(res[sv])

    return {
        sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
             if any(r.success for r in allr[sv]) else None)
        for sv in active_models
    }


# ============================================================
# Bootstrap machinery
# ============================================================

def _bootstrap_worker(active_models, theta_params, bmax, bmin, mutation,
                      objf, x0, method, data, system, models, logging, multi, varcov, maxit, tol,
                      boot_idx):
    """Run ONE bootstrap replicate with a fixed index vector 'boot_idx'."""
    models_boot = {**models,
                   '__bootstrap__': True,
                   '__boot_idx__': boot_idx}

    if multi:
        # theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
        theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models, mutation=mutation, theta_fixed=models['theta'])
        ref = _multi_start_runner_serial(active_models, theta_params_r, bmax, bmin,
                                         mutation, objf, method, data, system,
                                         models_boot, logging, varcov, maxit, tol)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
                      x0, method, data, system, models_boot, logging, varcov, maxit, tol)
    return ref

# def truncated_moments_from_data(X, eps=1e-10):
#     """
#     Compute mean and covariance matrix from truncated normal samples
#     using analytical univariate moments and empirical correlations.
#     """
#     N, ndim = X.shape
#     mu_emp = np.mean(X, axis=0)
#     sigma_emp = np.std(X, axis=0, ddof=1)
#     bmin = np.min(X, axis=0)
#     bmax = np.max(X, axis=0)
#
#     mu_trunc = np.zeros(ndim)
#     var_trunc = np.zeros(ndim)
#
#     for i in range(ndim):
#         loc = mu_emp[i]
#         scale = max(sigma_emp[i], eps)
#         a = (bmin[i] - loc) / scale
#         b = (bmax[i] - loc) / scale
#         dist = truncnorm(a, b, loc=loc, scale=scale)
#         mu_trunc[i] = dist.mean()
#         var_trunc[i] = dist.var()
#
#     # Constant dimensions
#     constant_mask = sigma_emp < eps
#     if np.any(constant_mask):
#         print("Warning: constant parameters detected at:", np.where(constant_mask)[0])
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         corr_emp = np.corrcoef(X.T)
#     corr_emp[np.isnan(corr_emp)] = 0.0
#     np.fill_diagonal(corr_emp, 1.0)
#
#     sigma_trunc = np.sqrt(var_trunc)
#     cov_trunc = np.outer(sigma_trunc, sigma_trunc) * corr_emp
#
#     return mu_trunc, cov_trunc

def truncated_moments_from_data(X, eps=1e-10):
    """
    Compute mean and covariance matrix from truncated normal samples
    using analytical univariate moments and empirical correlations.
    """
    N, ndim = X.shape
    mu_emp = np.mean(X, axis=0)
    sigma_emp = np.std(X, axis=0, ddof=1)
    bmin = np.min(X, axis=0)
    bmax = np.max(X, axis=0)

    mu_trunc = np.zeros(ndim)
    var_trunc = np.zeros(ndim)

    for i in range(ndim):
        loc = mu_emp[i]
        scale = max(sigma_emp[i], eps)
        a = (bmin[i] - loc) / scale
        b = (bmax[i] - loc) / scale
        dist = truncnorm(a, b, loc=loc, scale=scale)
        mu_trunc[i] = dist.mean()
        var_trunc[i] = dist.var()

    # Constant dimensions
    constant_mask = sigma_emp < eps
    if np.any(constant_mask):
        print("Warning: constant parameters detected at:", np.where(constant_mask)[0])

    # Corr matrix handling — ensure it's always 2D
    with np.errstate(invalid='ignore', divide='ignore'):
        corr_emp = np.corrcoef(X.T)
        if np.isscalar(corr_emp):  # edge case: only one active parameter
            corr_emp = np.array([[1.0]])
        else:
            corr_emp = np.nan_to_num(corr_emp, nan=0.0)
            np.fill_diagonal(corr_emp, 1.0)

    sigma_trunc = np.sqrt(var_trunc)
    cov_trunc = np.outer(sigma_trunc, sigma_trunc) * corr_emp

    return mu_trunc, cov_trunc


def compute_trunc_mean_cov_from_empirical(resultpr, solver='M', scpr_key='scpr', include_X=True):
    """
    Compute truncated mean and covariance from empirical bootstrap samples.
    """
    resultpr2 = copy.deepcopy(resultpr)
    samples = resultpr2[solver]['samples']
    X = np.array([np.ravel(getattr(s, scpr_key)) for s in samples], dtype=np.float64)

    mu, cov = truncated_moments_from_data(X)

    resultpr2[solver]['scpr'] = mu
    resultpr2[solver]['v'] = cov
    if include_X:
        resultpr2[solver]['X'] = X

    return resultpr2


def _bootstrap_runner(active_models, theta_params, bmax, bmin, mutation,
                      objf, x0, method, data, system, models, logging,
                      nboot, multi, varcov, maxit, tol):

    if multi:
        trunc_mc_samps = nboot
        nstarts = max(1, int(0.7 * mp.cpu_count()))
        # theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
        theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models, mutation=mutation, theta_fixed=models['theta'])
        ref = _multi_start_runner(active_models, theta_params_r, bmax, bmin,
                                  mutation, objf, method, data, system, models,
                                  logging, varcov, maxit, tol, nstarts)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation,
                      objf, x0, method, data, system, models, logging,
                      varcov, maxit, tol)

    # Bootstrap sampling
    N = _count_observations(data, system)
    rng = np.random.default_rng()
    boot_indices = [rng.integers(0, N, size=N, endpoint=False) for _ in range(nboot)]

    allP = {sv: [] for sv in active_models}
    bootR = {sv: [] for sv in active_models}

    args_common = (active_models, theta_params, bmax, bmin, mutation,
                   objf, x0, method, data, system, models, logging,
                   multi, varcov, maxit, tol)

    with ProcessPoolExecutor(max_workers=min(nboot, mp.cpu_count()), mp_context=CTX) as exe:
        futures = [exe.submit(_bootstrap_worker, *args_common, bidx) for bidx in boot_indices]
        for fut in as_completed(futures):
            res = fut.result()
            for sv in active_models:
                r = res.get(sv)
                if r and getattr(r, 'success', True):
                    if ref[sv] is not None and r.fun <= 2 * ref[sv].fun:
                        if logging:
                            print(f'bootstrap WLS is {r.fun:.1f} and real one is {ref[sv].fun:.1f} for {sv}')
                        bootR[sv].append(r)
                        allP[sv].append(r.scpr)

    # Create full resultpr with both raw and trunc-normal stats
    resultpr = {}
    for sv in active_models:
        ref_sv = ref.get(sv)
        if ref_sv is None:
            continue

        # Attach bootstrap samples and stats
        ref_sv.samples = bootR[sv]
        if bootR[sv]:
            # X = np.stack([s.scpr for s in bootR[sv]])
            # mu_raw = np.mean(X, axis=0)
            # cov_raw = np.cov(X, rowvar=False, ddof=1)
            #
            # mu_trunc, cov_trunc = truncated_moments_from_data(X)
            #
            # ref_sv.X = X
            # ref_sv.scpr = mu_trunc
            # ref_sv.v = cov_trunc
            # ref_sv.scpr_raw = mu_raw
            # ref_sv.v_raw = cov_raw
            X_full = np.stack([s.scpr for s in bootR[sv]])  # shape: (nboot, nparams)
            mu_raw = np.mean(X_full, axis=0)

            # Masked (False) parameters are removed for covariance estimation
            mask = np.array(mutation[sv], dtype=bool)
            X_masked = X_full[:, mask]  # shape: (nboot, n_active_params)

            # Compute truncated stats only for unmasked parameters
            mu_trunc_masked, cov_trunc_masked = truncated_moments_from_data(X_masked)

            # Build full-length covariance matrix with zeros in masked rows/cols
            v_full = np.zeros((X_full.shape[1], X_full.shape[1]), dtype=np.float64)
            v_full[np.ix_(mask, mask)] = cov_trunc_masked

            ref_sv.scpr_raw = ref_sv.scpr
            ref_sv.X = X_full  # full sample matrix
            ref_sv.scpr = mu_raw  # full mean, including fixed params
            ref_sv.v = v_full  # full matrix, only active filled
            ref_sv.v_raw = np.cov(X_full, rowvar=False, ddof=1)


        else:
            ref_sv.X = None
            ref_sv.scpr = None
            ref_sv.v = None
            ref_sv.scpr_raw = None
            ref_sv.v_raw = None

    # Return the updated ref (which is now the enriched resultpr)
    return ref

# ============================================================
# Core optimizer runner
# ============================================================

def _runner(active_models, theta_params, bmax, bmin, mutation, objf,
            x0_dict, method, data, system, models, logging, varcov, maxit, tol):
    results = {}
    for sv in active_models:
        thetac = np.array(theta_params[sv], float)
        x0 = np.array(x0_dict[sv], float)

        thetamax = np.array(bmax[sv]) / thetac
        thetamin = np.array(bmin[sv]) / thetac

        mask = np.array(mutation[sv], bool)
        narrow = 1e-40

        bounds = [((x * (1 - narrow), x * (1 + narrow)) if not m else (tmin, tmax))
                  for x, m, tmin, tmax in zip(x0, mask, thetamin, thetamax)]

        bootstrap_flag = models.get('__bootstrap__', False)
        boot_idx = models.get('__boot_idx__', None)

        args = (data, [sv], thetac, system, models, logging, objf,
                bootstrap_flag, boot_idx)

        try:
            if method == 'SLSQP':
                res = minimize(_objective_function, x0, args=args, method='SLSQP',
                               bounds=bounds,
                               options={'maxiter': maxit, 'ftol': tol, 'disp': False})
            elif method == 'LBFGSB':
                res = minimize(_objective_function, x0, args=args, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxit, 'ftol': tol, 'disp': False})
            elif method == 'SQP':
                res = minimize(_objective_function, x0, args=args, method='trust-constr',
                               bounds=bounds,
                               options={'maxiter': maxit, 'xtol': tol, 'gtol': 1e-2})
            elif method == 'NM':
                res = minimize(_objective_function, x0, args=args, method='Nelder-Mead',
                               options={'maxiter': maxit, 'fatol': tol, 'disp': False})
            elif method == 'BFGS':
                res = minimize(_objective_function, x0, args=args, method='BFGS',
                               options={'maxiter': maxit, 'disp': False})
            elif method == 'DE':
                # Avoid nested pools: DE in worker uses 1 worker
                n_workers = 1 if _in_child() else -1
                res = differential_evolution(_objective_function, bounds, args=args,
                                             maxiter=maxit, popsize=18, tol=1e-8,
                                             strategy='best1bin', mutation=(0.5, 1.5),
                                             recombination=0.7, polish=False,
                                             updating='deferred', workers=n_workers)
            else:
                raise ValueError(f"Unknown method '{method}'")
        except Exception as e:
            if logging:
                print(f"[{sv}] optimize error: {e}")
            continue

        # Scale back to original theta
        res.scpr = res.x * thetac
        res.activeparams = mask

        # Hessian & inverse (only on the reference fits, not bootstrap replicates)

        if not bootstrap_flag and method in ['SLSQP', 'SQP', 'NM', 'DE'] and varcov == 'H':
            try:
                loss = lambda t: _objective_function(t, data, [sv], thetac, system, models,
                                                     logging, objf, False, None)
                H = Hessian(loss, step=1e-4, method='central', order=2)(res.x)
                S = np.diag(1.0 / thetac.astype(float))
                Hs = S @ H @ S
                res.hessian = Hs
            except Exception as e:
                if logging:
                    print(f"[{sv}] Hessian computation failed: {e}")
                res.hessian = None
                Hs = np.eye(len(thetac)) * 1e6

            D = np.diag(thetac.astype(float))
            try:
                inv = D @ (res.hess_inv if hasattr(res, 'hess_inv') else np.linalg.pinv(Hs)) @ D
                res.hess_inv = inv
            except Exception as e:
                if logging:
                    print(f"[{sv}] Hessian inversion failed: {e}")
                res.hess_inv = np.eye(len(thetac)) * 1e6

        results[sv] = res


    return results


# ============================================================
# Objective wrappers
# ============================================================

def _objective_function(theta, data, active, thetac, system, models,
                        logging, objf, bootstrap, boot_idx):
    # start = time.time()
    _, m = _objective(theta, data, active, thetac, system, models,
                      bootstrap=bootstrap, boot_idx=boot_idx)
    # if logging:
    #     print(f"Obj '{objf}'|{active[0]}|{time.time()-start:.3f}s")
    return {'LS': m['LS'], 'WLS': m['WLS'], 'MLE': m['MLE'], 'Chi': m['Chi']}[objf]


def _objective(theta, data, active, thetac, system, models,
               bootstrap=False, boot_idx=None):
    import pandas as _pd
    theta = theta.tolist()
    tv_i, ti_i = list(system['tvi']), list(system['tii'])
    tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
    ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]

    std_dev = {
        v: (cfg.get('unc', 1.0) if not np.isnan(cfg.get('unc', 1.0)) else 1.0)
        for v, cfg in {**system['tvo'], **system['tio']}.items()
        if cfg.get('meas', True)
    }

    phisc, phitsc, tsc = {v: 1 for v in ti_i}, {v: 1 for v in tv_i}, 1
    solver = active[0]

    global_y_true, global_y_pred, global_y_err, varm = {}, {}, {}, {}

    # --------------------------------------------------
    # Simulate & gather measurements
    # --------------------------------------------------
    for sh in data.values():
        t_all = np.unique(sh["X:all"].dropna().values)

        # sweeps (time varying inputs)
        swps = {}
        for v in tv_i:
            tk, lk = f"{v}t", f"{v}l"
            if tk in sh and lk in sh:
                ta, la = sh[tk].dropna().values, sh[lk].dropna().values
                if ta.size and la.size:
                    swps[tk], swps[lk] = ta, la

        # constants (time invariant inputs)
        ti_in = {v: sh.get(v, _pd.Series([np.nan])).iloc[0] for v in ti_i}
        tv_in = {v: sh.get(v, _pd.Series()).dropna().values for v in tv_i}
        cvp = {v: sh[f"CVP:{v}"].iloc[0] for v in system['tvi']}

        tvs, tis, _ = simula(t_all, swps, ti_in, phisc, phitsc, tsc,
                             theta, thetac, cvp, tv_in, solver, system, models)

        # time-varying outputs
        for v in tv_o:
            xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
            if xc not in sh or yc not in sh:
                continue
            mask = ~sh[xc].isna().values
            times = sh[xc][mask].values
            y_t = sh[yc][mask].values

            idx = np.isin(t_all, times)
            y_p = np.array(tvs[v])[idx]

            ye_col = f"MES_E:{v}"
            y_e = (sh[ye_col][mask].values if ye_col in sh
                   else np.full_like(y_t, std_dev[v]))

            global_y_true.setdefault(v, []).extend(y_t.tolist())
            global_y_pred.setdefault(v, []).extend(y_p.tolist())
            global_y_err.setdefault(v, []).extend(y_e.tolist())

            for yt, yp, ye in zip(y_t, y_p, y_e):
                varm.setdefault(v, []).append((yt, yp, ye))

        # time-invariant outputs
        for v in ti_o:
            yc = f"MES_Y:{v}"
            if yc not in sh:
                continue
            y_t = sh[yc].iloc[0]
            y_p = tis.get(v, np.nan)
            if np.isnan(y_t) or np.isnan(y_p):
                continue
            sigma = std_dev[v]
            global_y_true.setdefault(v, []).append(y_t)
            global_y_pred.setdefault(v, []).append(y_p)
            global_y_err.setdefault(v, []).append(sigma)
            varm.setdefault(v, []).append((y_t, y_p, sigma))

    # --------------------------------------------------
    # Flatten
    # --------------------------------------------------
    all_y_true, all_y_pred, all_y_err, all_vars = [], [], [], []
    for v, ytl in global_y_true.items():
        ypl = global_y_pred[v]
        yel = global_y_err[v]
        n = min(len(ytl), len(ypl), len(yel))
        all_y_true += ytl[:n]
        all_y_pred += ypl[:n]
        all_y_err += yel[:n]
        all_vars += [v] * n

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_err = np.array(all_y_err)
    all_vars = np.array(all_vars)
    N = len(all_y_true)

    # --------------------------------------------------
    # Bootstrap resampling (fixed index)
    # --------------------------------------------------
    if bootstrap:
        if boot_idx is None:
            raise ValueError("bootstrap=True but boot_idx is None.")
        idx = boot_idx
        if idx.size != N:
            raise ValueError(f"bootstrap index length {idx.size} != N {N}")
        y_t, y_p, y_e, vars_ = (all_y_true[idx], all_y_pred[idx],
                                all_y_err[idx], all_vars[idx])
    else:
        y_t, y_p, y_e, vars_ = all_y_true, all_y_pred, all_y_err, all_vars

    # --------------------------------------------------
    # Losses
    # --------------------------------------------------
    eps = 1e-8
    per_var = {'LS': {}, 'WLS': {}, 'MLE': {}, 'Chi': {}}
    uniq_vars = np.unique(vars_)
    for v in uniq_vars:
        sel = np.where(vars_ == v)[0]
        yt, yp, ye = y_t[sel], y_p[sel], y_e[sel]
        n = len(sel)

        per_var['LS'][v] = np.sum((yt - yp) ** 2) / n
        per_var['WLS'][v] = np.sum(((yt - yp) / std_dev[v]) ** 2) / n

        rel_err = (yt - yp) / np.maximum(np.abs(yt), eps)
        rel_unc = np.clip(ye / np.maximum(np.abs(yt), eps), 1e-3, 1e2)
        per_var['MLE'][v] = np.mean(
            0.5 * (np.log(2 * np.pi * rel_unc ** 2) + (rel_err ** 2) / (rel_unc ** 2))
        )
        per_var['Chi'][v] = np.sum(rel_err ** 2) / n

    LS = sum(per_var['LS'].values()) * N
    WLS = sum(per_var['WLS'].values()) * N
    MLE = sum(per_var['MLE'].values()) * N
    Chi = sum(per_var['Chi'].values()) * N

    # R^2 per response (for reporting)
    r2 = {}
    for v, me in varm.items():
        yta = np.array([m[0] for m in me])
        ypa = np.array([m[1] for m in me])
        if len(yta) > 1:
            r2[v] = 1 - np.sum((yta - ypa) ** 2) / np.sum((yta - yta.mean()) ** 2)
        else:
            r2[v] = np.nan

    metrics = {'LS': LS, 'WLS': WLS, 'MLE': MLE, 'Chi': Chi, 'R2_responses': r2}
    return data, metrics


