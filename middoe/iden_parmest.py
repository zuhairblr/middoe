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

def _initialise_theta_parameters(theta_min, theta_max, active_models):
    """Sample random theta values within bounds for each active model."""
    theta_parameters = {}
    for sv in active_models:
        theta_parameters[sv] = [np.random.uniform(lo, hi)
                                for lo, hi in zip(theta_min[sv], theta_max[sv])]
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

    theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models)
    theta_params_f = models['theta']

    x0_dict = {sv: [1.0] * len(theta_params_r[sv]) for sv in active_models}

    if bootstrap:
        return _bootstrap_runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                                 objf, x0_dict, method, data, system, models,
                                 logging, nboot, multi, varcov)

    if multi:
        nstarts = max(1, int(0.7 * mp.cpu_count()))
        return _multi_start_runner(active_models, theta_params_r, bound_max, bound_min, mutation,
                                   objf, method, data, system, models, logging, varcov, nstarts)

    return _runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                   objf, x0_dict, method, data, system, models, logging, varcov)


# ============================================================
# Multi-start machinery
# ============================================================

def _run_single_start(sv, thetac, thetamin, thetamax, thetas, objf,
                      method, data, system, models, logging, varcov):
    try:
        if logging:
            print(f"[{sv}] Sampled x0: {[1.0] * len(thetac)}")
        res = _runner([sv], {sv: thetac}, {sv: thetamax}, {sv: thetamin}, {sv: thetas},
                      objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov)
        if logging:
            print(f"[{sv}] Finished with result: {res}")
        return (sv, res)
    except Exception as e:
        import traceback, sys
        tb = ''.join(traceback.format_exception(*sys.exc_info()))
        return (sv, e, tb)


def _multi_start_runner(active_models, _unused, bmax, bmin, mutation, objf,
                        method, data, system, models, logging, varcov, nstarts=None):
    del _unused
    nwork = max(1, int(0.7 * mp.cpu_count()))
    nstarts = nstarts or max(2 * nwork, 10)

    allr = {sv: [] for sv in active_models}
    tasks = []
    for sv in active_models:
        for _ in range(nstarts):
            tasks.append((sv,
                          [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])],
                          bmin[sv], bmax[sv], mutation[sv], objf, method,
                          data, system, models, logging, varcov))

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
                               logging, varcov, nstarts=None):
    del _unused
    nstarts = nstarts or 20
    allr = {sv: [] for sv in active_models}

    for sv in active_models:
        for _ in range(nstarts):
            thetac = [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])]
            if logging:
                print(f"[{sv}][SERIAL] Trying theta: {thetac}")
            res = _runner([sv], {sv: thetac}, {sv: bmax[sv]}, {sv: bmin[sv]}, {sv: mutation[sv]},
                          objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov)
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
                      objf, x0, method, data, system, models, logging, multi, varcov,
                      boot_idx):
    """Run ONE bootstrap replicate with a fixed index vector 'boot_idx'."""
    models_boot = {**models,
                   '__bootstrap__': True,
                   '__boot_idx__': boot_idx}

    if multi:
        theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
        ref = _multi_start_runner_serial(active_models, theta_params_r, bmax, bmin,
                                         mutation, objf, method, data, system,
                                         models_boot, varcov, logging)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
                      x0, method, data, system, models_boot, logging, varcov)
    return ref


def _bootstrap_runner(active_models, theta_params, bmax, bmin, mutation,
                      objf, x0, method, data, system, models, logging, nboot, multi, varcov):
    # First, get a good reference fit (not bootstrap)
    nstarts = max(1, int(0.7 * mp.cpu_count()))
    theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
    ref = _multi_start_runner(active_models, theta_params_r, bmax, bmin,
                              mutation, objf, method, data, system, models,
                              logging, varcov, nstarts)

    # Pre-generate bootstrap index arrays
    N = _count_observations(data, system)
    rng = np.random.default_rng()
    boot_indices = [rng.integers(0, N, size=N, endpoint=False) for _ in range(nboot)]

    allP = {sv: [] for sv in active_models}
    bootR = {sv: [] for sv in active_models}

    args_common = (active_models, theta_params, bmax, bmin, mutation,
                   objf, x0, method, data, system, models, logging, multi, varcov)

    with ProcessPoolExecutor(max_workers=min(nboot, mp.cpu_count()), mp_context=CTX) as exe:
        futures = [exe.submit(_bootstrap_worker, *args_common, bidx) for bidx in boot_indices]
        for fut in as_completed(futures):
            res = fut.result()
            for sv in active_models:
                r = res.get(sv)
                if r and getattr(r, 'success', True):
                    # sanity check against exploding objectives
                    if ref[sv] is not None and r.fun <= 30 * ref[sv].fun:
                        if logging:
                            print(f'bootstrap WLS is {r.fun:.1f} and real one is {ref[sv].fun:.1f} for {sv}')
                        bootR[sv].append(r)
                        allP[sv].append(r.scpr)

    # Attach bootstrap samples & covariance
    for sv in active_models:
        ref_sv = ref.get(sv)
        if ref_sv is None:
            continue
        ref_sv.samples = bootR[sv]
        if bootR[sv]:
            matrix = np.stack([s.scpr for s in bootR[sv]])
            ref_sv.v = np.cov(matrix, rowvar=False, ddof=1)
        else:
            ref_sv.v = None

    return ref


# ============================================================
# Core optimizer runner
# ============================================================

def _runner(active_models, theta_params, bmax, bmin, mutation, objf,
            x0_dict, method, data, system, models, logging, varcov):
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
                               options={'maxiter': 100000, 'ftol': 1e-1, 'disp': True})
            elif method == 'SQP':
                res = minimize(_objective_function, x0, args=args, method='trust-constr',
                               bounds=bounds,
                               options={'maxiter': 500, 'xtol': 1e-2, 'gtol': 1e-2})
            elif method == 'NM':
                res = minimize(_objective_function, x0, args=args, method='Nelder-Mead',
                               options={'maxiter': 1e5, 'fatol': 1e-6, 'disp': False})
            elif method == 'BFGS':
                res = minimize(_objective_function, x0, args=args, method='BFGS',
                               options={'maxiter': 1e3, 'disp': False})
            elif method == 'DE':
                # Avoid nested pools: DE in worker uses 1 worker
                n_workers = 1 if _in_child() else -1
                res = differential_evolution(_objective_function, bounds, args=args,
                                             maxiter=1e3, popsize=18, tol=1e-8,
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
    start = time.time()
    _, m = _objective(theta, data, active, thetac, system, models,
                      bootstrap=bootstrap, boot_idx=boot_idx)
    if logging:
        print(f"Obj '{objf}'|{active[0]}|{time.time()-start:.3f}s")
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



#
# """
# Fast(er) parameter estimation with multistart + bootstrap.
#
# Drop-in replacement for your original file.
#
# Key changes:
# - Globals + initializer to avoid pickling huge dicts every task.
# - Cached metadata (std_dev, masks) so _objective stops rebuilding them.
# - Bootstrap: no multistart inside reps, warm-start + jitter, run in chunks.
# - Optional finite-diff Hessian instead of slow numdifftools (toggle USE_NUMDIFFTOOLS).
# """
#
# import os
# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message="Values in x were outside bounds during a minimize step, clipping to bounds"
# )
#
# # Limit BLAS oversubscription (optional but helps stability)
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
#
# import time
# import numpy as np
# import multiprocessing as mp
# from operator import attrgetter
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# # ------------------------------------------------------------------
# # Optional numdifftools (slow) – set flag False to skip
# # ------------------------------------------------------------------
# USE_NUMDIFFTOOLS = False
# if USE_NUMDIFFTOOLS:
#     from numdifftools import Hessian
#
# from scipy.optimize import minimize, differential_evolution
#
# # External project imports
# from middoe.krnl_simula import simula
# from middoe.iden_utils import _initialize_dictionaries
#
# # ------------------------------------------------------------------
# # Multiprocessing context (important on Windows / Jupyter)
# # ------------------------------------------------------------------
# try:
#     mp.set_start_method("spawn", force=True)
# except RuntimeError:
#     # already set
#     pass
# CTX = mp.get_context("spawn")
#
# def _in_child():
#     """Return True if running inside a worker process."""
#     return mp.current_process().name != "MainProcess"
#
# # ------------------------------------------------------------------
# # Globals in workers to avoid huge pickling costs
# # ------------------------------------------------------------------
# _GLOBALS = {}
# def _init_worker(data, system, models):
#     global _GLOBALS
#     _GLOBALS = dict(data=data, system=system, models=models)
#
# def _get_globals(data, system, models):
#     if mp.current_process().name == "MainProcess":
#         return data, system, models
#     return _GLOBALS['data'], _GLOBALS['system'], _GLOBALS['models']
#
# # ------------------------------------------------------------------
# # Metadata cache (std dev, masks, etc.)
# # ------------------------------------------------------------------
# _CACHE = {}
#
# def _build_cache(system, data):
#     import pandas as _pd
#     tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
#     ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]
#
#     std_dev = {
#         v: (cfg.get('unc', 1.0) if not np.isnan(cfg.get('unc', 1.0)) else 1.0)
#         for v, cfg in {**system['tvo'], **system['tio']}.items()
#         if cfg.get('meas', True)
#     }
#
#     mes_masks = {}
#     for name, sh in data.items():
#         mask_tv = {}
#         for v in tv_o:
#             xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
#             if xc in sh and yc in sh:
#                 m = ~sh[xc].isna().values
#                 if m.any():
#                     mask_tv[v] = (m, sh[xc][m].values)
#         mask_ti = {}
#         for v in ti_o:
#             yc = f"MES_Y:{v}"
#             mask_ti[v] = (yc in sh) and (not _pd.isna(sh[yc].iloc[0]))
#         mes_masks[name] = (mask_tv, mask_ti)
#
#     return dict(tv_o=tv_o, ti_o=ti_o, std_dev=std_dev, mes_masks=mes_masks)
#
# def _count_observations(data, system):
#     """
#     Count total number of scalar measurements (N) that will be used by _objective.
#     """
#     import pandas as _pd
#     tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
#     ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]
#
#     N = 0
#     for sh in data.values():
#         # time-varying outputs
#         for v in tv_o:
#             xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
#             if xc in sh and yc in sh:
#                 mask = ~sh[xc].isna().values
#                 N += int(mask.sum())
#         # time-invariant outputs
#         for v in ti_o:
#             yc = f"MES_Y:{v}"
#             if yc in sh and not _pd.isna(sh[yc].iloc[0]):
#                 N += 1
#     return N
#
# # ============================================================
# # Helpers
# # ============================================================
#
# def _initialise_theta_parameters(theta_min, theta_max, active_models):
#     """Sample random theta values within bounds for each active model."""
#     theta_parameters = {}
#     for sv in active_models:
#         theta_parameters[sv] = [np.random.uniform(lo, hi)
#                                 for lo, hi in zip(theta_min[sv], theta_max[sv])]
#     return theta_parameters
#
# # ============================================================
# # Public entry point
# # ============================================================
#
# def parmest(system, models, iden_opt, data, case=None):
#     """
#     Parameter estimation main entry.
#     """
#     _initialize_dictionaries(models, iden_opt)
#
#     # Build / store cache once
#     _CACHE['meta'] = _build_cache(system, data)
#
#     active_models = models['can_m']
#     bound_max, bound_min = models['t_u'], models['t_l']
#     mutation = models['mutation']
#
#     method   = iden_opt['meth']
#     objf     = iden_opt['ob']
#     multi    = iden_opt.get('ms', False)
#
#     bootstrap = (iden_opt.get('var-cov') == 'B')
#     varcov    = iden_opt.get('var-cov', 'H')
#     nboot     = iden_opt.get('nboot', 100)
#     logging   = iden_opt.get('log', False)
#
#     theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models)
#     theta_params_f = models['theta']
#     x0_dict = {sv: [1.0] * len(theta_params_r[sv]) for sv in active_models}
#
#     if bootstrap:
#         return _bootstrap_runner(active_models, theta_params_f, bound_max, bound_min, mutation,
#                                  objf, x0_dict, method, data, system, models,
#                                  logging, nboot, multi, varcov)
#
#     if multi:
#         nstarts = max(1, int(0.7 * mp.cpu_count()))
#         return _multi_start_runner(active_models, theta_params_r, bound_max, bound_min, mutation,
#                                    objf, method, data, system, models, logging, varcov, nstarts)
#
#     return _runner(active_models, theta_params_f, bound_max, bound_min, mutation,
#                    objf, x0_dict, method, data, system, models, logging, varcov)
#
# # ============================================================
# # Multi-start machinery
# # ============================================================
#
# def _run_single_start(sv, thetac, thetamin, thetamax, thetas, objf,
#                       method, data, system, models, logging, varcov):
#     try:
#         if logging:
#             print(f"[{sv}] Sampled x0: {[1.0] * len(thetac)}")
#         res = _runner([sv], {sv: thetac}, {sv: thetamax}, {sv: thetamin}, {sv: thetas},
#                       objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov)
#         if logging:
#             print(f"[{sv}] Finished with result: {res}")
#         return (sv, res)
#     except Exception as e:
#         import traceback, sys
#         tb = ''.join(traceback.format_exception(*sys.exc_info()))
#         return (sv, e, tb)
#
# def _multi_start_runner(active_models, _unused, bmax, bmin, mutation, objf,
#                         method, data, system, models, logging, varcov, nstarts=None):
#     del _unused
#     nwork   = max(1, int(0.7 * mp.cpu_count()))
#     nstarts = nstarts or max(2 * nwork, 10)
#
#     allr = {sv: [] for sv in active_models}
#     tasks = []
#     for sv in active_models:
#         for _ in range(nstarts):
#             tasks.append((sv,
#                           [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])],
#                           bmin[sv], bmax[sv], mutation[sv], objf, method,
#                           data, system, models, logging, varcov))
#
#     with ProcessPoolExecutor(max_workers=nwork,
#                              mp_context=CTX,
#                              initializer=_init_worker,
#                              initargs=(data, system, models)) as exe:
#         for fut in as_completed([exe.submit(_run_single_start, *t) for t in tasks]):
#             out = fut.result()
#             # Crash in worker?
#             if len(out) == 3 and isinstance(out[1], Exception):
#                 sv, e, tb = out
#                 print(f"[{sv}] worker crashed:\n{tb}")
#                 continue
#             sv, res = out
#             if res and res.get(sv):
#                 allr[sv].append(res[sv])
#
#     return {
#         sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
#              if any(r.success for r in allr[sv]) else None)
#         for sv in active_models
#     }
#
# def _multi_start_runner_serial(active_models, _unused, bmax, bmin,
#                                mutation, objf, method, data, system, models,
#                                logging, varcov, nstarts=None):
#     del _unused
#     nstarts = nstarts or 20
#     allr = {sv: [] for sv in active_models}
#
#     for sv in active_models:
#         for _ in range(nstarts):
#             thetac = [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])]
#             if logging:
#                 print(f"[{sv}][SERIAL] Trying theta: {thetac}")
#             res = _runner([sv], {sv: thetac}, {sv: bmax[sv]}, {sv: bmin[sv]}, {sv: mutation[sv]},
#                           objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov)
#             if res and res.get(sv):
#                 allr[sv].append(res[sv])
#
#     return {
#         sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
#              if any(r.success for r in allr[sv]) else None)
#         for sv in active_models
#     }
#
# # ============================================================
# # Bootstrap machinery
# # ============================================================
#
# def _bootstrap_worker(active_models, theta_params, bmax, bmin, mutation,
#                       objf, x0, method, data, system, models, logging, multi, varcov,
#                       boot_idx):
#     """Run ONE bootstrap replicate with a fixed index vector 'boot_idx'."""
#     data, system, models = _get_globals(data, system, models)
#     models_boot = {**models,
#                    '__bootstrap__': True,
#                    '__boot_idx__': boot_idx}
#
#     # Single-start inside bootstrap; jitter to avoid same minimum
#     rng = np.random.default_rng()
#     x0_jitter = {}
#     for sv in active_models:
#         base = np.array(x0[sv], float)
#         x0_jitter[sv] = base * (1.0 + rng.normal(0.0, 0.05, size=base.size))
#
#     ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
#                   x0_jitter, method, data, system, models_boot, logging, varcov)
#     # annotate
#     for sv in active_models:
#         if ref.get(sv) is not None:
#             ref[sv].boot_idx = boot_idx
#     return ref
#
# def _bootstrap_chunk(args_common, idx_list):
#     """Run several bootstrap replicates in one worker to amortize overhead."""
#     (active_models, theta_params, bmax, bmin, mutation,
#      objf, x0, method, data, system, models, logging, multi, varcov) = args_common
#
#     last = None
#     out  = []
#     rng  = np.random.default_rng()
#
#     for bidx in idx_list:
#         # warm start from last solution if available
#         if last:
#             x0_local = {sv: last[sv].x * (1 + rng.normal(0, 0.03, len(last[sv].x)))
#                         for sv in active_models if last.get(sv) is not None}
#             # fall back if missing
#             for sv in active_models:
#                 if sv not in x0_local:
#                     x0_local[sv] = x0[sv]
#         else:
#             x0_local = x0
#
#         res = _bootstrap_worker(active_models, theta_params, bmax, bmin, mutation,
#                                 objf, x0_local, method, data, system, models,
#                                 logging, False, varcov, bidx)
#         last = {sv: r for sv, r in res.items() if r is not None}
#         out.append(res)
#     return out
#
# def _bootstrap_runner(active_models, theta_params, bmax, bmin, mutation,
#                       objf, x0, method, data, system, models, logging, nboot, multi, varcov):
#     # First, get a good reference fit (not bootstrap)
#     nstarts = max(1, int(0.7 * mp.cpu_count()))
#     theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
#     ref = _multi_start_runner(active_models, theta_params_r, bmax, bmin,
#                               mutation, objf, method, data, system, models,
#                               logging, varcov, nstarts)
#
#     # Pre-generate bootstrap index arrays
#     N = _count_observations(data, system)
#     rng = np.random.default_rng()
#     boot_indices = [rng.choice(N, size=N, replace=True) for _ in range(nboot)]
#
#     allP = {sv: [] for sv in active_models}
#     bootR = {sv: [] for sv in active_models}
#
#     args_common = (active_models, theta_params, bmax, bmin, mutation,
#                    objf, x0, method, data, system, models, logging, multi, varcov)
#
#     chunk = max(1, nboot // (mp.cpu_count() * 2))  # small chunk size
#     with ProcessPoolExecutor(max_workers=min(nboot, mp.cpu_count()),
#                              mp_context=CTX,
#                              initializer=_init_worker,
#                              initargs=(data, system, models)) as exe:
#         futures = []
#         for i in range(0, nboot, chunk):
#             futures.append(exe.submit(_bootstrap_chunk, args_common,
#                                       boot_indices[i:i+chunk]))
#         for fut in as_completed(futures):
#             res_list = fut.result()
#             for res in res_list:
#                 for sv in active_models:
#                     r = res.get(sv)
#                     if r and getattr(r, 'success', True):
#                         # sanity check against exploding objectives
#                         if ref[sv] is not None and r.fun <= 30 * ref[sv].fun:
#                             if logging:
#                                 print(f'bootstrap WLS is {r.fun:.1f} and real one is {ref[sv].fun:.1f} for {sv}')
#                             # avoid duplicate parameter vectors
#                             if not any(np.allclose(r.scpr, prev.scpr, rtol=1e-4, atol=1e-6) for prev in bootR[sv]):
#                                 bootR[sv].append(r)
#                                 allP[sv].append(r.scpr)
#
#     # Attach bootstrap samples & covariance
#     for sv in active_models:
#         ref_sv = ref.get(sv)
#         if ref_sv is None:
#             continue
#         ref_sv.samples = bootR[sv]
#         if bootR[sv]:
#             matrix = np.stack([s.scpr for s in bootR[sv]])
#             ref_sv.v = np.cov(matrix, rowvar=False, ddof=1)
#         else:
#             ref_sv.v = None
#
#     return ref
#
# # ============================================================
# # Core optimizer runner
# # ============================================================
#
# def _runner(active_models, theta_params, bmax, bmin, mutation, objf,
#             x0_dict, method, data, system, models, logging, varcov):
#     data, system, models = _get_globals(data, system, models)
#     if 'meta' not in _CACHE:
#         _CACHE['meta'] = _build_cache(system, data)
#
#     results = {}
#     for sv in active_models:
#         thetac = np.array(theta_params[sv], float)
#         x0     = np.array(x0_dict[sv], float)
#
#         thetamax = np.array(bmax[sv]) / thetac
#         thetamin = np.array(bmin[sv]) / thetac
#
#         mask   = np.array(mutation[sv], bool)
#         narrow = 1e-40
#
#         bounds = [((x * (1 - narrow), x * (1 + narrow)) if not m else (tmin, tmax))
#                   for x, m, tmin, tmax in zip(x0, mask, thetamin, thetamax)]
#
#         bootstrap_flag = models.get('__bootstrap__', False)
#         boot_idx       = models.get('__boot_idx__', None)
#
#         args = (data, [sv], thetac, system, models, logging, objf,
#                 bootstrap_flag, boot_idx)
#
#         try:
#             if method == 'SLSQP':
#                 res = minimize(_objective_function, x0, args=args, method='SLSQP',
#                                bounds=bounds,
#                                options={'maxiter': 2000, 'ftol': 1e-2, 'disp': False})
#             elif method == 'SQP':
#                 res = minimize(_objective_function, x0, args=args, method='trust-constr',
#                                bounds=bounds,
#                                options={'maxiter': 500, 'xtol': 1e-2, 'gtol': 1e-2})
#             elif method == 'NM':
#                 res = minimize(_objective_function, x0, args=args, method='Nelder-Mead',
#                                options={'maxiter': 1e5, 'fatol': 1e-6, 'disp': False})
#             elif method == 'BFGS':
#                 res = minimize(_objective_function, x0, args=args, method='BFGS',
#                                options={'maxiter': 1e3, 'disp': False})
#             elif method == 'DE':
#                 # Avoid nested pools: DE in worker uses 1 worker
#                 n_workers = 1 if _in_child() else -1
#                 res = differential_evolution(_objective_function, bounds, args=args,
#                                              maxiter=600, popsize=12, tol=1e-6,
#                                              strategy='best1bin', mutation=(0.5, 1.3),
#                                              recombination=0.7, polish=False,
#                                              updating='deferred', workers=n_workers)
#             else:
#                 raise ValueError(f"Unknown method '{method}'")
#         except Exception as e:
#             if logging:
#                 print(f"[{sv}] optimize error: {e}")
#             continue
#
#         # Scale back to original theta
#         res.scpr = res.x * thetac
#
#         # Hessian & inverse (only on the reference fits, not bootstrap replicates)
#         if not bootstrap_flag and varcov == 'H':
#             try:
#                 if USE_NUMDIFFTOOLS:
#                     loss = lambda t: _objective_function(t, data, [sv], thetac, system, models,
#                                                          logging, objf, False, None)
#                     H = Hessian(loss, step=1e-4, method='central', order=2)(res.x)
#                 else:
#                     H = _hess_fd(res.x, _objective_function, data, [sv], thetac, system, models,
#                                  logging, objf, False, None)
#
#                 S  = np.diag(1.0 / thetac.astype(float))
#                 Hs = S @ H @ S
#                 res.hessian = Hs
#
#                 D = np.diag(thetac.astype(float))
#                 inv = D @ np.linalg.pinv(Hs) @ D
#                 res.hess_inv = inv
#             except Exception as e:
#                 if logging:
#                     print(f"[{sv}] Hessian/covariance failed: {e}")
#                 res.hessian  = None
#                 res.hess_inv = None
#
#         results[sv] = res
#
#     return results
#
# # ============================================================
# # Objective wrappers
# # ============================================================
#
# def _objective_function(theta, data, active, thetac, system, models,
#                         logging, objf, bootstrap, boot_idx):
#     start = time.time()
#     _, m = _objective(theta, data, active, thetac, system, models,
#                       bootstrap=bootstrap, boot_idx=boot_idx)
#     if logging:
#         print(f"Obj '{objf}'|{active[0]}|{time.time()-start:.3f}s")
#     return {'LS': m['LS'], 'WLS': m['WLS'], 'MLE': m['MLE'], 'Chi': m['Chi']}[objf]
#
# def _objective(theta, data, active, thetac, system, models,
#                bootstrap=False, boot_idx=None):
#     import pandas as _pd
#
#     theta = theta.tolist()
#     tv_i, ti_i = list(system['tvi']), list(system['tii'])
#     meta   = _CACHE['meta']
#     tv_o, ti_o = meta['tv_o'], meta['ti_o']
#     std_dev    = meta['std_dev']
#
#     phisc, phitsc, tsc = {v: 1 for v in ti_i}, {v: 1 for v in tv_i}, 1
#     solver = active[0]
#
#     global_y_true, global_y_pred, global_y_err, varm = {}, {}, {}, {}
#
#     # --------------------------------------------------
#     # Simulate & gather measurements
#     # --------------------------------------------------
#     for name, sh in data.items():
#         mask_tv, mask_ti = meta['mes_masks'][name]
#
#         t_all = np.unique(sh["X:all"].dropna().values)
#
#         # sweeps (time varying inputs)
#         swps = {}
#         for v in tv_i:
#             tk, lk = f"{v}t", f"{v}l"
#             if tk in sh and lk in sh:
#                 ta, la = sh[tk].dropna().values, sh[lk].dropna().values
#                 if ta.size and la.size:
#                     swps[tk], swps[lk] = ta, la
#
#         # constants (time invariant inputs)
#         ti_in = {v: sh.get(v, _pd.Series([np.nan])).iloc[0] for v in ti_i}
#         tv_in = {v: sh.get(v, _pd.Series()).dropna().values for v in tv_i}
#         cvp   = {v: sh[f"CVP:{v}"].iloc[0] for v in system['tvi']}
#
#         tvs, tis, _ = simula(t_all, swps, ti_in, phisc, phitsc, tsc,
#                              theta, thetac, cvp, tv_in, solver, system, models)
#
#         # time-varying outputs
#         for v in tv_o:
#             if v not in mask_tv:
#                 continue
#             m, _times = mask_tv[v]
#             if not m.any():
#                 continue
#             xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
#             times = sh[xc][m].values
#             y_t   = sh[yc][m].values
#
#             idx = np.isin(t_all, times)
#             y_p = np.array(tvs[v])[idx]
#
#             ye_col = f"MES_E:{v}"
#             y_e = (sh[ye_col][m].values if ye_col in sh
#                    else np.full_like(y_t, std_dev[v]))
#
#             global_y_true.setdefault(v, []).extend(y_t.tolist())
#             global_y_pred.setdefault(v, []).extend(y_p.tolist())
#             global_y_err.setdefault(v, []).extend(y_e.tolist())
#
#             for yt, yp, ye in zip(y_t, y_p, y_e):
#                 varm.setdefault(v, []).append((yt, yp, ye))
#
#         # time-invariant outputs
#         for v in ti_o:
#             if not mask_ti.get(v, False):
#                 continue
#             yc = f"MES_Y:{v}"
#             y_t = sh[yc].iloc[0]
#             y_p = tis.get(v, np.nan)
#             if np.isnan(y_t) or np.isnan(y_p):
#                 continue
#             sigma = std_dev[v]
#             global_y_true.setdefault(v, []).append(y_t)
#             global_y_pred.setdefault(v, []).append(y_p)
#             global_y_err.setdefault(v, []).append(sigma)
#             varm.setdefault(v, []).append((y_t, y_p, sigma))
#
#     # --------------------------------------------------
#     # Flatten
#     # --------------------------------------------------
#     all_y_true, all_y_pred, all_y_err, all_vars = [], [], [], []
#     for v, ytl in global_y_true.items():
#         ypl = global_y_pred[v]
#         yel = global_y_err[v]
#         n = min(len(ytl), len(ypl), len(yel))
#         all_y_true += ytl[:n]
#         all_y_pred += ypl[:n]
#         all_y_err  += yel[:n]
#         all_vars   += [v] * n
#
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
#     all_y_err  = np.array(all_y_err)
#     all_vars   = np.array(all_vars)
#     N = len(all_y_true)
#
#     # --------------------------------------------------
#     # Bootstrap resampling (fixed index)
#     # --------------------------------------------------
#     if bootstrap:
#         if boot_idx is None:
#             raise ValueError("bootstrap=True but boot_idx is None.")
#         if len(boot_idx) != N:
#             raise ValueError(f"bootstrap index length {len(boot_idx)} != N {N}")
#         sel = boot_idx
#         y_t, y_p, y_e, vars_ = (all_y_true[sel], all_y_pred[sel],
#                                 all_y_err[sel], all_vars[sel])
#     else:
#         y_t, y_p, y_e, vars_ = all_y_true, all_y_pred, all_y_err, all_vars
#
#     # --------------------------------------------------
#     # Losses
#     # --------------------------------------------------
#     eps = 1e-8
#     per_var = {'LS': {}, 'WLS': {}, 'MLE': {}, 'Chi': {}}
#     uniq_vars = np.unique(vars_)
#     for v in uniq_vars:
#         sel = np.where(vars_ == v)[0]
#         yt, yp, ye = y_t[sel], y_p[sel], y_e[sel]
#         n = len(sel)
#
#         per_var['LS'][v]  = np.sum((yt - yp) ** 2) / n
#         per_var['WLS'][v] = np.sum(((yt - yp) / std_dev[v]) ** 2) / n
#
#         rel_err = (yt - yp) / np.maximum(np.abs(yt), eps)
#         rel_unc = np.clip(ye / np.maximum(np.abs(yt), eps), 1e-3, 1e2)
#         per_var['MLE'][v] = np.mean(
#             0.5 * (np.log(2 * np.pi * rel_unc ** 2) + (rel_err ** 2) / (rel_unc ** 2))
#         )
#         per_var['Chi'][v] = np.sum(rel_err ** 2) / n
#
#     LS  = sum(per_var['LS'].values())  * N
#     WLS = sum(per_var['WLS'].values()) * N
#     MLE = sum(per_var['MLE'].values()) * N
#     Chi = sum(per_var['Chi'].values()) * N
#
#     metrics = {'LS': LS, 'WLS': WLS, 'MLE': MLE, 'Chi': Chi}
#     return data, metrics
#
# # ============================================================
# # Finite difference Hessian (fallback)
# # ============================================================
#
# def _hess_fd(x, f, *args, eps=1e-4):
#     n = len(x)
#     H = np.zeros((n, n))
#     fx = f(x, *args)
#     for i in range(n):
#         xi = x.copy(); xi[i] += eps
#         fxi = f(xi, *args)
#         xi[i] -= 2 * eps
#         fxd = f(xi, *args)
#         H[i, i] = (fxi - 2 * fx + fxd) / (eps ** 2)
#         for j in range(i+1, n):
#             xij = x.copy(); xij[i] += eps; xij[j] += eps
#             fij = f(xij, *args)
#
#             xij[j] -= 2 * eps
#             fij2 = f(xij, *args)
#
#             xij[i] -= 2 * eps
#             fij3 = f(xij, *args)
#
#             xij[j] += 2 * eps
#             fij4 = f(xij, *args)
#
#             H[i, j] = H[j, i] = (fij - fij2 - fij4 + fij3) / (4 * eps ** 2)
#     return H
