# #iden_parmest.py
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
#
# from scipy.stats import truncnorm
# import numpy as np
# import copy
# import multiprocessing as mp
# from operator import attrgetter
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from numdifftools import Hessian
# from scipy.optimize import minimize, differential_evolution
# from middoe.krnl_simula import simula
# from middoe.iden_utils import _initialize_dictionaries
# from middoe.log_utils import  read_excel
#
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
#
# def _in_child():
#     """Return True if running inside a worker process."""
#     return mp.current_process().name != "MainProcess"
#
#
# # ============================================================
# # Helpers
# # ============================================================
#
# def _initialise_theta_parameters(theta_min, theta_max, active_models,  mutation=None, theta_fixed=None):
#     """
#     Initialize theta parameters for each active model.
#
#     This function samples theta values within the specified bounds for each active model.
#     If a mutation mask is provided, inactive parameters are fixed to the values in `theta_fixed`.
#
#     Parameters
#     ----------
#     theta_min : dict
#         Dictionary containing the lower bounds for theta values for each solver.
#     theta_max : dict
#         Dictionary containing the upper bounds for theta values for each solver.
#     active_models : list
#         List of solver keys for which theta parameters need to be initialized.
#     mutation : dict, optional
#         Dictionary of boolean masks per solver indicating active parameters (True=active).
#         If not provided, all parameters are considered active.
#     theta_fixed : dict, optional
#         Dictionary of fixed theta values per solver. Inactive parameters (as per `mutation`)
#         are set to these fixed values.
#
#     Returns
#     -------
#     dict
#         A dictionary where each key corresponds to a solver, and the value is a list of
#         initialized theta parameters for that solver.
#     """
#     theta_parameters = {}
#     for sv in active_models:
#         if mutation is not None and theta_fixed is not None:
#             mask = np.array(mutation[sv], dtype=bool)
#             theta_parameters[sv] = [
#                 np.random.uniform(lo, hi) if m else theta_fixed[sv][i]
#                 for i, (lo, hi, m) in enumerate(zip(theta_min[sv], theta_max[sv], mask))
#             ]
#         else:
#             theta_parameters[sv] = [
#                 np.random.uniform(lo, hi) for lo, hi in zip(theta_min[sv], theta_max[sv])
#             ]
#     return theta_parameters
#
#
# def _count_observations(data, system):
#     """
#     Count total number of scalar measurements (N) in 'data' that will be used by _objective.
#     Does NOT require simulation, just metadata.
#     """
#     import pandas as _pd
#
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
#
#         # time-invariant outputs
#         for v in ti_o:
#             yc = f"MES_Y:{v}"
#             if yc in sh and not _pd.isna(sh[yc].iloc[0]):
#                 N += 1
#
#     return N
#
#
# # ============================================================
# # Public entry point
# # ============================================================
#
# def parmest(system, models, iden_opt, case=None):
#     """
#     Perform parameter estimation for a given system and models.
#
#     This function serves as the main entry point for parameter estimation. It initializes
#     the required parameters, handles different estimation cases, and executes the estimation
#     process using single or multi-start optimization, or bootstrap resampling.
#
#     Parameters
#     ----------
#     system : dict
#         Dictionary containing the system configuration, including time intervals, variable bounds,
#         and constraints.
#     models : dict
#         Dictionary containing model-related data, such as active solvers, parameter bounds,
#         and mutation settings.
#     iden_opt : dict
#         Dictionary of identification options, including optimization method, objective function,
#         and other settings.
#     data : dict
#         Dictionary containing the experimental data used for parameter estimation.
#     case : str, optional
#         Specifies the estimation case. Default is None. Supported cases include:
#         - 'freeze': Freezes mutation settings.
#         - 'strov': Uses starting theta values from the models.
#
#     Returns
#     -------
#     dict
#         A dictionary containing the results of the parameter estimation, including optimized
#         parameters, objective function values, and additional metrics.
#
#     Raises
#     ------
#     ValueError
#         If invalid options or configurations are provided.
#
#     Notes
#     -----
#     The function supports multi-start optimization and bootstrap resampling for robust
#     parameter estimation. It also handles logging and variance-covariance matrix computation
#     based on the specified options.
#     """
#     data = read_excel()
#     if case != 'freeze' and 'mutation' in models:
#         for solver in models['mutation']:
#             models['mutation'][solver] = [True] * len(models['mutation'][solver])
#     _initialize_dictionaries(models, iden_opt)
#
#     active_models = models['can_m']
#     bound_max, bound_min = models['t_u'], models['t_l']
#     mutation = models['mutation']
#
#     method = iden_opt['meth']
#     objf = iden_opt['ob']
#     multi = iden_opt.get('ms', False)
#
#     bootstrap = (iden_opt.get('var-cov') == 'B')
#     varcov= iden_opt.get('var-cov','H')
#     nboot = iden_opt.get('nboot', 100)
#     logging = iden_opt.get('log', False)
#     maxit = iden_opt.get('maxit', 1000)
#     tol = iden_opt.get('tol', 1e-6)
#
#     if case == 'strov' and 'thetastart' in models:
#         theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models, mutation=mutation, theta_fixed=models['thetastart'])
#         theta_params_f = models['thetastart']
#     else:
#         theta_params_r = _initialise_theta_parameters(bound_min, bound_max, active_models, mutation=mutation, theta_fixed=models['theta'])
#         theta_params_f = models['theta']
#
#     x0_dict = {sv: [1.0] * len(theta_params_r[sv]) for sv in active_models}
#
#     if bootstrap:
#         return _bootstrap_runner(active_models, theta_params_f, bound_max, bound_min, mutation,
#                                  objf, x0_dict, method, data, system, models,
#                                  logging, nboot, multi, varcov, maxit, tol)
#
#     if multi:
#         nstarts = max(1, int(0.7 * mp.cpu_count()))
#         return _multi_start_runner(active_models, theta_params_r, bound_max, bound_min, mutation,
#                                    objf, method, data, system, models, logging, varcov, maxit, tol, nstarts)
#
#     return _runner(active_models, theta_params_f, bound_max, bound_min, mutation,
#                    objf, x0_dict, method, data, system, models, logging, varcov, maxit, tol)
#
#
# # ============================================================
# # Multi-start machinery
# # ============================================================
#
# def _run_single_start(sv, thetac, thetamin, thetamax, thetas, objf,
#                       method, data, system, models, logging, varcov, maxit, tol):
#     try:
#         if logging:
#             print(f"[{sv}] Sampled x0: {[1.0] * len(thetac)}")
#         res = _runner([sv], {sv: thetac}, {sv: thetamax}, {sv: thetamin}, {sv: thetas},
#                       objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov, maxit, tol)
#         if logging:
#             print(f"[{sv}] Finished with result: {res}")
#         return (sv, res)
#     except Exception as e:
#         import traceback, sys
#         tb = ''.join(traceback.format_exception(*sys.exc_info()))
#         return (sv, e, tb)
#
#
# def _multi_start_runner(active_models, _unused, bmax, bmin, mutation, objf,
#                         method, data, system, models, logging, varcov, maxit, tol, nstarts=None):
#     del _unused
#     nwork = max(1, int(0.7 * mp.cpu_count()))
#     nstarts = nstarts or max(2 * nwork, 10)
#
#     allr = {sv: [] for sv in active_models}
#     tasks = []
#     for sv in active_models:
#         for _ in range(nstarts):
#             thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=models['theta'])[sv]
#             tasks.append((sv, thetac, bmin[sv], bmax[sv], mutation[sv], objf, method,
#                           data, system, models, logging, varcov, maxit, tol))
#
#     with ProcessPoolExecutor(max_workers=nwork, mp_context=CTX) as exe:
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
#
# def _multi_start_runner_serial(active_models, _unused, bmax, bmin,
#                                mutation, objf, method, data, system, models,
#                                logging, varcov, maxit, tol, nstarts=None):
#     del _unused
#     nstarts = nstarts or 20
#     allr = {sv: [] for sv in active_models}
#
#     for sv in active_models:
#         for _ in range(nstarts):
#             # thetac = [np.random.uniform(lo, hi) for lo, hi in zip(bmin[sv], bmax[sv])]
#             thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=models['theta'])[sv]
#
#             if logging:
#                 print(f"[{sv}][SERIAL] Trying theta: {thetac}")
#             res = _runner([sv], {sv: thetac}, {sv: bmax[sv]}, {sv: bmin[sv]}, {sv: mutation[sv]},
#                           objf, {sv: [1.0] * len(thetac)}, method, data, system, models, logging, varcov, maxit, tol)
#             if res and res.get(sv):
#                 allr[sv].append(res[sv])
#
#     return {
#         sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
#              if any(r.success for r in allr[sv]) else None)
#         for sv in active_models
#     }
#
#
# # ============================================================
# # Bootstrap machinery
# # ============================================================
#
# def _bootstrap_worker(active_models, theta_params, bmax, bmin, mutation,
#                       objf, x0, method, data, system, models, logging, multi, varcov, maxit, tol,
#                       boot_idx):
#     """Run ONE bootstrap replicate with a fixed index vector 'boot_idx'."""
#     models_boot = {**models,
#                    '__bootstrap__': True,
#                    '__boot_idx__': boot_idx}
#
#     if multi:
#         theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models, mutation=mutation, theta_fixed=models['theta'])
#         ref = _multi_start_runner_serial(active_models, theta_params_r, bmax, bmin,
#                                          mutation, objf, method, data, system,
#                                          models_boot, logging, varcov, maxit, tol)
#     else:
#         ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
#                       x0, method, data, system, models_boot, logging, varcov, maxit, tol)
#     return ref
#
#
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
#     # Corr matrix handling — ensure it's always 2D
#     with np.errstate(invalid='ignore', divide='ignore'):
#         corr_emp = np.corrcoef(X.T)
#         if np.isscalar(corr_emp):  # edge case: only one active parameter
#             corr_emp = np.array([[1.0]])
#         else:
#             corr_emp = np.nan_to_num(corr_emp, nan=0.0)
#             np.fill_diagonal(corr_emp, 1.0)
#
#     sigma_trunc = np.sqrt(var_trunc)
#     cov_trunc = np.outer(sigma_trunc, sigma_trunc) * corr_emp
#
#     return mu_trunc, cov_trunc
#
#
# def compute_trunc_mean_cov_from_empirical(resultpr, solver='M', scpr_key='scpr', include_X=True):
#     """
#     Compute truncated mean and covariance from empirical bootstrap samples.
#     """
#     resultpr2 = copy.deepcopy(resultpr)
#     samples = resultpr2[solver]['samples']
#     X = np.array([np.ravel(getattr(s, scpr_key)) for s in samples], dtype=np.float64)
#
#     mu, cov = truncated_moments_from_data(X)
#
#     resultpr2[solver]['scpr'] = mu
#     resultpr2[solver]['v'] = cov
#     if include_X:
#         resultpr2[solver]['X'] = X
#
#     return resultpr2
#
#
# def _bootstrap_runner(active_models, theta_params, bmax, bmin, mutation,
#                       objf, x0, method, data, system, models, logging,
#                       nboot, multi, varcov, maxit, tol):
#
#     if multi:
#         trunc_mc_samps = nboot
#         nstarts = max(1, int(0.7 * mp.cpu_count()))
#         theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models, mutation=mutation, theta_fixed=models['theta'])
#         ref = _multi_start_runner(active_models, theta_params_r, bmax, bmin,
#                                   mutation, objf, method, data, system, models,
#                                   logging, varcov, maxit, tol, nstarts)
#     else:
#         ref = _runner(active_models, theta_params, bmax, bmin, mutation,
#                       objf, x0, method, data, system, models, logging,
#                       varcov, maxit, tol)
#
#     # Bootstrap sampling
#     N = _count_observations(data, system)
#     rng = np.random.default_rng()
#     boot_indices = [rng.integers(0, N, size=N, endpoint=False) for _ in range(nboot)]
#
#     allP = {sv: [] for sv in active_models}
#     bootR = {sv: [] for sv in active_models}
#
#     args_common = (active_models, theta_params, bmax, bmin, mutation,
#                    objf, x0, method, data, system, models, logging,
#                    multi, varcov, maxit, tol)
#
#     with ProcessPoolExecutor(max_workers=min(nboot, mp.cpu_count()), mp_context=CTX) as exe:
#         futures = [exe.submit(_bootstrap_worker, *args_common, bidx) for bidx in boot_indices]
#         for fut in as_completed(futures):
#             res = fut.result()
#             for sv in active_models:
#                 r = res.get(sv)
#                 if r and getattr(r, 'success', True):
#                     if ref[sv] is not None and r.fun <= 2 * ref[sv].fun:
#                         if logging:
#                             print(f'bootstrap WLS is {r.fun:.1f} and real one is {ref[sv].fun:.1f} for {sv}')
#                         bootR[sv].append(r)
#                         allP[sv].append(r.scpr)
#
#     # Create full resultpr with both raw and trunc-normal stats
#     resultpr = {}
#     for sv in active_models:
#         ref_sv = ref.get(sv)
#         if ref_sv is None:
#             continue
#
#         # Attach bootstrap samples and stats
#         ref_sv.samples = bootR[sv]
#         if bootR[sv]:
#             X_full = np.stack([s.scpr for s in bootR[sv]])  # shape: (nboot, nparams)
#             mu_raw = np.mean(X_full, axis=0)
#
#             # Masked (False) parameters are removed for covariance estimation
#             mask = np.array(mutation[sv], dtype=bool)
#             X_masked = X_full[:, mask]  # shape: (nboot, n_active_params)
#
#             # Compute truncated stats only for unmasked parameters
#             mu_trunc_masked, cov_trunc_masked = truncated_moments_from_data(X_masked)
#
#             # Build full-length covariance matrix with zeros in masked rows/cols
#             v_full = np.zeros((X_full.shape[1], X_full.shape[1]), dtype=np.float64)
#             v_full[np.ix_(mask, mask)] = cov_trunc_masked
#
#             ref_sv.scpr_raw = ref_sv.scpr
#             ref_sv.X = X_full  # full sample matrix
#             ref_sv.scpr = mu_raw  # full mean, including fixed params
#             ref_sv.v = v_full  # full matrix, only active filled
#             ref_sv.v_raw = np.cov(X_full, rowvar=False, ddof=1)
#
#
#         else:
#             ref_sv.X = None
#             ref_sv.scpr = None
#             ref_sv.v = None
#             ref_sv.scpr_raw = None
#             ref_sv.v_raw = None
#
#     # Return the updated ref (which is now the enriched resultpr)
#     return ref
#
# # ============================================================
# # Core optimizer runner
# # ============================================================
#
# def _runner(active_models, theta_params, bmax, bmin, mutation, objf,
#             x0_dict, method, data, system, models, logging, varcov, maxit, tol):
#     results = {}
#     for sv in active_models:
#         thetac = np.array(theta_params[sv], float)
#         x0 = np.array(x0_dict[sv], float)
#
#         thetamax = np.array(bmax[sv]) / thetac
#         thetamin = np.array(bmin[sv]) / thetac
#
#         mask = np.array(mutation[sv], bool)
#         narrow = 1e-40
#
#         bounds = [((x * (1 - narrow), x * (1 + narrow)) if not m else (tmin, tmax))
#                   for x, m, tmin, tmax in zip(x0, mask, thetamin, thetamax)]
#
#         bootstrap_flag = models.get('__bootstrap__', False)
#         boot_idx = models.get('__boot_idx__', None)
#
#         args = (data, [sv], thetac, system, models, logging, objf,
#                 bootstrap_flag, boot_idx)
#
#         try:
#             if method == 'SLSQP':
#                 res = minimize(_objective_function, x0, args=args, method='SLSQP',
#                                bounds=bounds,
#                                options={'maxiter': maxit, 'ftol': tol, 'disp': False})
#             elif method == 'LBFGSB':
#                 res = minimize(_objective_function, x0, args=args, method='L-BFGS-B',
#                                bounds=bounds,
#                                options={'maxiter': maxit, 'ftol': tol, 'disp': False})
#             elif method == 'SQP':
#                 res = minimize(_objective_function, x0, args=args, method='trust-constr',
#                                bounds=bounds,
#                                options={'maxiter': maxit, 'xtol': tol, 'gtol': 1e-2})
#             elif method == 'NM':
#                 res = minimize(_objective_function, x0, args=args, method='Nelder-Mead',
#                                options={'maxiter': maxit, 'fatol': tol, 'disp': False})
#             elif method == 'BFGS':
#                 res = minimize(_objective_function, x0, args=args, method='BFGS',
#                                options={'maxiter': maxit, 'disp': False})
#             elif method == 'DE':
#                 # Avoid nested pools: DE in worker uses 1 worker
#                 n_workers = 1 if _in_child() else -1
#                 res = differential_evolution(_objective_function, bounds, args=args,
#                                              maxiter=maxit, popsize=18, tol=1e-8,
#                                              strategy='best1bin', mutation=(0.5, 1.5),
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
#         res.activeparams = mask
#
#         # Hessian & inverse (only on the reference fits, not bootstrap replicates)
#
#         if not bootstrap_flag and method in ['SLSQP', 'SQP', 'NM', 'DE'] and varcov == 'H':
#             try:
#                 loss = lambda t: _objective_function(t, data, [sv], thetac, system, models,
#                                                      logging, objf, False, None)
#                 H = Hessian(loss, step=1e-4, method='central', order=2)(res.x)
#                 S = np.diag(1.0 / thetac.astype(float))
#                 Hs = S @ H @ S
#                 res.hessian = Hs
#             except Exception as e:
#                 if logging:
#                     print(f"[{sv}] Hessian computation failed: {e}")
#                 res.hessian = None
#                 Hs = np.eye(len(thetac)) * 1e6
#
#             D = np.diag(thetac.astype(float))
#             try:
#                 inv = D @ (res.hess_inv if hasattr(res, 'hess_inv') else np.linalg.pinv(Hs)) @ D
#                 res.hess_inv = inv
#             except Exception as e:
#                 if logging:
#                     print(f"[{sv}] Hessian inversion failed: {e}")
#                 res.hess_inv = np.eye(len(thetac)) * 1e6
#
#         results[sv] = res
#
#
#     return results
#
#
# # ============================================================
# # Objective wrappers
# # ============================================================
#
# def _objective_function(theta, data, active, thetac, system, models,
#                         logging, objf, bootstrap, boot_idx):
#     _, m = _objective(theta, data, active, thetac, system, models,
#                       bootstrap=bootstrap, boot_idx=boot_idx)
#     return {'LS': m['LS'], 'WLS': m['WLS'], 'MLE': m['MLE'], 'Chi': m['Chi']}[objf]
#
#
# def _objective(theta, data, active, thetac, system, models,
#                bootstrap=False, boot_idx=None):
#     import pandas as _pd
#     theta = theta.tolist()
#     tv_i, ti_i = list(system['tvi']), list(system['tii'])
#     tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
#     ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]
#
#     std_dev = {
#         v: (cfg.get('unc', 1.0) if not np.isnan(cfg.get('unc', 1.0)) else 1.0)
#         for v, cfg in {**system['tvo'], **system['tio']}.items()
#         if cfg.get('meas', True)
#     }
#
#     phisc, phitsc, tsc = {v: 1 for v in ti_i}, {v: 1 for v in tv_i}, 1
#     solver = active[0]
#
#     global_y_true, global_y_pred, global_y_err, varm = {}, {}, {}, {}
#
#     # --------------------------------------------------
#     # Simulate & gather measurements
#     # --------------------------------------------------
#     for sh in data.values():
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
#         cvp = {v: sh[f"CVP:{v}"].iloc[0] for v in system['tvi']}
#
#         tvs, tis, _ = simula(t_all, swps, ti_in, phisc, phitsc, tsc,
#                              theta, thetac, cvp, tv_in, solver, system, models)
#
#         # time-varying outputs
#         for v in tv_o:
#             xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
#             if xc not in sh or yc not in sh:
#                 continue
#             mask = ~sh[xc].isna().values
#             times = sh[xc][mask].values
#             y_t = sh[yc][mask].values
#
#             idx = np.isin(t_all, times)
#             y_p = np.array(tvs[v])[idx]
#
#             ye_col = f"MES_E:{v}"
#             y_e = (sh[ye_col][mask].values if ye_col in sh
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
#             yc = f"MES_Y:{v}"
#             if yc not in sh:
#                 continue
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
#         all_y_err += yel[:n]
#         all_vars += [v] * n
#
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
#     all_y_err = np.array(all_y_err)
#     all_vars = np.array(all_vars)
#     N = len(all_y_true)
#
#     # --------------------------------------------------
#     # Bootstrap resampling (fixed index)
#     # --------------------------------------------------
#     if bootstrap:
#         if boot_idx is None:
#             raise ValueError("bootstrap=True but boot_idx is None.")
#         idx = boot_idx
#         if idx.size != N:
#             raise ValueError(f"bootstrap index length {idx.size} != N {N}")
#         y_t, y_p, y_e, vars_ = (all_y_true[idx], all_y_pred[idx],
#                                 all_y_err[idx], all_vars[idx])
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
#         per_var['LS'][v] = np.sum((yt - yp) ** 2) / n
#         per_var['WLS'][v] = np.sum(((yt - yp) / std_dev[v]) ** 2) / n
#
#         rel_err = (yt - yp) / np.maximum(np.abs(yt), eps)
#         rel_unc = np.clip(ye / np.maximum(np.abs(yt), eps), 1e-3, 1e2)
#         per_var['MLE'][v] = np.mean(
#             0.5 * (np.log(2 * np.pi * rel_unc ** 2) + (rel_err ** 2) / (rel_unc ** 2))
#         )
#         per_var['Chi'][v] = np.sum(rel_err ** 2) / n
#
#     LS = sum(per_var['LS'].values()) * N
#     WLS = sum(per_var['WLS'].values()) * N
#     MLE = sum(per_var['MLE'].values()) * N
#     Chi = sum(per_var['Chi'].values()) * N
#
#     # R^2 per response (for reporting)
#     r2 = {}
#     for v, me in varm.items():
#         yta = np.array([m[0] for m in me])
#         ypa = np.array([m[1] for m in me])
#         if len(yta) > 1:
#             r2[v] = 1 - np.sum((yta - ypa) ** 2) / np.sum((yta - yta.mean()) ** 2)
#         else:
#             r2[v] = np.nan
#
#     metrics = {'LS': LS, 'WLS': WLS, 'MLE': MLE, 'Chi': Chi, 'R2_responses': r2}
#     return data, metrics
#
#
#
#
#
#




#iden_parmest.py

import os
import warnings
warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step, clipping to bounds"
)

# Limit BLAS oversubscription (optional but helps stability)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


from scipy.stats import truncnorm
import numpy as np
import copy
import multiprocessing as mp
from operator import attrgetter
from concurrent.futures import ProcessPoolExecutor, as_completed
from numdifftools import Hessian
from scipy.optimize import minimize, differential_evolution
from middoe.krnl_simula import simula
from middoe.iden_utils import _initialize_dictionaries
from middoe.log_utils import  read_excel


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
    """
    Check if code is executing inside a worker process.

    Returns
    -------
    bool
        True if running inside a worker process (multiprocessing child),
        False if running in the main process.

    Notes
    -----
    This function is used to prevent nested parallelism in optimization routines.
    For example, when using differential_evolution inside a multi-start loop,
    the DE algorithm should use workers=1 if already inside a parallel worker.

    See Also
    --------
    _runner : Uses this check to configure nested optimization workers.
    _multi_start_runner : Parent process that spawns workers.

    Examples
    --------
    >>> if _in_child():
    ...     n_workers = 1  # Serial execution in worker
    ... else:
    ...     n_workers = -1  # Use all available cores
    """
    return mp.current_process().name != "MainProcess"


# ============================================================
# Helpers
# ============================================================

def _initialise_theta_parameters(theta_min, theta_max, active_models,  mutation=None, theta_fixed=None):
    """
    Initialize parameter vectors for active models with optional masking.

    This function samples parameter values uniformly within specified bounds for
    each active model. If a mutation mask is provided, inactive (fixed) parameters
    are set to values from theta_fixed rather than being sampled.

    Parameters
    ----------
    theta_min : dict[str, list[float]]
        Lower bounds for parameters. Keys are model/solver names, values are
        lists of lower bounds for each parameter.
    theta_max : dict[str, list[float]]
        Upper bounds for parameters. Keys are model/solver names, values are
        lists of upper bounds for each parameter.
    active_models : list[str]
        List of model/solver names for which to initialize parameters.
    mutation : dict[str, list[bool]], optional
        Boolean masks indicating which parameters are free (True) vs fixed (False).
        If None, all parameters are considered free (default: None).
    theta_fixed : dict[str, list[float]], optional
        Fixed parameter values for inactive parameters. Used when mutation is
        provided. Keys are model names, values are parameter vectors (default: None).

    Returns
    -------
    theta_parameters : dict[str, list[float]]
        Initialized parameter vectors for each model. Free parameters are sampled
        uniformly within bounds; fixed parameters are set to theta_fixed values.

    Notes
    -----
    **Sampling Strategy**:
        - Free parameters (mutation[i] == True): Sampled uniformly from [theta_min[i], theta_max[i]].
        - Fixed parameters (mutation[i] == False): Set to theta_fixed[i].

    This function is typically used for:
        - Multi-start optimization (sampling random initial guesses).
        - Bootstrap resampling (re-initializing parameters for each replicate).

    **Mutation Masks**:
    Mutation masks allow partial parameter estimation where some parameters are
    held constant (e.g., known physical constants or previously estimated values).

    See Also
    --------
    parmest : Main entry point that uses this initialization.
    _multi_start_runner : Uses this for generating multiple starting points.

    Examples
    --------
    >>> theta_min = {'M1': [0.1, 0.5, 1.0]}
    >>> theta_max = {'M1': [10.0, 5.0, 100.0]}
    >>> active_models = ['M1']
    >>> mutation = {'M1': [True, False, True]}  # Fix second parameter
    >>> theta_fixed = {'M1': [1.0, 2.5, 50.0]}
    >>> theta_init = _initialise_theta_parameters(
    ...     theta_min, theta_max, active_models, mutation, theta_fixed
    ... )
    >>> # theta_init['M1'][1] will be 2.5 (fixed), others are sampled
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
    Count total number of scalar measurements in experimental data.

    This function computes the total number of data points (N) that will be used
    in the objective function, without requiring simulation. It counts both
    time-variant and time-invariant measurements across all experimental sheets.

    Parameters
    ----------
    data : dict
        Dictionary of experimental data sheets (typically DataFrames). Keys are
        sheet identifiers, values contain measurement columns.
    system : dict
        System configuration with output definitions:
            - 'tvo' : dict
                Time-variant output specifications with 'meas' flag.
            - 'tio' : dict
                Time-invariant output specifications with 'meas' flag.

    Returns
    -------
    N : int
        Total number of scalar measurements across all sheets and variables.

    Notes
    -----
    **Measurement Counting**:
        - **Time-variant outputs**: Count non-NaN entries in 'MES_X:{var}' and 'MES_Y:{var}' columns.
        - **Time-invariant outputs**: Count non-NaN entries in 'MES_Y:{var}' column (single value per sheet).

    This count is used for:
        - Bootstrap resampling (determining index vector length).
        - Degrees of freedom calculations.
        - Objective function normalization.

    **Column Naming Conventions**:
        - 'MES_X:{var}': Measurement times for time-variant output var.
        - 'MES_Y:{var}': Measured values for output var.
        - 'MES_E:{var}': Measurement uncertainties (optional).

    See Also
    --------
    _objective : Uses N for loss normalization.
    _bootstrap_runner : Uses N for bootstrap index generation.

    Examples
    --------
    >>> system = {
    ...     'tvo': {'y1': {'meas': True}, 'y2': {'meas': True}},
    ...     'tio': {'y3': {'meas': True}}
    ... }
    >>> # Assume data has 10 time points for y1, 15 for y2, 1 for y3
    >>> N = _count_observations(data, system)
    >>> print(N)  # Output: 26
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

def parmest(system, models, iden_opt, case=None):
    r"""
    Perform parameter estimation for dynamic models using experimental data.

    This is the main entry point for parameter estimation. It supports single-start
    and multi-start optimization, multiple objective functions (LS, WLS, MLE, CS),
    various optimization algorithms, and uncertainty quantification methods including
    Hessian-based, Jacobian-based, and bootstrap resampling with truncated normal
    approximation.

    Parameters
    ----------
    system : dict
        System configuration including:
            - 't_s' : tuple[float, float]
                Start and end times.
            - 't_d' : tuple[float, float]
                Restricted initial/final intervals (dead time).
            - 't_r' : float
                Time resolution.
            - 'tvi' : dict
                Time-variant input definitions.
            - 'tii' : dict
                Time-invariant input definitions.
            - 'tvo' : dict
                Time-variant output definitions:
                    * 'meas': bool — include in estimation
                    * 'unc': float — measurement standard deviation
            - 'tio' : dict
                Time-invariant output definitions.

    models : dict
        Model definitions and parameter bounds:
            - 'can_m' : list[str]
                Active model/solver names.
            - 't_u' : dict[str, list[float]]
                Upper parameter bounds.
            - 't_l' : dict[str, list[float]]
                Lower parameter bounds.
            - 'theta' : dict[str, list[float]]
                Nominal parameter values (used as fallback for fixed parameters).
            - 'mutation' : dict[str, list[bool]]
                Parameter masks (True=free, False=fixed).
            - 'thetastart' : dict[str, list[float]], optional
                Starting parameter values (used when case='strov').

    iden_opt : dict
        Identification options:
            - 'meth' : str
                Optimization method (from paper Table S3):
                    * 'SLSQP': Sequential Least Squares Programming
                    * 'LMBFGS': L-M BFGS (Limited-memory BFGS for large parameter spaces)
                    * 'TC': Trust-region method
                    * 'NMS': Nelder-Mead simplex
                    * 'BFGS': BFGS quasi-Newton
                    * 'DE': Differential Evolution (global)
            - 'ob' : str
                Objective function (from paper Table S2):
                    * 'LS': Least Squares (deterministic models, no error information)
                    * 'WLS': Weighted Least Squares (heteroscedastic errors)
                    * 'MLE': Maximum Likelihood Estimation (incorporates uncertainty)
                    * 'CS': Chi-Square (normalized residuals)
            - 'ms' : bool, optional
                Enable multi-start optimization (default: False).
            - 'var-cov' : str, optional
                Covariance estimation method (from paper Table S3):
                    * 'H': Hessian-based (local, from optimizer)
                    * 'J': Jacobian-based (local, from sensitivity matrix)
                    * 'B': Bootstrap resampling (global, computationally expensive)
            - 'nboot' : int, optional
                Number of bootstrap replicates (default: 100).
            - 'log' : bool, optional
                Enable verbose logging (default: False).
            - 'maxit' : int, optional
                Maximum optimization iterations (default: 1000).
            - 'tol' : float, optional
                Convergence tolerance (default: 1e-6).

    case : str, optional
        Estimation mode:
            - None: Standard estimation with all parameters active.
            - 'freeze': Keep existing mutation masks (for sequential estimation).
            - 'strov': Use starting values from models['thetastart'].

    Returns
    -------
    results : dict[str, scipy.optimize.OptimizeResult]
        Estimation results for each model. Keys are model/solver names.
        Each OptimizeResult object contains:
            - 'x' : np.ndarray
                Optimized normalized parameters (theta / thetac).
            - 'scpr' : np.ndarray
                Optimized parameters in original scale.
            - 'fun' : float
                Final objective function value.
            - 'success' : bool
                Optimization convergence flag.
            - 'hessian' : np.ndarray, optional
                Hessian matrix at optimum (if var-cov='H').
            - 'hess_inv' : np.ndarray, optional
                Inverse Hessian / covariance matrix (if var-cov='H').
            - 'v' : np.ndarray, optional
                Covariance matrix (if var-cov='B' or 'J').
            - 'X' : np.ndarray, optional
                Bootstrap parameter samples (if var-cov='B').
            - 'samples' : list, optional
                Bootstrap OptimizeResult objects (if var-cov='B').

    Raises
    ------
    ValueError
        If invalid method, objective function, or configuration is provided.

    Notes
    -----
    **Parameter Normalization**:
    All optimization is performed on normalized parameters \( x = \theta / \theta_c \),
    where \( \theta_c \) are the nominal values. This improves numerical conditioning
    for parameters spanning multiple orders of magnitude.

    **Multi-Start Optimization**:
    When ms=True, multiple optimizations are run in parallel from different initial
    guesses sampled uniformly within bounds. The best result (lowest objective) is returned.
    Default number of starts: 70% of CPU cores.

    **Uncertainty Quantification Methods**:

    - **Hessian-based (H)**: Local uncertainty from optimizer's Hessian matrix at optimum.
      Fast but assumes local linearity and may be inaccurate for ill-conditioned problems.

    - **Jacobian-based (J)**: Local uncertainty from sensitivity matrix (Fisher Information).
      More robust than Hessian for parameter-sensitive models.

    - **Bootstrap (B)**: Global uncertainty via residual resampling with replacement.
      Most robust but computationally expensive. Uses truncated normal approximation
      to account for parameter bounds.

    **Bootstrap Procedure**:
        1. Perform reference estimation on full dataset
        2. Generate bootstrap replicates by resampling residuals
        3. Re-estimate parameters for each replicate
        4. Compute covariance using truncated normal approximation
        5. Return both raw and truncated-normal statistics

    **Objective Functions** (from paper Table S2):
        - **LS**: \( \sum_i (y_i - \hat{y}_i)^2 / N \)
          Suitable for deterministic models with no error information.

        - **WLS**: \( \sum_i [(y_i - \hat{y}_i) / \sigma_i]^2 / N \)
          Accounts for heteroscedastic measurement errors.

        - **MLE**: \( \sum_i [\log(2\pi\sigma_i^2) + (y_i - \hat{y}_i)^2 / \sigma_i^2] / (2N) \)
          Full probabilistic framework incorporating measurement uncertainty.

        - **CS**: \( \sum_i [(y_i - \hat{y}_i) / y_i]^2 / N \)
          Normalized residuals for relative error minimization.

    **Fixed Parameters**:
    Parameters with mutation[i]=False are held constant during optimization. Their
    values are taken from theta_fixed (models['theta'] or models['thetastart']).

    **Experimental Data Format**:
    Data is loaded via read_excel() and should contain columns:
        - 'X:all': All simulation time points
        - '{var}t', '{var}l': Switching times and levels for time-variant inputs
        - '{var}': Time-invariant input values
        - 'MES_X:{var}', 'MES_Y:{var}': Measurement times and values
        - 'MES_E:{var}': Measurement uncertainties (optional, defaults to unc in system)

    References
    ----------
    .. [1] Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
       MIDDoE: An MBDoE Python package for model identification, discrimination,
       and calibration. *Computers & Chemical Engineering*.
       See Supplementary Material Tables S2 and S3 for details.

    .. [2] Bard, Y. (1974).
       *Nonlinear Parameter Estimation*. Academic Press, New York.
       Referenced for MLE formulation and R² calculation.

    .. [3] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.
       https://doi.org/10.1016/j.ces.2008.07.006

    See Also
    --------
    _runner : Core single-start optimization.
    _multi_start_runner : Multi-start parallelization.
    _bootstrap_runner : Bootstrap uncertainty quantification.
    _objective : Objective function computation.
    uncert : Post-estimation uncertainty and sensitivity analysis.

    Examples
    --------
    >>> # Basic single-start estimation with WLS
    >>> iden_opt = {'meth': 'SLSQP', 'ob': 'WLS'}
    >>> results = parmest(system, models, iden_opt)
    >>> print(f"Optimized parameters: {results['M1'].scpr}")
    >>> print(f"Final objective: {results['M1'].fun:.6f}")

    >>> # Multi-start with Jacobian-based uncertainty
    >>> iden_opt = {
    ...     'meth': 'DE',
    ...     'ob': 'MLE',
    ...     'ms': True,
    ...     'var-cov': 'J'
    ... }
    >>> results = parmest(system, models, iden_opt)
    >>> print(f"Parameter covariance: {results['M1'].v}")

    >>> # Bootstrap uncertainty quantification
    >>> iden_opt = {
    ...     'meth': 'LMBFGS',
    ...     'ob': 'WLS',
    ...     'var-cov': 'B',
    ...     'nboot': 500
    ... }
    >>> results = parmest(system, models, iden_opt)
    >>> print(f"Bootstrap covariance: {results['M1'].v}")
    >>> print(f"Bootstrap samples shape: {results['M1'].X.shape}")
    """

    data = read_excel()
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
    """
    Execute a single optimization start for one model (multi-start worker function).

    This function is designed to be called by ProcessPoolExecutor workers. It wraps
    _runner() with exception handling and returns results or error information.

    Parameters
    ----------
    sv : str
        Model/solver name.
    thetac : list[float]
        Nominal parameter values for normalization.
    thetamin : list[float]
        Lower parameter bounds (unnormalized).
    thetamax : list[float]
        Upper parameter bounds (unnormalized).
    thetas : list[bool]
        Mutation mask (True=free, False=fixed).
    objf : str
        Objective function ('LS', 'WLS', 'MLE', 'CS').
    method : str
        Optimization method ('SLSQP', 'LBFGSB', etc.).
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    logging : bool
        Enable verbose logging.
    varcov : str
        Covariance method ('H' or 'B').
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    result : tuple
        If successful: (sv, results_dict) where results_dict contains OptimizeResult.
        If failed: (sv, exception, traceback_string) for debugging.

    Notes
    -----
    This function is executed in a separate process (via ProcessPoolExecutor).
    Exception handling is crucial to prevent worker crashes from terminating
    the entire multi-start run.

    See Also
    --------
    _multi_start_runner : Parent function that spawns workers.
    _runner : Core optimization routine called internally.

    Examples
    --------
    >>> # Typically called indirectly via ProcessPoolExecutor
    >>> result = _run_single_start(
    ...     'M1', [1.0, 2.0], [0.1, 0.5], [10.0, 5.0], [True, True],
    ...     'WLS', 'SLSQP', data, system, models, False, 'H', 1000, 1e-6
    ... )
    """
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
    """
    Perform multi-start parameter estimation with parallel execution.

    This function runs multiple optimization attempts in parallel from different
    randomly sampled initial guesses. The best result (lowest objective value)
    from all successful runs is returned for each model.

    Parameters
    ----------
    active_models : list[str]
        Model/solver names to estimate.
    _unused : dict
        Placeholder argument (ignored, for API consistency).
    bmax : dict[str, list[float]]
        Upper parameter bounds for each model.
    bmin : dict[str, list[float]]
        Lower parameter bounds for each model.
    mutation : dict[str, list[bool]]
        Parameter masks for each model.
    objf : str
        Objective function ('LS', 'WLS', 'MLE', 'Chi').
    method : str
        Optimization method.
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    logging : bool
        Enable verbose logging.
    varcov : str
        Covariance method ('H' or 'B').
    maxit : int
        Maximum iterations per start.
    tol : float
        Convergence tolerance.
    nstarts : int, optional
        Number of random starts per model. If None, defaults to max(2*n_workers, 10),
        where n_workers = 70% of CPU cores.

    Returns
    -------
    results : dict[str, scipy.optimize.OptimizeResult or None]
        Best optimization result for each model. None if all starts failed.

    Notes
    -----
    **Parallelization Strategy**:
        - Number of workers: 70% of available CPU cores.
        - Number of starts: max(2 * n_workers, 10) by default.
        - Each start samples parameters uniformly within bounds.

    **Result Selection**:
    For each model, the result with the lowest objective function value among
    all successful starts is selected. If all starts fail, None is returned.

    **Worker Crash Handling**:
    If a worker process crashes, the exception and traceback are logged, but
    the multi-start run continues with remaining tasks.

    **Fixed vs Free Parameters**:
    Free parameters (mutation=True) are sampled; fixed parameters (mutation=False)
    are set to models['theta'] values.

    See Also
    --------
    _run_single_start : Worker function for a single start.
    _multi_start_runner_serial : Serial version (for debugging).
    parmest : Main entry point that invokes this for ms=True.

    Examples
    --------
    >>> results = _multi_start_runner(
    ...     ['M1'], None, bound_max, bound_min, mutation, 'WLS',
    ...     'SLSQP', data, system, models, False, 'H', 1000, 1e-6, nstarts=20
    ... )
    >>> print(results['M1'].fun)  # Best objective value found
    """
    del _unused
    nwork = max(1, int(0.7 * mp.cpu_count()))
    nstarts = nstarts or max(2 * nwork, 10)

    allr = {sv: [] for sv in active_models}
    tasks = []
    for sv in active_models:
        for _ in range(nstarts):
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
    """
    Perform multi-start parameter estimation with serial execution.

    This is a serial (single-threaded) version of _multi_start_runner, used
    primarily for debugging or when parallelization is unavailable. It performs
    the same logic but runs starts sequentially.

    Parameters
    ----------
    active_models : list[str]
        Model/solver names to estimate.
    _unused : dict
        Placeholder argument (ignored, for API consistency).
    bmax : dict[str, list[float]]
        Upper parameter bounds for each model.
    bmin : dict[str, list[float]]
        Lower parameter bounds for each model.
    mutation : dict[str, list[bool]]
        Parameter masks for each model.
    objf : str
        Objective function ('LS', 'WLS', 'MLE', 'Chi').
    method : str
        Optimization method.
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    logging : bool
        Enable verbose logging.
    varcov : str
        Covariance method ('H' or 'B').
    maxit : int
        Maximum iterations per start.
    tol : float
        Convergence tolerance.
    nstarts : int, optional
        Number of random starts per model (default: 20).

    Returns
    -------
    results : dict[str, scipy.optimize.OptimizeResult or None]
        Best optimization result for each model. None if all starts failed.

    Notes
    -----
    This function is useful for:
        - Debugging multi-start logic without parallelization overhead.
        - Environments where multiprocessing is problematic (e.g., some Jupyter setups).
        - Profiling individual optimization runs.

    It is functionally equivalent to _multi_start_runner but executes serially,
    making it easier to trace errors and step through code.

    See Also
    --------
    _multi_start_runner : Parallel version (preferred for production).
    _runner : Core optimization routine called internally.

    Examples
    --------
    >>> results = _multi_start_runner_serial(
    ...     ['M1'], None, bound_max, bound_min, mutation, 'WLS',
    ...     'SLSQP', data, system, models, True, 'H', 1000, 1e-6, nstarts=5
    ... )
    """
    del _unused
    nstarts = nstarts or 20
    allr = {sv: [] for sv in active_models}

    for sv in active_models:
        for _ in range(nstarts):
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
    """
    Execute a single bootstrap replicate estimation (worker function).

    This function performs parameter estimation on a bootstrap-resampled dataset
    defined by boot_idx (a vector of measurement indices sampled with replacement).
    It is designed to be called by ProcessPoolExecutor workers.

    Parameters
    ----------
    active_models : list[str]
        Model/solver names to estimate.
    theta_params : dict[str, list[float]]
        Starting parameter values for each model.
    bmax : dict[str, list[float]]
        Upper parameter bounds.
    bmin : dict[str, list[float]]
        Lower parameter bounds.
    mutation : dict[str, list[bool]]
        Parameter masks.
    objf : str
        Objective function ('LS', 'WLS', 'MLE', 'Chi').
    x0 : dict[str, list[float]]
        Normalized starting points (typically [1.0, 1.0, ...]).
    method : str
        Optimization method.
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions (will be augmented with bootstrap flags).
    logging : bool
        Enable verbose logging.
    multi : bool
        If True, use multi-start optimization for this bootstrap replicate.
    varcov : str
        Covariance method (passed to _runner, typically 'H' for bootstrap).
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    boot_idx : np.ndarray
        Bootstrap resampling indices (length = N, values in [0, N-1]).

    Returns
    -------
    ref : dict[str, scipy.optimize.OptimizeResult]
        Estimation results for this bootstrap replicate.

    Notes
    -----
    **Bootstrap Resampling**:
    The boot_idx array defines which measurements are included in this replicate.
    Measurements are resampled with replacement, creating a dataset of the same
    size as the original but with some observations repeated and others omitted.

    **Models Dictionary Augmentation**:
    This function adds two special keys to models:
        - '__bootstrap__': True (flag for _objective to use boot_idx).
        - '__boot_idx__': boot_idx array.

    **Multi-Start Option**:
    If multi=True, each bootstrap replicate uses serial multi-start optimization
    (_multi_start_runner_serial) to improve robustness. This is more expensive
    but reduces the risk of converging to local minima in noisy bootstrap data.

    See Also
    --------
    _bootstrap_runner : Parent function that spawns bootstrap workers.
    _objective : Uses __boot_idx__ for resampling.
    _multi_start_runner_serial : Used when multi=True.

    Examples
    --------
    >>> # Typically called indirectly via ProcessPoolExecutor
    >>> boot_idx = np.array([0, 2, 2, 3, 1, 0])  # Example resampling
    >>> result = _bootstrap_worker(
    ...     ['M1'], theta_params, bmax, bmin, mutation, 'WLS',
    ...     x0_dict, 'SLSQP', data, system, models, False, False, 'H', 1000, 1e-6,
    ...     boot_idx
    ... )
    """
    models_boot = {**models,
                   '__bootstrap__': True,
                   '__boot_idx__': boot_idx}

    if multi:
        theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models, mutation=mutation, theta_fixed=models['theta'])
        ref = _multi_start_runner_serial(active_models, theta_params_r, bmax, bmin,
                                         mutation, objf, method, data, system,
                                         models_boot, logging, varcov, maxit, tol)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
                      x0, method, data, system, models_boot, logging, varcov, maxit, tol)
    return ref


def truncated_moments_from_data(X, eps=1e-10):
    """
    Compute mean and covariance from parameter samples assuming truncated normal distribution.

    This function fits a truncated normal distribution to parameter samples (constrained
    by bounds) and computes analytical moments. It uses empirical mean/std for each
    dimension, then applies truncated normal formulas for corrected statistics.

    Parameters
    ----------
    X : np.ndarray
        Parameter sample matrix, shape (N_samples, N_parameters). Each row is one
        bootstrap or Monte Carlo sample.
    eps : float, optional
        Small constant for numerical stability (default: 1e-10). Used to handle
        near-zero standard deviations.

    Returns
    -------
    mu_trunc : np.ndarray
        Mean vector of truncated normal distribution, shape (N_parameters,).
    cov_trunc : np.ndarray
        Covariance matrix of truncated normal distribution, shape (N_parameters, N_parameters).

    Notes
    -----
    **Truncated Normal Distribution**:
    When parameters are bounded (e.g., physical constraints like k > 0), bootstrap
    samples are constrained to [theta_min, theta_max]. A standard normal approximation
    is biased. The truncated normal accounts for this truncation analytically.

    **Algorithm**:
        1. Compute empirical mean (\( \mu_{emp} \)) and std (\( \sigma_{emp} \)) per dimension.
        2. Determine truncation bounds from sample range: [min(X), max(X)].
        3. Fit scipy.stats.truncnorm for each dimension.
        4. Extract analytical mean (\( \mu_{trunc} \)) and variance (\( \sigma_{trunc}^2 \)).
        5. Compute correlation matrix from empirical samples (assumes truncation doesn't affect correlation).
        6. Reconstruct covariance: \( \text{Cov}_{trunc} = \text{diag}(\sigma_{trunc}) \cdot \text{Corr}_{emp} \cdot \text{diag}(\sigma_{trunc}) \).

    **Constant Parameters**:
    If \( \sigma_{emp} < \epsilon \), the parameter is treated as constant (variance ≈ 0).
    A warning is printed listing constant parameter indices.

    **Correlation Handling**:
    The function assumes truncation affects marginal distributions but not correlation
    structure. This is a simplification but works well for moderately correlated parameters.

    References
    ----------
    .. [1] Burkardt, J. (2014). The Truncated Normal Distribution.
       Department of Scientific Computing, Florida State University.

    .. [2] scipy.stats.truncnorm documentation:
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

    See Also
    --------
    compute_trunc_mean_cov_from_empirical : High-level wrapper for this function.
    _bootstrap_runner : Uses this for final covariance estimation.

    Examples
    --------
    >>> # Simulate 100 bootstrap samples for 3 parameters
    >>> X = np.random.randn(100, 3) + 5  # Centered around 5
    >>> mu, cov = truncated_moments_from_data(X)
    >>> print(mu)  # Truncated mean
    >>> print(np.diag(cov))  # Truncated variances
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
    Compute truncated normal statistics from bootstrap result object.

    This is a convenience wrapper around truncated_moments_from_data that operates
    on a result dictionary containing bootstrap samples. It updates the result
    object with truncated mean and covariance in-place.

    Parameters
    ----------
    resultpr : dict[str, scipy.optimize.OptimizeResult]
        Bootstrap estimation results. Each OptimizeResult should have a 'samples'
        attribute containing bootstrap replicate results.
    solver : str, optional
        Model/solver name to process (default: 'M').
    scpr_key : str, optional
        Attribute name containing parameters in bootstrap samples (default: 'scpr').
    include_X : bool, optional
        If True, store full sample matrix in resultpr[solver]['X'] (default: True).

    Returns
    -------
    resultpr2 : dict
        Updated result dictionary with:
            - resultpr[solver]['scpr']: Truncated mean.
            - resultpr[solver]['v']: Truncated covariance.
            - resultpr[solver]['X']: Sample matrix (if include_X=True).

    Notes
    -----
    This function is typically used for post-processing bootstrap results when
    a non-bootstrap reference fit is desired but truncated statistics are needed.

    It modifies a deep copy of the input, leaving the original unchanged.

    See Also
    --------
    truncated_moments_from_data : Core statistical computation.
    _bootstrap_runner : Directly computes truncated stats inline.

    Examples
    --------
    >>> # Assume resultpr contains bootstrap samples
    >>> resultpr_updated = compute_trunc_mean_cov_from_empirical(resultpr, solver='M1')
    >>> print(resultpr_updated['M1']['scpr'])  # Truncated mean
    >>> print(resultpr_updated['M1']['v'])  # Truncated covariance
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
    """
    Perform bootstrap-based parameter estimation with truncated normal uncertainty.

    This function executes the full bootstrap workflow:
        1. Reference fit on original dataset.
        2. Bootstrap resampling (N replicates, sampling with replacement).
        3. Parallel estimation on each bootstrap replicate.
        4. Statistical analysis (raw and truncated normal approximations).

    Parameters
    ----------
    active_models : list[str]
        Model/solver names to estimate.
    theta_params : dict[str, list[float]]
        Starting parameter values (typically nominal or from previous fit).
    bmax : dict[str, list[float]]
        Upper parameter bounds.
    bmin : dict[str, list[float]]
        Lower parameter bounds.
    mutation : dict[str, list[bool]]
        Parameter masks (True=free, False=fixed).
    objf : str
        Objective function ('LS', 'WLS', 'MLE', 'Chi').
    x0 : dict[str, list[float]]
        Normalized starting points.
    method : str
        Optimization method.
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    logging : bool
        Enable verbose logging.
    nboot : int
        Number of bootstrap replicates.
    multi : bool
        If True, use multi-start optimization for reference fit and bootstrap replicates.
    varcov : str
        Covariance method (typically 'B' for bootstrap, 'H' for Hessian reference).
    maxit : int
        Maximum iterations per optimization.
    tol : float
        Convergence tolerance.

    Returns
    -------
    ref : dict[str, scipy.optimize.OptimizeResult]
        Augmented reference fit results with bootstrap statistics:
            - 'samples' : list[OptimizeResult]
                All successful bootstrap replicate results.
            - 'X' : np.ndarray, shape (n_successful_replicates, n_params)
                Bootstrap parameter sample matrix.
            - 'scpr' : np.ndarray
                Mean parameters (raw empirical mean, including fixed params).
            - 'v' : np.ndarray
                Truncated normal covariance matrix (full-size, fixed params have zero variance).
            - 'scpr_raw' : np.ndarray
                Original reference fit parameters (before bootstrap averaging).
            - 'v_raw' : np.ndarray
                Raw empirical covariance (no truncation correction).

    Notes
    -----
    **Bootstrap Workflow**:
        1. **Reference Fit**: Estimate parameters on original dataset using _runner
           or _multi_start_runner (if multi=True).
        2. **Index Sampling**: Generate nboot index vectors by sampling [0, N-1]
           with replacement, where N = total number of measurements.
        3. **Parallel Estimation**: Submit bootstrap replicates to ProcessPoolExecutor,
           each fitting on resampled data defined by boot_idx.
        4. **Quality Filtering**: Keep only replicates with objective ≤ 2 * reference objective
           (prevents divergent fits from biasing covariance).
        5. **Statistical Analysis**:
            - Compute raw empirical mean and covariance.
            - Fit truncated normal to account for parameter bounds.
            - Store both raw and truncated statistics.

    **Truncated Normal Correction**:
    Bootstrap samples hitting parameter bounds create artificially truncated distributions.
    The truncated normal approximation corrects mean and covariance estimates to reflect
    the true (untruncated) underlying distribution.

    **Fixed Parameters**:
    Only active (mutation=True) parameters are included in covariance estimation.
    The full covariance matrix has zeros in rows/columns corresponding to fixed parameters.

    **Parallelization**:
    Bootstrap replicates are executed in parallel using min(nboot, n_cpus) workers.
    Each replicate is independent, making this embarrassingly parallel.

    **Objective Threshold**:
    The 2× threshold for accepting bootstrap fits prevents poor local minima or
    numerical failures from contaminating the covariance estimate. This is conservative
    but improves robustness.

    References
    ----------
    .. [1] Efron, B., & Tibshirani, R. J. (1993).
       *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

    .. [2] Joshi, M., Seidel-Morgenstern, A., & Kremling, A. (2006).
       Exploiting the bootstrap method for quantifying parameter confidence intervals
       in dynamical systems. *Metabolic Engineering*, 8(5), 447–455.

    See Also
    --------
    _bootstrap_worker : Worker function for individual replicates.
    truncated_moments_from_data : Statistical correction for bounded parameters.
    _multi_start_runner : Used for reference fit if multi=True.

    Examples
    --------
    >>> results = _bootstrap_runner(
    ...     ['M1'], theta_params, bound_max, bound_min, mutation, 'WLS',
    ...     x0_dict, 'SLSQP', data, system, models, False, 100, False, 'B', 1000, 1e-6
    ... )
    >>> print(results['M1'].scpr)  # Bootstrap mean parameters
    >>> print(results['M1'].v)  # Bootstrap covariance (truncated)
    >>> print(results['M1'].X.shape)  # (n_successful, n_params)
    """
    if multi:
        trunc_mc_samps = nboot
        nstarts = max(1, int(0.7 * mp.cpu_count()))
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
    """
    Execute parameter estimation for one or more models using specified optimization method.

    This is the core optimization routine that performs parameter estimation for a single
    starting point. It handles parameter normalization, bound construction, optimizer
    invocation, result post-processing (Hessian computation, scaling), and error handling.

    Parameters
    ----------
    active_models : list[str]
        Model/solver names to estimate.
    theta_params : dict[str, list[float]]
        Nominal parameter values (thetac) for each model, used for normalization.
    bmax : dict[str, list[float]]
        Upper parameter bounds (unnormalized).
    bmin : dict[str, list[float]]
        Lower parameter bounds (unnormalized).
    mutation : dict[str, list[bool]]
        Parameter masks (True=free to optimize, False=fixed).
    objf : str
        Objective function: 'LS', 'WLS', 'MLE', or 'CS'.
    x0_dict : dict[str, list[float]]
        Normalized starting points (typically [1.0, 1.0, ...] or perturbed values).
    method : str
        Optimization method:
            - 'SLSQP': Sequential Least Squares Programming.
            - 'LMBFGS': L-M BFGS (Limited-memory BFGS for large parameter spaces).
            - 'TC': Trust-region method.
            - 'NMS': Nelder-Mead simplex (derivative-free).
            - 'BFGS': Broyden-Fletcher-Goldfarb-Shanno.
            - 'DE': Differential Evolution (global, stochastic).
    data : dict
        Experimental data sheets.
    system : dict
        System configuration.
    models : dict
        Model definitions. May contain special keys for bootstrap:
            - '__bootstrap__': bool (True if bootstrap replicate).
            - '__boot_idx__': np.ndarray (resampling indices).
    logging : bool
        Enable verbose logging.
    varcov : str
        Covariance method: 'H' (Hessian-based) or 'B' (bootstrap).
    maxit : int
        Maximum optimization iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    results : dict[str, scipy.optimize.OptimizeResult]
        Estimation results for each model. Empty dict if all models fail.
        Each OptimizeResult is augmented with:
            - 'scpr' : np.ndarray
                Optimized parameters in original scale (theta_opt = x_opt * thetac).
            - 'activeparams' : np.ndarray[bool]
                Mutation mask (True for free parameters).
            - 'hessian' : np.ndarray, optional
                Scaled Hessian matrix at optimum (if varcov='H' and method supports it).
            - 'hess_inv' : np.ndarray, optional
                Scaled inverse Hessian / covariance approximation.

    Notes
    -----
    **Parameter Normalization**:
    Optimization is performed on normalized variables \( x = \theta / \theta_c \), where
    \( \theta_c \) are the nominal values (theta_params). This improves conditioning
    for parameters spanning multiple orders of magnitude.

    Bounds are transformed accordingly:
        - \( x_{min} = \theta_{min} / \theta_c \)
        - \( x_{max} = \theta_{max} / \theta_c \)

    **Fixed Parameters**:
    Parameters with mutation[i] = False are constrained to narrow bounds around x0[i],
    effectively fixing them. The narrow window (1e-40 relative) prevents optimizer issues.

    **Hessian Computation** (only when varcov='H' and not bootstrap):
    For gradient-based methods (SLSQP, SQP, NM, DE after polishing), the Hessian
    is computed numerically using numdifftools.Hessian with central differences:
        1. Compute raw Hessian \( \mathbf{H} \) in normalized space.
        2. Scale to original space: \( \mathbf{H}_s = \mathbf{S} \mathbf{H} \mathbf{S} \),
           where \( \mathbf{S} = \text{diag}(1 / \theta_c) \).
        3. Invert to get covariance: \( \mathbf{V} = \mathbf{D} \mathbf{H}_s^{-1} \mathbf{D} \),
           where \( \mathbf{D} = \text{diag}(\theta_c) \).

    **Differential Evolution (DE)**:
    To avoid nested parallelization, DE uses workers=1 inside child processes
    (_in_child() check) and workers=-1 (all cores) in main process.

    **Error Handling**:
    If optimization fails (exception raised), the model is skipped and not included
    in the results dictionary. This allows other models to proceed in multi-model estimation.

    References
    ----------
    .. [1] Nocedal, J., & Wright, S. J. (2006).
       *Numerical Optimization* (2nd ed.). Springer.

    .. [2] scipy.optimize.minimize documentation:
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    See Also
    --------
    _objective_function : Objective function called by optimizers.
    parmest : Main entry point.
    _multi_start_runner : Calls this repeatedly with different x0.

    Examples
    --------
    >>> results = _runner(
    ...     ['M1'], theta_params, bound_max, bound_min, mutation, 'WLS',
    ...     x0_dict, 'SLSQP', data, system, models, False, 'H', 1000, 1e-6
    ... )
    >>> print(results['M1'].scpr)  # Optimized parameters
    >>> print(results['M1'].fun)  # Final objective value
    """
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
            elif method == 'LMBFGS':
                res = minimize(_objective_function, x0, args=args, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxit, 'ftol': tol, 'disp': False})
            elif method == 'TC':
                res = minimize(_objective_function, x0, args=args, method='trust-constr',
                               bounds=bounds,
                               options={'maxiter': maxit, 'xtol': tol, 'gtol': 1e-2})
            elif method == 'NMS':
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
    """
    Wrapper for objective function computation (called by scipy optimizers).

    This thin wrapper extracts the requested objective from the metrics dictionary
    returned by _objective(). It exists because scipy optimizers expect a scalar
    return value, while _objective() computes multiple metrics simultaneously.

    Parameters
    ----------
    theta : np.ndarray
        Normalized parameter vector (x = theta / thetac).
    data : dict
        Experimental data sheets.
    active : list[str]
        Active model names (typically single element).
    thetac : np.ndarray
        Nominal parameter values for denormalization.
    system : dict
        System configuration.
    models : dict
        Model definitions (may include bootstrap flags).
    logging : bool
        Enable verbose logging (unused here, passed to _objective).
    objf : str
        Objective function to return: 'LS', 'WLS', 'MLE', or 'Chi'.
    bootstrap : bool
        If True, use bootstrap resampling.
    boot_idx : np.ndarray or None
        Bootstrap resampling indices (if bootstrap=True).

    Returns
    -------
    obj_value : float
        Scalar objective function value.

    See Also
    --------
    _objective : Computes all metrics and returns dictionary.
    _runner : Calls this via scipy.optimize.minimize.

    Examples
    --------
    >>> obj = _objective_function(
    ...     x, data, ['M1'], thetac, system, models, False, 'WLS', False, None
    ... )
    >>> print(obj)  # Scalar WLS objective value
    """
    _, m = _objective(theta, data, active, thetac, system, models,
                      bootstrap=bootstrap, boot_idx=boot_idx)
    return {'LS': m['LS'], 'WLS': m['WLS'], 'MLE': m['MLE'], 'Chi': m['Chi']}[objf]


def _objective(theta, data, active, thetac, system, models,
               bootstrap=False, boot_idx=None):
    """
    Compute all objective function metrics for parameter estimation.

    This function performs forward simulation for all experiments with the given
    parameters, compares predictions to measurements, and computes multiple loss
    metrics (LS, WLS, MLE, Chi-squared) as well as per-response R² values.

    Parameters
    ----------
    theta : np.ndarray or list
        Normalized parameter vector (x = theta / thetac).
    data : dict
        Experimental data sheets. Keys are sheet identifiers, values are DataFrames with:
            - 'X:all': Time vector for simulation.
            - '{var}t', '{var}l': Switching times/levels for time-variant inputs.
            - '{var}': Time-invariant input values.
            - 'MES_X:{var}', 'MES_Y:{var}': Measurement times and values.
            - 'MES_E:{var}': Measurement uncertainties (optional).
            - 'CVP:{var}': Control variable parameterisation flags.
    active : list[str]
        Active model/solver names (typically single element).
    thetac : np.ndarray
        Nominal parameter values for denormalization (theta_unnorm = theta * thetac).
    system : dict
        System configuration with input/output definitions and uncertainties.
    models : dict
        Model definitions (passed to simula).
    bootstrap : bool, optional
        If True, use bootstrap resampling defined by boot_idx (default: False).
    boot_idx : np.ndarray or None, optional
        Bootstrap resampling indices (length N, values in [0, N-1]) (default: None).

    Returns
    -------
    data : dict
        Input data (unchanged, returned for consistency).
    metrics : dict
        Dictionary of computed metrics:
            - 'LS' : float
                Least Squares: \( \sum_i (y_i - \hat{y}_i)^2 \).
            - 'WLS' : float
                Weighted Least Squares: \( \sum_i [(y_i - \hat{y}_i) / \sigma_i]^2 \).
            - 'MLE' : float
                Negative log-likelihood (relative errors):
                \( \sum_i [\log(2\pi\sigma_i^2) + (r_i / \sigma_i)^2] / 2 \),
                where \( r_i = (y_i - \hat{y}_i) / y_i \).
            - 'Chi' : float
                Chi-squared (relative errors): \( \sum_i [(y_i - \hat{y}_i) / y_i]^2 \).
            - 'R2_responses' : dict[str, float]
                Coefficient of determination for each response variable.

    Raises
    ------
    ValueError
        If bootstrap=True but boot_idx is None.
        If boot_idx length does not match total number of measurements.

    Notes
    -----
    **Simulation Workflow**:
        1. For each experimental sheet:
            - Extract time-variant inputs (switching profiles).
            - Extract time-invariant inputs (constants).
            - Call simula() to get model predictions.
            - Compare predictions to measurements at measurement times.
        2. Concatenate all residuals across sheets and variables.
        3. Apply bootstrap resampling if requested.
        4. Compute loss metrics.

    **Bootstrap Resampling**:
    When bootstrap=True, measurements are resampled with replacement according to
    boot_idx. This creates a dataset of the same size but with some observations
    repeated and others omitted, enabling uncertainty quantification.

    **Objective Functions**:
        - **LS (Least Squares)**: Minimizes sum of squared absolute errors.
          Best for homoscedastic noise (constant variance).

        - **WLS (Weighted Least Squares)**: Minimizes sum of squared weighted errors.
          Weights are \( w_i = 1 / \sigma_i^2 \), where \( \sigma_i \) is from
          system['tvo'][var]['unc'] or system['tio'][var]['unc'].
          Best for heteroscedastic noise (variance depends on response).

        - **MLE (Maximum Likelihood Estimation)**: Assumes relative errors
          \( (y - \hat{y}) / y \) follow normal distribution.
          Equivalent to minimizing negative log-likelihood.
          Best for multiplicative noise (percentage errors).

        - **Chi (Chi-squared)**: Similar to MLE but without log term.
          Minimizes sum of squared relative errors.

    **Missing Data Handling**:
        - NaN values in measurement times/values are automatically skipped.
        - If no valid measurements exist for a variable in a sheet, that variable
          is skipped for that sheet.

    **Uncertainty Defaults**:
    If measurement uncertainties ('MES_E:{var}') are not provided, defaults are:
        - system['tvo'][var]['unc'] or system['tio'][var]['unc'] if specified.
        - 1.0 otherwise.

    **Coefficient of Determination (R²)**:
    For each response variable:
        \[
        R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
        \]
    R² = 1 indicates perfect fit, R² < 0 indicates worse than mean predictor.

    See Also
    --------
    _objective_function : Wrapper that returns single metric.
    simula : Forward simulation kernel.
    _runner : Calls this via _objective_function.

    Examples
    --------
    >>> theta_norm = np.array([1.2, 0.8, 1.0])
    >>> data, metrics = _objective(
    ...     theta_norm, data, ['M1'], thetac, system, models, False, None
    ... )
    >>> print(metrics['WLS'])  # Weighted least squares objective
    >>> print(metrics['R2_responses'])  # R² for each variable
    """
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