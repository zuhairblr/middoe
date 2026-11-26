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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize, differential_evolution, OptimizeResult
from middoe.krnl_simula import simula
try:
    from middoe.krnl_simula import SimulationError
except ImportError:  # pragma: no cover - older krnl_simula without the class
    class SimulationError(RuntimeError):
        """Raised when a user model returns invalid output."""
from middoe.iden_utils import _initialize_dictionaries
from middoe.log_utils import  read_excel


# ------------------------------------------------------------------
# Option names: canonical values and backward-compatible aliases
# (paper Table S3 keys 'G'/'Ls'/'Ln'; pre-2025-11-23 keys 'SQP'/'NM'/'LBFGSB';
#  paper docstring keys 'CS' and 'J')
# ------------------------------------------------------------------
_METH_ALIAS = {'G': 'DE', 'Ls': 'SLSQP', 'Ln': 'NMS',
               'SQP': 'TC', 'NM': 'NMS', 'LBFGSB': 'LMBFGS'}
_OB_ALIAS = {'CS': 'Chi'}
_VC_ALIAS = {'J': 'M'}
_VALID_METH = ('SLSQP', 'LMBFGS', 'TC', 'NMS', 'BFGS', 'DE')
_VALID_OB = ('LS', 'WLS', 'MLE', 'Chi')
_VALID_VC = ('H', 'M', 'B')
_VALID_SENS = ('central', 'forward')


def _canonical_option(iden_opt, key, aliases, valid, default=None, required=False):
    """Map an alias to its canonical name, validate it and write it back."""
    if key not in iden_opt or iden_opt[key] is None:
        if required:
            raise ValueError(f"iden_opt['{key}'] is required; valid values: {list(valid)}")
        if default is None:
            return None
        iden_opt[key] = default
    val = iden_opt[key]
    if not isinstance(val, str):
        raise ValueError(f"iden_opt['{key}'] must be a string, got {val!r}; valid values: {list(valid)}")
    canon = aliases.get(val, val)
    if canon not in valid:
        raise ValueError(f"Unknown iden_opt['{key}']={val!r}; valid values: {list(valid)}"
                         f" (accepted aliases: {aliases})")
    iden_opt[key] = canon
    return canon


def _normalize_iden_opt(iden_opt, need_meth=True):
    """
    Validate ``iden_opt`` and map paper-era / documented aliases to canonical names
    (``meth``, ``ob``, ``var-cov``, ``sens_m``). The canonical names are written back
    into ``iden_opt`` so that downstream code (uncert, estima, reports) sees them.
    """
    if need_meth:
        _canonical_option(iden_opt, 'meth', _METH_ALIAS, _VALID_METH, required=True)
        _canonical_option(iden_opt, 'ob', {**_OB_ALIAS}, _VALID_OB, required=True)
    else:
        if iden_opt.get('meth') is not None:
            _canonical_option(iden_opt, 'meth', _METH_ALIAS, _VALID_METH)
        if iden_opt.get('ob') is not None:
            _canonical_option(iden_opt, 'ob', _OB_ALIAS, _VALID_OB)
    _canonical_option(iden_opt, 'var-cov', _VC_ALIAS, _VALID_VC, default='H')
    if iden_opt.get('sens_m') is not None:
        _canonical_option(iden_opt, 'sens_m', {}, _VALID_SENS)
    return iden_opt


def _signed_scaling(theta, mask=None, name=''):
    """
    Return (thetac, x0) with thetac = |theta| and x0 = sign(theta), so that
    theta = thetac * x0 and bounds/thetac keep their orientation. A free
    parameter with theta == 0 cannot be scaled and raises ValueError.
    """
    th = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError(f"Model '{name}': non-finite nominal parameter values {th.tolist()}")
    m = np.ones(th.shape, bool) if mask is None else np.asarray(mask, bool)
    zero = np.where((th == 0) & m)[0]
    if zero.size:
        raise ValueError(f"Model '{name}': free parameter(s) at index {zero.tolist()} have a nominal "
                         f"value of 0, which cannot be used for scaling; give a non-zero nominal value.")
    return np.abs(th), np.sign(th)


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
    r"""
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
    r"""
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
                mask = ~sh[xc].isna().values & ~sh[yc].isna().values
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

def parmest(system, models, iden_opt, case=None, data=None):
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
                    * 'Chi': relative least squares, ``sum(((y - y_hat)/max(|y|, sigma))^2)`` (alias 'CS')

              Aliases accepted for 'meth': 'G'->'DE', 'Ls'->'SLSQP', 'Ln'->'NMS',
              'SQP'->'TC', 'NM'->'NMS', 'LBFGSB'->'LMBFGS'. Unknown values raise
              ValueError; the canonical name is written back into iden_opt.
            - 'ms' : bool, optional
                Enable multi-start optimization (default: False).
            - 'var-cov' : str, optional
                Covariance estimation method (from paper Table S3):
                    * 'H': Hessian-based (numerical Hessian of the objective at the optimum)
                    * 'M': Jacobian/FIM-based (alias 'J')
                    * 'B': Bootstrap resampling (global, computationally expensive)

                Default: 'H'.
            - 'nboot' : int, optional
                Number of bootstrap replicates (default: 100).
            - 'log' : bool, optional
                Enable verbose logging (default: False).
            - 'maxit' : int, optional
                Maximum optimization iterations (default: 1000), per optimiser pass.
            - 'tol' : float, optional
                Convergence tolerance (default: 1e-6). With 'fnorm' (default) the
                objective is divided by its own size, so for every method it acts
                relative to the objective at the optimum: LMBFGS 'ftol' (relative
                decrease per iteration), SLSQP 'ftol', NMS 'fatol', TC 'gtol'
                (gradient w.r.t. the internal variables). BFGS uses SciPy's gtol
                (1e-5) and DE its own tol (1e-8); 'tol' does not apply to them.
                Values like 0.1 stop LMBFGS when an iteration improves the objective
                by less than 10 % and can end far from the minimum.
            - 'pscale' : {'log', 'lin'}, optional
                Internal variables of the optimiser (default 'log'): log|theta/theta0|
                for parameters whose bounds exclude 0 (equal steps = equal relative
                changes, however far the optimum is from the start), theta/theta0 for
                the others. 'lin': theta/theta0 for all (behaviour before 2026-11).
                Results, Hessian and covariances are not affected, only the path.
            - 'grad' : {'2-point', '3-point', 'legacy'}, optional
                Finite-difference gradient of SLSQP, LMBFGS, TC and BFGS (default
                '2-point': forward differences with a relative step; '3-point':
                central, twice the cost). 'legacy': SciPy's default forward
                differences with an absolute step of 1e-8, which is mostly noise for
                weakly sensitive parameters.
            - 'grad_step' : float, optional
                Relative step of the '2-point'/'3-point' gradient (default 1e-6).
            - 'fnorm' : bool, optional
                Normalise the objective for the optimiser (default True): MLE is
                minimised without its constant 0.5*sum(log 2 pi sigma^2) (same
                minimiser), and the objective is scaled so that every method's stopping
                test is relative to the objective's size: SLSQP/NMS/TC/BFGS divide by
                the starting value, LMBFGS (already relative above 1) only scales
                objectives below 1 up; the optimiser is restarted from its result while
                the objective keeps falling by more than 10x (at most 5 passes). The
                reported 'fun' is the full objective.
            - 'boot_start' : str, optional
                Starting points of each bootstrap replicate (var-cov='B'). 'mixed'
                (default): the reference estimate, plus 4 random starts when 'ms' is
                True; the best fit is kept. 'ref': the reference estimate only (fastest,
                but can miss a second minimum of a poorly identifiable parameter).
                'multi': as the reference fit (20 random starts with 'ms', about 4x the
                cost of 'mixed'; else the initial guess).

    case : str, optional
        Estimation mode:
            - None: Standard estimation. Existing ``models['mutation']`` masks are
              kept; a missing mask is created as all-True (all parameters free).
            - 'freeze': Same mask handling as None (kept for backward compatibility).
            - 'strov': Use starting values from models['thetastart'].

    data : dict, optional
        Experimental data ``{sheet_name: DataFrame}``. If None (default) the data
        are read from ``data.xlsx`` in the working directory.

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
        - **LS**: \( \sum_i (y_i - \hat{y}_i)^2 \)
          Suitable for deterministic models with no error information.

        - **WLS**: \( \sum_i [(y_i - \hat{y}_i) / \sigma_i]^2 \), per-point
          \( \sigma_i \) from 'MES_E' (fallback: system 'unc'), paper eq. S7.

        - **MLE**: \( \frac{1}{2}\sum_i [\log(2\pi\sigma_i^2) + (y_i - \hat{y}_i)^2 / \sigma_i^2] \)
          (paper eq. S6), per-point \( \sigma_i \).

        - **Chi** (alias 'CS'): :math:`\sum_i [(y_i - \hat{y}_i) / \max(|y_i|, \sigma_i)]^2`
          Relative least squares. ``|y|`` is floored at the point's \( \sigma_i \), so values
          measured near zero (below the noise) do not get an unbounded weight.

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
    1. Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
       MIDDoE: An MBDoE Python package for model identification, discrimination,
       and calibration.
       *Digital Chemical Engineering*, 17, 100276.
       https://doi.org/10.1016/j.dche.2025.100276

    2. Bard, Y. (1974).
       *Nonlinear Parameter Estimation*. Academic Press, New York.

    3. Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.
       https://doi.org/10.1016/j.ces.2007.11.034

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

    _normalize_iden_opt(iden_opt)
    if not isinstance(models.get('krt'), dict):
        raise ValueError("models['krt'] must be a dict {model_name: kind}, got "
                         f"{type(models.get('krt')).__name__}")
    data = data if data is not None else read_excel()
    _check_data(data, system)
    # case=None / 'freeze': keep existing masks; only create missing ones (all free)
    if 'mutation' not in models or models['mutation'] is None:
        models['mutation'] = {sv: [True] * len(models['theta'][sv]) for sv in models['can_m']}
    _initialize_dictionaries(models, iden_opt)

    active_models = models['can_m']
    bound_max, bound_min = models['t_u'], models['t_l']
    mutation = models['mutation']

    method = iden_opt['meth']
    objf = iden_opt['ob']
    multi = iden_opt.get('ms', False)

    varcov = iden_opt['var-cov']
    bootstrap = (varcov == 'B')
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

    # thetac = |theta|, x0 = sign(theta): negative nominal values keep their sign
    x0_dict, thc = {}, {}
    for sv in active_models:
        thc[sv], x0 = _signed_scaling(theta_params_f[sv], mutation[sv], sv)
        x0_dict[sv] = x0.tolist()
    # signed values of the fixed parameters (multi-start keeps them; same source as the
    # single start: models['thetastart'] for case='strov', else models['theta'])
    theta_fixed = {sv: (np.asarray(x0_dict[sv]) * thc[sv]).tolist() for sv in active_models}
    theta_params_f = {**theta_params_f, **{sv: thc[sv].tolist() for sv in active_models}}
    opts = _optim_opts({k: iden_opt.get(k) for k in ('pscale', 'grad', 'grad_step', 'fnorm',
                                                     'boot_start')})
    opts['theta_fixed'] = theta_fixed

    if bootstrap:
        return _bootstrap_runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                                 objf, x0_dict, method, data, system, models,
                                 logging, nboot, multi, varcov, maxit, tol, opts)

    if multi:
        nstarts = max(1, int(0.7 * mp.cpu_count()))
        return _multi_start_runner(active_models, theta_params_r, bound_max, bound_min, mutation,
                                   objf, method, data, system, models, logging, varcov, maxit, tol,
                                   nstarts, opts)

    return _runner(active_models, theta_params_f, bound_max, bound_min, mutation,
                   objf, x0_dict, method, data, system, models, logging, varcov, maxit, tol, opts)



# ============================================================
# Multi-start machinery
# ============================================================

def _fixed_values(opts, models):
    """Signed values of the fixed parameters for multi-start: opts['theta_fixed'] (set by
    parmest from the same source as the single start) or, if absent, models['theta']."""
    tf = (opts or {}).get('theta_fixed')
    return tf if tf is not None else models['theta']


def _run_single_start(sv, thetac, thetamin, thetamax, thetas, objf,
                      method, data, system, models, logging, varcov, maxit, tol, opts=None):
    r"""
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
        thc, x0 = _signed_scaling(thetac, thetas, sv)
        if logging:
            print(f"[{sv}] Sampled theta: {list(thetac)}")
        res = _runner([sv], {sv: thc.tolist()}, {sv: thetamax}, {sv: thetamin}, {sv: thetas},
                      objf, {sv: x0.tolist()}, method, data, system, models, logging, varcov, maxit, tol,
                      opts)
        if logging:
            print(f"[{sv}] Finished with result: {res}")
        return (sv, res)
    except Exception as e:
        import traceback, sys
        tb = ''.join(traceback.format_exception(*sys.exc_info()))
        return (sv, e, tb)


def _raise_if_all_starts_failed(active_models, allr, errs, nstarts):
    """Raise RuntimeError when no start of any model succeeded."""
    if any(getattr(r, 'success', False) for sv in active_models for r in allr[sv]):
        failed = [sv for sv in active_models if not any(getattr(r, 'success', False) for r in allr[sv])]
        if failed:
            warnings.warn(f"parmest (multi-start): no successful start for model(s) {failed}; "
                          f"they are returned as None. First errors: "
                          + "; ".join(f"{sv}: {errs[sv][:1]}" for sv in failed))
        return
    msg = "; ".join(f"{sv}: {errs[sv][0] if errs[sv] else 'no start converged (success=False)'}"
                    for sv in active_models)
    raise RuntimeError(f"parmest (multi-start): every model failed in all {nstarts} starts. {msg}")


def _multi_start_runner(active_models, _unused, bmax, bmin, mutation, objf,
                        method, data, system, models, logging, varcov, maxit, tol, nstarts=None,
                        opts=None):
    r"""
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
    errs = {sv: [] for sv in active_models}
    tasks = []
    fixed = _fixed_values(opts, models)
    for sv in active_models:
        for _ in range(nstarts):
            thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=fixed)[sv]
            tasks.append((sv, thetac, bmin[sv], bmax[sv], mutation[sv], objf, method,
                          data, system, models, logging, varcov, maxit, tol, opts))

    with ProcessPoolExecutor(max_workers=nwork, mp_context=CTX) as exe:
        for fut in as_completed([exe.submit(_run_single_start, *t) for t in tasks]):
            out = fut.result()
            # Crash in worker?
            if len(out) == 3 and isinstance(out[1], Exception):
                sv, e, tb = out
                errs[sv].append(f"{type(e).__name__}: {e}")
                if logging:
                    print(f"[{sv}] worker crashed:\n{tb}")
                continue
            sv, res = out
            if res and res.get(sv):
                allr[sv].append(res[sv])

    _raise_if_all_starts_failed(active_models, allr, errs, nstarts)
    return {
        sv: (min([r for r in allr[sv] if r.success], key=attrgetter("fun"))
             if any(r.success for r in allr[sv]) else None)
        for sv in active_models
    }


def _multi_start_runner_serial(active_models, _unused, bmax, bmin,
                               mutation, objf, method, data, system, models,
                               logging, varcov, maxit, tol, nstarts=None, opts=None):
    r"""
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
    errs = {sv: [] for sv in active_models}

    fixed = _fixed_values(opts, models)
    for sv in active_models:
        for _ in range(nstarts):
            thetac = _initialise_theta_parameters(bmin, bmax, [sv], mutation=mutation, theta_fixed=fixed)[sv]

            if logging:
                print(f"[{sv}][SERIAL] Trying theta: {thetac}")
            try:
                thc, x0 = _signed_scaling(thetac, mutation[sv], sv)
                res = _runner([sv], {sv: thc.tolist()}, {sv: bmax[sv]}, {sv: bmin[sv]}, {sv: mutation[sv]},
                              objf, {sv: x0.tolist()}, method, data, system, models, logging, varcov, maxit, tol,
                              opts)
            except Exception as e:
                errs[sv].append(f"{type(e).__name__}: {e}")
                continue
            if res and res.get(sv):
                allr[sv].append(res[sv])

    _raise_if_all_starts_failed(active_models, allr, errs, nstarts)
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
                      boot_idx, opts=None):
    r"""
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
        ref = _multi_start_runner_serial(active_models, None, bmax, bmin,
                                         mutation, objf, method, data, system,
                                         models_boot, logging, varcov, maxit, tol, opts=opts)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf,
                      x0, method, data, system, models_boot, logging, varcov, maxit, tol, opts)
        n_extra = int((opts or {}).get('_boot_extra', 0) or 0)
        if n_extra > 0:
            # boot_start='mixed': also try a few random starts and keep the better fit per model
            try:
                extra = _multi_start_runner_serial(active_models, None, bmax, bmin,
                                                   mutation, objf, method, data, system,
                                                   models_boot, logging, varcov, maxit, tol,
                                                   nstarts=n_extra, opts=opts)
            except RuntimeError:
                extra = {}
            for sv in active_models:
                r0, r1 = ref.get(sv), extra.get(sv)
                ok0 = r0 is not None and getattr(r0, 'success', True) and np.isfinite(r0.fun)
                if r1 is not None and (not ok0 or r1.fun < r0.fun):
                    ref[sv] = r1
    return ref


def truncated_moments_from_data(X, eps=1e-10, lo=None, hi=None, rtol=1e-6):
    r"""
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
    lo, hi : array-like, optional
        Lower/upper parameter bounds (one per column of X). A dimension is
        truncated at a bound only if the samples actually reach that bound
        (within ``rtol``); otherwise that side is left untruncated. With
        ``lo``/``hi`` None no truncation is applied on that side.
    rtol : float, optional
        Relative tolerance for deciding that a bound was hit (default 1e-6).

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
        2. Truncate only at the parameter bounds (lo/hi) that the samples actually hit.
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
    1. Burkardt, J. (2014). The Truncated Normal Distribution.
       Department of Scientific Computing, Florida State University.

    2. scipy.stats.truncnorm documentation:
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
        # truncate only at parameter bounds that are actually hit
        a, b = -np.inf, np.inf
        if lo is not None and np.isfinite(lo[i]) and bmin[i] <= lo[i] + rtol * max(abs(lo[i]), 1e-300):
            a = (lo[i] - loc) / scale
        if hi is not None and np.isfinite(hi[i]) and bmax[i] >= hi[i] - rtol * max(abs(hi[i]), 1e-300):
            b = (hi[i] - loc) / scale
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
    r"""
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
                      nboot, multi, varcov, maxit, tol, opts=None):
    r"""
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
        4. **Quality Filtering**: Keep only successful replicates with a finite objective.
           If every replicate is rejected a RuntimeError is raised.
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

    **Replicate filter**:
    Replicates that failed (exception or success=False), have a non-finite
    objective, or whose optimum is more than twice the reference optimum (compared
    on the constant-free objective, so MLE works too) are rejected; the counts are
    printed and stored in ``boot_counts``.

    **Replicate start** (``opts['boot_start']``):
    'mixed' (default) starts every replicate from the reference estimate and, when
    ``multi``, also from _BOOT_EXTRA random points, keeping the best fit; 'ref' uses
    the reference estimate only; 'multi' repeats the reference fit's starts.

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
        nstarts = max(1, int(0.7 * mp.cpu_count()))
        ref = _multi_start_runner(active_models, None, bmax, bmin,
                                  mutation, objf, method, data, system, models,
                                  logging, varcov, maxit, tol, nstarts, opts)
    else:
        ref = _runner(active_models, theta_params, bmax, bmin, mutation,
                      objf, x0, method, data, system, models, logging,
                      varcov, maxit, tol, opts)

    # Bootstrap sampling
    N = _count_observations(data, system)
    rng = np.random.default_rng()
    boot_indices = [rng.integers(0, N, size=N, endpoint=False) for _ in range(nboot)]

    allP = {sv: [] for sv in active_models}
    bootR = {sv: [] for sv in active_models}

    # Starting points of the replicates. 'ref': the reference optimum (a replicate is a refit
    # of slightly different data, so its optimum is usually close); 'mixed': the reference
    # optimum plus a few random starts, which also finds a second, distant minimum of a
    # poorly identifiable parameter; 'multi': as the reference fit (multi-start / initial
    # guess). A model whose reference fit failed, or that has a free estimate of exactly 0
    # (cannot be scaled), falls back to 'multi'.
    rep_theta, rep_x0, rep_multi = theta_params, x0, multi
    boot_start = (opts or {}).get('boot_start', 'mixed')
    rep_opts = opts
    if boot_start in ('ref', 'mixed'):
        try:
            th, xs = {}, {}
            for sv in active_models:
                r0 = ref.get(sv)
                if r0 is None or not np.all(np.isfinite(np.asarray(r0.scpr, float))):
                    raise ValueError(sv)
                thc_sv, x0_sv = _signed_scaling(np.asarray(r0.scpr, float), mutation[sv], sv)
                th[sv], xs[sv] = thc_sv.tolist(), x0_sv.tolist()
            rep_theta, rep_x0, rep_multi = th, xs, False
            if boot_start == 'mixed' and multi:
                rep_opts = {**(opts or {}), '_boot_extra': _BOOT_EXTRA}
        except (ValueError, AttributeError):
            pass
    args_common = (active_models, rep_theta, bmax, bmin, mutation,
                   objf, rep_x0, method, data, system, models, logging,
                   rep_multi, varcov, maxit, tol)

    # Acceptance test of a replicate: its own optimum on the resampled data must be no worse
    # than twice the reference optimum, compared on the constant-free objective (fun_fit;
    # for MLE the constant 0.5*sum(log 2 pi sigma^2) is negative and would break a ratio).
    # A resample of the same N points has an optimum of the same size as the reference
    # (WLS ~ chi2: ~N +- sqrt(2N)), so this rejects optimisations that stopped far from the
    # minimum, not unlucky resamples; without it, stalled fits inflate the bootstrap spread.
    def _fit_value(r):
        v = getattr(r, 'fun_fit', None)
        return float(r.fun if v is None else v)

    n_failed = 0
    n_unsucc = {sv: 0 for sv in active_models}
    n_reject = {sv: 0 for sv in active_models}
    with ProcessPoolExecutor(max_workers=min(nboot, mp.cpu_count()), mp_context=CTX) as exe:
        futures = [exe.submit(_bootstrap_worker, *args_common, bidx, rep_opts) for bidx in boot_indices]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                if logging:
                    print(f"bootstrap replicate failed: {type(e).__name__}: {e}")
                continue
            for sv in active_models:
                r = res.get(sv)
                if ref.get(sv) is None:
                    continue
                if not (r and getattr(r, 'success', True) and np.isfinite(r.fun)):
                    n_unsucc[sv] += 1
                    continue
                f_ref = _fit_value(ref[sv])
                limit = 2.0 * f_ref + 1e-12 * max(abs(f_ref), 1.0)
                if _fit_value(r) > limit:
                    n_reject[sv] += 1
                    continue
                if logging:
                    print(f'bootstrap objective {_fit_value(r):.4g} (reference {f_ref:.4g}) for {sv}')
                bootR[sv].append(r)
                allP[sv].append(r.scpr)
    for sv in active_models:
        if ref.get(sv) is None:
            continue
        print(f"[parmest] bootstrap '{sv}': {len(bootR[sv])}/{nboot} replicates kept "
              f"({n_reject[sv]} rejected: objective > 2x the reference fit; "
              f"{n_unsucc[sv]} not converged; {n_failed} raised an error)")

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
            mu_trunc_masked, cov_trunc_masked = truncated_moments_from_data(
                X_masked, lo=np.asarray(bmin[sv], float)[mask], hi=np.asarray(bmax[sv], float)[mask])

            # Build full-length covariance matrix with zeros in masked rows/cols
            v_full = np.zeros((X_full.shape[1], X_full.shape[1]), dtype=np.float64)
            v_full[np.ix_(mask, mask)] = cov_trunc_masked

            ref_sv.scpr_raw = ref_sv.scpr
            ref_sv.X = X_full  # full sample matrix
            ref_sv.scpr = mu_raw  # full mean, including fixed params
            ref_sv.v = v_full  # full matrix, only active filled
            ref_sv.v_raw = np.cov(X_full, rowvar=False, ddof=1)
            ref_sv.boot_counts = {'requested': nboot, 'kept': len(bootR[sv]),
                                  'rejected_2x': n_reject[sv], 'not_converged': n_unsucc[sv],
                                  'errors': n_failed}

        else:
            raise RuntimeError(
                f"Bootstrap for model '{sv}': all {nboot} replicates were rejected "
                f"({n_reject[sv]} with an objective > 2x the reference fit, {n_unsucc[sv]} not "
                f"converged, {n_failed} raised an error); no bootstrap covariance can be computed.")
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

# Numerical options of the optimiser (iden_opt keys of the same name; see parmest docstring)
_OPT_DEFAULTS = {
    'pscale': 'log',       # 'log': optimise log|x| for parameters whose bounds exclude 0; 'lin': x
    'grad': '2-point',     # finite-difference gradient: '2-point', '3-point', or 'legacy'
                           # (SciPy default: forward differences with absolute step 1e-8)
    'grad_step': 1e-6,     # relative step of the '2-point'/'3-point' gradient
    'fnorm': True,         # minimise the constant-free objective divided by its own size
    'theta_fixed': None,   # {model: signed values} for fixed parameters in multi-start
    'boot_start': 'mixed', # start of each bootstrap replicate: 'mixed' = reference fit plus
                           # _BOOT_EXTRA random starts (with multi-start), 'ref' = reference fit
                           # only, 'multi' = as the reference fit (multi-start / initial guess)
}
_BOOT_EXTRA = 4            # random starts added to the reference start with boot_start='mixed'


def _optim_opts(opts=None):
    """Complete and validate the optimiser options (defaults: _OPT_DEFAULTS)."""
    o = dict(_OPT_DEFAULTS)
    if opts:
        o.update({k: v for k, v in opts.items() if k in o and (v is not None or k == 'theta_fixed')})
    if o['pscale'] not in ('log', 'lin'):
        raise ValueError(f"iden_opt['pscale'] must be 'log' or 'lin', got {o['pscale']!r}")
    if o['grad'] not in ('3-point', '2-point', 'legacy'):
        raise ValueError(f"iden_opt['grad'] must be '3-point', '2-point' or 'legacy', got {o['grad']!r}")
    gs = o['grad_step']
    if not (isinstance(gs, (int, float)) and np.isfinite(gs) and 0 < gs < 0.1):
        raise ValueError(f"iden_opt['grad_step'] must be a number in (0, 0.1), got {gs!r}")
    o['fnorm'] = bool(o['fnorm'])
    if o['boot_start'] not in ('mixed', 'ref', 'multi'):
        raise ValueError(f"iden_opt['boot_start'] must be 'mixed', 'ref' or 'multi', "
                         f"got {o['boot_start']!r}")
    return o


def _optimise(method, x0, free, lo, hi, args, opts, maxit, tol):
    r"""
    Minimise the objective over the free normalised parameters ``x[free]``.

    Parameters
    ----------
    method : str
        'SLSQP', 'LMBFGS', 'TC', 'NMS', 'BFGS' or 'DE'.
    x0 : np.ndarray
        Full normalised start vector (+-1 for every parameter); fixed entries are kept.
    free : np.ndarray[int]
        Indices of the free parameters.
    lo, hi : np.ndarray
        Bounds of ``x[free]`` (normalised).
    args : tuple
        ``_objective_function`` arguments after ``x``.
    opts : dict
        Output of :func:`_optim_opts`.
    maxit, tol : int, float
        Iteration limit (per pass) and tolerance (meaning per method, see parmest).

    Returns
    -------
    OptimizeResult
        ``x`` is the optimal ``x[free]`` (normalised, not the internal variables).

    Notes
    -----
    * Internal variables: see :class:`_OptProblem` (``pscale='log'``: log|x| where the
      bounds exclude 0).
    * Gradient (SLSQP, LMBFGS, TC, BFGS): SciPy finite differences with a *relative*
      step ``grad_step * max(1, |z|)``, kept inside the bounds. SciPy's default
      (forward differences, absolute step 1e-8) is mostly noise for parameters to which
      the objective is weakly sensitive.
    * ``fnorm``: the objective is divided by a scale s and, if it fell below 0.1 s, the
      optimiser is restarted from its result with s = the new objective (at most 5 passes).
      SLSQP, NMS, TC, BFGS (absolute stopping tests): s = f(start), so the tests act
      relative to the objective at the optimum. LMBFGS (relative 'ftol', absolute only
      for |f| < 1): s = min(f(start), 1), i.e. only objectives below 1 are scaled up.
      Either way the tests do not depend on units or on the size of sigma. Without it
      (legacy) only SLSQP is scaled, by max(|f(x0)|, 1).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if opts['pscale'] == 'log':
        log_dims = ((lo > 0) & (hi > 0)) | ((lo < 0) & (hi < 0))
    else:
        log_dims = np.zeros(lo.size, bool)
    sign = np.where(hi < 0, -1.0, 1.0)
    prob = _OptProblem(x0, free, log_dims, sign, args)

    xs = np.asarray(x0, float)[free].copy()
    xs[log_dims] = np.clip(xs[log_dims], lo[log_dims], hi[log_dims])   # a log variable needs x inside
    z0 = prob.z_of(xs)
    zlo, zhi = lo.copy(), hi.copy()
    pos = log_dims & (sign > 0)
    neg = log_dims & (sign < 0)
    with np.errstate(divide='ignore'):
        zlo[pos], zhi[pos] = np.log(lo[pos]), np.log(hi[pos])
        zlo[neg], zhi[neg] = np.log(-hi[neg]), np.log(-lo[neg])
    bounds = list(zip(zlo, zhi))

    if opts['grad'] == 'legacy':
        jac, fd = None, {}
    else:
        jac, fd = opts['grad'], {'finite_diff_rel_step': opts['grad_step']}

    def run(z_start, scale):
        prob.scale = scale
        if method == 'SLSQP':
            return minimize(prob, z_start, method='SLSQP', jac=jac, bounds=bounds,
                            options={'maxiter': maxit, 'ftol': tol, 'disp': False, **fd})
        if method == 'LMBFGS':
            return minimize(prob, z_start, method='L-BFGS-B', jac=jac, bounds=bounds,
                            options={'maxiter': maxit, 'ftol': tol, 'disp': False, **fd})
        if method == 'TC':
            return minimize(prob, z_start, method='trust-constr', jac=jac or '2-point', bounds=bounds,
                            options={'maxiter': maxit, 'xtol': 1e-10, 'gtol': tol, **fd})
        if method == 'NMS':
            nm = {'maxiter': maxit, 'fatol': tol, 'disp': False}
            if log_dims.any():
                # SciPy's default simplex: +5 % of each non-zero coordinate (0.00025 if 0).
                # For a log variable the same +5 % of theta is a step log(1.05) in z.
                step = np.where(z_start != 0, 0.05 * z_start, 0.00025)
                step[log_dims] = np.log(1.05)
                nm['initial_simplex'] = np.vstack([z_start, z_start + np.diag(step)])
            return minimize(prob, z_start, method='Nelder-Mead', bounds=bounds, options=nm)
        if method == 'BFGS':
            # SciPy's BFGS takes no bounds; fixed parameters are respected via _OptProblem
            return minimize(prob, z_start, method='BFGS', jac=jac,
                            options={'maxiter': maxit, 'disp': False, **fd})
        if method == 'DE':
            # Avoid nested pools: DE in a worker process uses 1 worker
            n_workers = 1 if _in_child() else -1
            return differential_evolution(prob, bounds, maxiter=maxit, popsize=18, tol=1e-8,
                                          strategy='best1bin', mutation=(0.5, 1.5),
                                          recombination=0.7, polish=False,
                                          updating='deferred', workers=n_workers)
        raise ValueError(f"Unknown method '{method}'; valid: {list(_VALID_METH)}")

    f0 = float(prob(z0))           # prob.scale is 1 here
    if opts['fnorm']:
        f0_ok = f0 if (np.isfinite(f0) and f0 > 0) else 1.0
        # L-BFGS-B's 'ftol' test is relative, (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1), but becomes
        # absolute once |f| < 1: scale only small objectives UP (scale <= 1) so it stays relative.
        # SLSQP 'ftol', NMS 'fatol', TC 'gtol' (and BFGS' gtol) are absolute: scale to the
        # objective's own size so they become relative.
        scale = min(f0_ok, 1.0) if method == 'LMBFGS' else f0_ok
    else:
        scale = max(abs(f0), 1.0) if method == 'SLSQP' else 1.0
    res = run(z0, scale)
    f_res = float(res.fun) * scale
    nit = int(res.get('nit', 0) or 0)
    nfev = int(res.get('nfev', 0) or 0)
    if opts['fnorm'] and method != 'DE':
        for _ in range(4):         # restart passes
            if not (np.isfinite(f_res) and 0 < f_res < 0.1 * scale):
                break
            scale = f_res
            res2 = run(np.asarray(res.x, float), scale)
            nit += int(res2.get('nit', 0) or 0)
            nfev += int(res2.get('nfev', 0) or 0)
            f2 = float(res2.fun) * scale
            if np.isfinite(f2) and f2 <= f_res:
                res2.success = bool(res2.success or res.success)
                res, f_res = res2, f2
            else:
                break
        # A gradient method that ends in a failed line search (e.g. L-BFGS-B
        # 'ABNORMAL_TERMINATION_IN_LNSRCH') has usually reached the precision of the
        # finite-difference gradient. Restart once from the result with a fresh memory: keep
        # the restart if it converges; if it cannot lower the objective at all, the point is
        # a minimum to numerical precision and is reported as converged. An iteration limit
        # is not touched (that is a real "not converged").
        msg = str(res.get('message', ''))
        hit_limit = any(k in msg.upper() for k in ('ITERATIONS REACHED LIMIT', 'ITERATION LIMIT',
                                                   'MAXIMUM NUMBER OF ITERATIONS'))
        if (method in ('LMBFGS', 'SLSQP', 'BFGS') and not res.success and not hit_limit
                and np.isfinite(f_res) and f_res > 0):
            scale = f_res if method != 'LMBFGS' else min(f_res, 1.0)
            res2 = run(np.asarray(res.x, float), scale)
            nit += int(res2.get('nit', 0) or 0)
            nfev += int(res2.get('nfev', 0) or 0)
            f2 = float(res2.fun) * scale
            if np.isfinite(f2) and f2 <= f_res:
                if not res2.success and f2 >= f_res * (1 - 1e-12):
                    res2.success = True
                    res2.message = (f"converged to numerical precision (a restart from the result "
                                    f"could not lower the objective; first stop: {msg})")
                res, f_res = res2, f2
    res.nit, res.nfev = nit, nfev
    res.fun = f_res
    res.x = prob.x_free(res.x)
    return res


def _runner(active_models, theta_params, bmax, bmin, mutation, objf,
            x0_dict, method, data, system, models, logging, varcov, maxit, tol, opts=None):
    r"""
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
        Estimation results for each model (models that failed are omitted;
        RuntimeError if all models fail).
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
    Only the free sub-vector (mutation[i] = True) is passed to the optimiser, for
    every method, so fixed parameters stay exactly at x0[i]. Bounds are passed to
    SLSQP, L-BFGS-B, trust-constr, Nelder-Mead and DE; SciPy's BFGS takes no bounds.

    **Hessian Computation** (only when varcov='H' and not bootstrap, all methods):
    The Hessian of the objective with respect to the free scaled parameters,
    \( \mathbf{H}_x \), is computed by central differences in self-whitened coordinates
    (see ``_whitened_hessian``; robust for strongly correlated parameters) and
    converted to physical units: \( \mathbf{D}_c \mathbf{H}_x^{-1} \mathbf{D}_c \),
    \( \mathbf{D}_c = \text{diag}(\theta_c) \) (free block; fixed rows/columns are 0).
    It is stored as ``hess_inv``; ``uncert`` applies the objective-specific factor.

    **Differential Evolution (DE)**:
    To avoid nested parallelization, DE uses workers=1 inside child processes
    (_in_child() check) and workers=-1 (all cores) in main process.

    **Error Handling**:
    If estimation of a model fails (exception raised), the model is skipped with a
    warning so that rival models can proceed. If every model fails, a RuntimeError
    listing the per-model errors is raised.

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
    opts = _optim_opts(opts)
    results = {}
    errors = {}
    for sv in active_models:
        thetac = np.array(theta_params[sv], float)
        x0 = np.array(x0_dict[sv], float)
        mask = np.array(mutation[sv], bool)
        free = np.where(mask)[0]

        bootstrap_flag = models.get('__bootstrap__', False)
        boot_idx = models.get('__boot_idx__', None)

        args = (data, [sv], thetac, system, models, logging, objf,
                bootstrap_flag, boot_idx)
        # constant-free version of the objective (MLE -> WLS/2): what the optimiser minimises
        # (unless opts['fnorm'] is False: legacy) and what the bootstrap filter compares
        args_fit = args[:6] + (_FIT_OBJ[objf],) + args[7:]
        args_opt = args_fit if opts['fnorm'] else args

        try:
            if np.any(thetac[free] <= 0) or not np.all(np.isfinite(thetac)):
                raise ValueError(f"scaling values thetac must be finite and > 0 for free parameters, "
                                 f"got {thetac.tolist()} (use thetac=|theta|, x0=sign(theta))")
            # Bounds in scaled space, only for the free sub-vector (fixed parameters
            # are not optimisation variables, so no optimiser can move them).
            thetamax = np.array(bmax[sv], float)[free] / thetac[free]
            thetamin = np.array(bmin[sv], float)[free] / thetac[free]
            if np.any(thetamin > thetamax):
                raise ValueError(f"lower bound above upper bound for free parameter(s) "
                                 f"{free[thetamin > thetamax].tolist()} of model '{sv}'")

            if free.size == 0:
                res = OptimizeResult(x=np.array([]),
                                     fun=_objective_free(np.array([]), x0, free, *args),
                                     success=True, status=0, nit=0,
                                     message='no free parameters: objective evaluated at the fixed values')
                res.fun_fit = _objective_free(np.array([]), x0, free, *args_fit)
            else:
                res = _optimise(method, x0, free, thetamin, thetamax, args_opt, opts, maxit, tol)
                xf = np.asarray(res.x, float).ravel()
                # fun: the user's objective (MLE including its constant); fun_fit: constant-free
                res.fun_fit = float(_objective_free(xf, x0, free, *args_fit))
                res.fun = (res.fun_fit if _FIT_OBJ[objf] == objf
                           else float(_objective_free(xf, x0, free, *args)))

            # Expand the free sub-vector back to the full scaled vector
            x_full = x0.copy()
            x_full[free] = np.asarray(res.x, float).ravel()
            res.x = x_full
            # Optimiser-internal inverse Hessians live in the free, scaled space;
            # they are not a parameter covariance, so drop them.
            if 'hess_inv' in res:
                del res['hess_inv']

            # Scale back to original theta
            res.scpr = res.x * thetac
            res.activeparams = mask

            # Covariance from the numerical Hessian of the objective (var-cov='H'),
            # only on reference fits (not bootstrap replicates). Taken over the free
            # parameters only; Dc converts from scaled x to physical theta.
            # V = Dc Hx^-1 Dc is the physical inverse Hessian; uncert applies the
            # objective-specific factor (WLS: 2, MLE: 1, LS/Chi: 2*fun/dof).
            if not bootstrap_flag and varcov == 'H':
                cov = np.zeros((len(thetac), len(thetac)))
                if free.size:
                    def loss_free(xf, _xr=res.x, _free=free):
                        x = _xr.copy()
                        x[_free] = xf
                        return _objective_function(x, data, [sv], thetac, system, models,
                                                   logging, objf, False, None)
                    Hx = _whitened_hessian(loss_free, res.x[free])
                    Dc = np.diag(thetac[free])
                    res.hessian = Hx
                    cov[np.ix_(free, free)] = Dc @ np.linalg.pinv(Hx) @ Dc
                res.hess_inv = cov
        except Exception as e:
            errors[sv] = f"{type(e).__name__}: {e}"
            if logging:
                print(f"[{sv}] optimize error: {e}")
            continue

        results[sv] = res

    if errors:
        if not results:
            raise RuntimeError("parmest: every model failed. " +
                               "; ".join(f"{k}: {v}" for k, v in errors.items()))
        warnings.warn("parmest: model(s) skipped because estimation failed: " +
                      "; ".join(f"{k}: {v}" for k, v in errors.items()))
    return results


def _match_times(t_all, times, var=''):
    """
    Index of the X:all node matching each measurement time (nearest node,
    tolerance 1e-9*max(1, t_end)). Works for unsorted and duplicated MES_X;
    raises ValueError for times not on the simulation grid.
    """
    t_all = np.asarray(t_all, float)
    times = np.asarray(times, float)
    if t_all.size == 0:
        raise ValueError(f"MES_X:{var}: the X:all column is empty")
    atol = 1e-9 * max(1.0, abs(t_all[-1]))
    pos = np.clip(np.searchsorted(t_all, times), 0, len(t_all) - 1)
    prev = np.maximum(pos - 1, 0)
    pos = np.where(np.abs(t_all[prev] - times) < np.abs(t_all[pos] - times), prev, pos)
    ok = np.isclose(t_all[pos], times, rtol=0, atol=atol)
    if not np.all(ok):
        raise ValueError(f"MES_X:{var} contains times not in X:all: {times[~ok][:5].tolist()}")
    return pos


def _check_data(data, system, rtol=1e-9):
    """
    Data-file checks:
      (a) a tii column must be constant (only row 0 is used)          -> ValueError
      (b) with CVP = no_CVP a tvi column should be constant            -> warning
      (c) the CPF/LPF profile rebuilt from <tvi>t/<tvi>l must match
          the <tvi> column (detects switch times in normalised units)   -> ValueError
      (d) rows with MES_Y but no MES_X are dropped                     -> warning
      (e) a missing tii column                                        -> ValueError
    """
    import pandas as _pd
    for name, sh in data.items():
        # (a)/(e) time-invariant inputs
        for v in system.get('tii', {}):
            if v not in sh:
                raise ValueError(f"Sheet '{name}': time-invariant input column '{v}' is missing")
            col = _pd.to_numeric(sh[v], errors='coerce').dropna().values.astype(float)
            if col.size == 0 or _pd.isna(sh[v].iloc[0]):
                raise ValueError(f"Sheet '{name}': time-invariant input '{v}' has no value in row 0")
            if np.ptp(col) > rtol * max(1.0, abs(col[0])):
                raise ValueError(f"Sheet '{name}': time-invariant input column '{v}' is not constant; "
                                 f"only its first value would be used")
        X = (_pd.to_numeric(sh['X:all'], errors='coerce').dropna().values.astype(float)
             if 'X:all' in sh else np.array([]))
        # (b)/(c) time-variant inputs
        for v in system.get('tvi', {}):
            ckey = f"CVP:{v}"
            if ckey not in sh or v not in sh:
                continue
            cvp = sh[ckey].iloc[0]
            col = _pd.to_numeric(sh[v], errors='coerce').dropna().values.astype(float)
            if col.size == 0:
                continue
            if cvp == 'no_CVP':
                if np.ptp(col) > rtol * max(1.0, abs(col[0])):
                    warnings.warn(f"Sheet '{name}': input '{v}' varies but CVP=no_CVP; only its "
                                  f"first value ({col[0]}) is used")
                continue
            if cvp not in ('CPF', 'LPF') or X.size == 0 or col.size != X.size:
                continue
            tk, lk = f"{v}t", f"{v}l"
            if tk not in sh or lk not in sh:
                continue
            ts = _pd.to_numeric(sh[tk], errors='coerce').dropna().values.astype(float)
            ls = _pd.to_numeric(sh[lk], errors='coerce').dropna().values.astype(float)
            if ts.size == 0 or ts.size != ls.size:
                continue
            order = np.argsort(ts, kind='stable')
            ts, ls = ts[order], ls[order]
            if cvp == 'CPF':
                k = np.clip(np.searchsorted(ts, X, side='right') - 1, 0, ls.size - 1)
                rebuilt = ls[k]
                # a node lying on a switch time is ambiguous (old or new level): skip it
                tol_t = 1e-9 * max(1.0, abs(X[-1]))
                near = np.min(np.abs(X[:, None] - ts[None, :]), axis=1) <= tol_t
            else:
                rebuilt = np.interp(X, ts, ls)
                near = np.zeros(X.size, bool)
            scale = max(1.0, float(np.max(np.abs(ls))), float(np.max(np.abs(col))))
            diff = np.abs(rebuilt - col)
            bad = (diff > 1e-6 * scale) & ~near
            if np.any(bad):
                raise ValueError(
                    f"Sheet '{name}': the {cvp} profile rebuilt from '{tk}'/'{lk}' does not match the "
                    f"'{v}' column at {int(bad.sum())} of {X.size} time points (max |diff| = "
                    f"{np.max(diff[bad]):.3g}). Switch times must be given in the same (physical) "
                    f"time units as X:all.")
        # (d) measurements without a time
        for v in system.get('tvo', {}):
            xc, yc = f"MES_X:{v}", f"MES_Y:{v}"
            if xc in sh and yc in sh:
                n = int((sh[xc].isna() & sh[yc].notna()).sum())
                if n:
                    warnings.warn(f"Sheet '{name}': {n} row(s) with MES_Y:{v} but no MES_X:{v} are ignored")
    # (f) sigma of measurements without an MES_E value
    _check_sigma_fallback(data, system)


def _unc_value(cfg):
    """system['tvo'|'tio'][var]['unc'] as a float (NaN if missing, None or not a number)."""
    try:
        u = float(cfg.get('unc', np.nan))
    except (TypeError, ValueError):
        return np.nan
    return u


def _check_sigma_fallback(data, system):
    r"""
    Check the measurements whose sigma must come from ``system[...]['unc']``.

    The weights of WLS/MLE (and the 'M' covariance) use the per-point sigma of the
    'MES_E:{var}' column. Where that column is missing (or empty for a measured row), the
    variable's ``'unc'`` is used **as an absolute standard deviation**. That is right for
    absolute errors, but for relative errors 'unc' is a fraction of the value (e.g. 0.08)
    and the weights would be wrong. So: raise if the fallback is needed and 'unc' is not a
    finite number > 0, and warn (once per call) which variables use it.
    """
    import pandas as _pd
    used = {}
    groups = (('tvo', True), ('tio', False))
    for name, sh in data.items():
        for grp, timed in groups:
            for v, cfg in system.get(grp, {}).items():
                if not cfg.get('meas', True):
                    continue
                yc, ec = f"MES_Y:{v}", f"MES_E:{v}"
                if yc not in sh:
                    continue
                if timed:
                    xc = f"MES_X:{v}"
                    if xc not in sh:
                        continue
                    rows = sh[xc].notna() & sh[yc].notna()
                else:
                    rows = _pd.Series([_pd.notna(sh[yc].iloc[0])])
                    if ec in sh:
                        rows = rows & _pd.Series([_pd.isna(sh[ec].iloc[0])])
                    if bool(rows.iloc[0]):
                        used.setdefault(v, []).append(name)
                    continue
                if ec in sh:
                    rows = rows & _pd.to_numeric(sh[ec], errors='coerce').isna()
                if bool(rows.any()):
                    used.setdefault(v, []).append(name)
    if not used:
        return
    bad = []
    for v in used:
        cfg = system.get('tvo', {}).get(v) or system.get('tio', {}).get(v) or {}
        u = _unc_value(cfg)
        if not (np.isfinite(u) and u > 0):
            bad.append(v)
    if bad:
        raise ValueError(f"No 'MES_E' sigma for {bad} (sheets {[used[v] for v in bad]}) and their "
                         f"system 'unc' is missing or not a finite number > 0: give the measurement "
                         f"standard deviation in an 'MES_E:<var>' column (or 'unc').")
    warnings.warn(f"No 'MES_E' column (or empty MES_E cells) for {sorted(used)}: their system 'unc' is "
                  f"used as the ABSOLUTE standard deviation of those points (sheets: {used}). For "
                  f"relative errors give the "
                  f"per-point sigma in 'MES_E:<var>' instead.")


# ============================================================
# Objective wrappers
# ============================================================

def _fd_hessian(fun, n, h):
    """Central-difference Hessian of ``fun`` at the origin of an n-dimensional space, step h."""
    H = np.zeros((n, n))
    f0 = fun(np.zeros(n))
    E = np.eye(n) * h
    fp = [fun(E[i]) for i in range(n)]
    fm = [fun(-E[i]) for i in range(n)]
    for i in range(n):
        H[i, i] = (fp[i] - 2.0 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = (fun(E[i] + E[j]) - fun(E[i] - E[j])
                                 - fun(-E[i] + E[j]) + fun(-E[i] - E[j])) / (4.0 * h * h)
    return H


def _whitened_hessian(fun, x, h0=1e-3, h_white=0.5, max_dx=0.2, n_iter=4, tol=0.05):
    """
    Hessian of ``fun`` at ``x`` that stays accurate for strongly correlated parameters.

    A plain finite-difference Hessian in the scaled parameters cannot resolve the
    smallest curvature when the parameters are strongly correlated (condition numbers
    of 1e3-1e4 are common), so it comes out indefinite and the covariance is NaN.
    Here the differences are taken in whitened coordinates z, with x = x* + A z and
    A = H^(-1/2) from the current Hessian estimate. In these coordinates every
    curvature is about 1, so one step size (``h_white``) suits all directions.

    1. Rough Hessian H0 by central differences in x with step ``h0``.
    2. Whiten with |eigenvalues| of the current estimate, difference in z with step
       ``h_white`` (each step changes x by at most ``max_dx``), map back:
       H = A^-T H_z A^-1.
    3. Repeat until H_z is within ``tol`` of the identity or ``n_iter`` is reached.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = _fd_hessian(lambda d: fun(x + d), n, h0)
    for _ in range(n_iter):
        Hs = 0.5 * (H + H.T)
        w, Q = np.linalg.eigh(Hs)
        w = np.abs(w)
        floor = max(w.max(), 1e-300) * 1e-12
        w = np.maximum(w, floor)
        A = Q @ np.diag(w ** -0.5) @ Q.T
        # cap the step: the largest change of any x for a step h_white along one z axis
        reach = h_white * np.max(np.abs(A))
        h = h_white if reach <= max_dx else h_white * max_dx / reach
        Hz = _fd_hessian(lambda z: fun(x + A @ z), n, h)
        Ainv = np.linalg.inv(A)
        H = Ainv.T @ Hz @ Ainv
        if np.linalg.norm(0.5 * (Hz + Hz.T) - np.eye(n), 2) < tol:
            break
    return 0.5 * (H + H.T)


def _objective_free(xf, x_base, free, *args):
    """Objective of the free sub-vector: fixed entries are taken from x_base
    (module level so that it can be pickled for parallel DE)."""
    x = np.array(x_base, dtype=float)
    x[free] = xf
    return _objective_function(x, *args)


def _objective_function(theta, data, active, thetac, system, models,
                        logging, objf, bootstrap, boot_idx):
    r"""
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
        Objective function to return: 'LS', 'WLS', 'MLE', or 'Chi'. Internally
        also 'MLE_fit' = WLS / 2, i.e. MLE without its parameter-independent
        constant 0.5*sum(log(2*pi*sigma^2)); same minimiser and same Hessian as
        'MLE', used by the optimiser so that relative stopping tests are not
        distorted by the constant (which can be large and negative).
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
    return {'LS': m['LS'], 'WLS': m['WLS'], 'MLE': m['MLE'], 'Chi': m['Chi'],
            'MLE_fit': 0.5 * m['WLS']}[objf]


# Objective actually minimised for each user objective: identical minimiser, no constant offset
_FIT_OBJ = {'LS': 'LS', 'WLS': 'WLS', 'MLE': 'MLE_fit', 'Chi': 'Chi'}


class _OptProblem:
    r"""
    Objective in the optimiser's internal variables ``z`` (free parameters only).

    MIDDoE's normalised parameters are ``x = theta / thetac`` (``thetac = |theta_start|``),
    so ``x`` starts at +-1. ``x`` is kept everywhere outside the optimiser (results,
    Hessian, bootstrap). Inside the optimiser a second, purely internal change of
    variables is used:

    * ``z = log(|x|)`` for free parameters whose bounds do not include 0 (both bounds
      > 0, or both < 0): equal steps in ``z`` are equal *relative* changes of theta,
      whatever the distance between the start and the optimum (e.g. a start of 100
      and an optimum of 9905 gives x ~ 99 but z ~ 4.6);
    * ``z = x`` for the other free parameters (bounds containing 0).

    The returned value is the fit objective divided by ``scale`` (set by the caller so
    that the optimiser's stopping tests act relative to the objective's size).
    Module level with plain attributes so that it pickles (parallel DE workers).
    """

    def __init__(self, x_base, free, log_dims, sign, args, scale=1.0):
        self.x_base = np.asarray(x_base, float)
        self.free = np.asarray(free, int)
        self.log_dims = np.asarray(log_dims, bool)
        self.sign = np.asarray(sign, float)
        self.args = args            # _objective_function arguments after x
        self.scale = float(scale)

    def x_free(self, z):
        z = np.asarray(z, float)
        xf = z.copy()
        ld = self.log_dims
        xf[ld] = self.sign[ld] * np.exp(z[ld])
        return xf

    def x_full(self, z):
        x = self.x_base.copy()
        x[self.free] = self.x_free(z)
        return x

    def z_of(self, xf):
        xf = np.asarray(xf, float)
        z = xf.copy()
        ld = self.log_dims
        z[ld] = np.log(np.abs(xf[ld]))
        return z

    def __call__(self, z):
        return _objective_function(self.x_full(z), *self.args) / self.scale


def _objective(theta, data, active, thetac, system, models,
               bootstrap=False, boot_idx=None):
    r"""
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
                Negative log-likelihood (paper eq. S6):
                \( \sum_i [\log(2\pi\sigma_i^2) + (r_i / \sigma_i)^2] / 2 \),
                where \( r_i = y_i - \hat{y}_i \) and \( \sigma_i \) is the per-point sigma.
            - 'Chi' : float
                Relative least squares: \( \sum_i [(y_i - \hat{y}_i) / |y_i|]^2 \)
                (|y| floored at 1e-8).
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
          Weights are \( w_i = 1 / \sigma_i^2 \) with the per-point \( \sigma_i \)
          from 'MES_E:{var}' (fallback: system['tvo'|'tio'][var]['unc']).

        - **MLE (Maximum Likelihood Estimation)**: Gaussian negative log-likelihood
          with per-point \( \sigma_i \) (paper eq. S6).

        - **Chi**: relative least squares, sum of squared relative errors, with ``|y|``
          floored at the point's sigma.

    All objectives are plain sums over all measurements (no per-response mean).

    **Missing Data Handling**:
        - Rows with NaN in MES_X or MES_Y are skipped.
        - Non-finite model output raises ValueError.
        - Measurement times are matched to the nearest node of 'X:all'; a time
          that is not on the grid raises ValueError.
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
    if not np.all(np.isfinite(np.asarray(theta, float))):
        raise ValueError(f"non-finite parameter vector passed to the objective: {np.asarray(theta).tolist()}")
    theta = np.asarray(theta, float).tolist()
    tv_i, ti_i = list(system['tvi']), list(system['tii'])
    tv_o = [v for v, c in system['tvo'].items() if c.get('meas', True)]
    ti_o = [v for v, c in system['tio'].items() if c.get('meas', True)]

    # fallback sigma where there is no MES_E value (checked and warned in _check_data)
    std_dev = {
        v: (_unc_value(cfg) if np.isfinite(_unc_value(cfg)) else 1.0)
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
            mask = ~sh[xc].isna().values & ~sh[yc].isna().values
            times = sh[xc][mask].values.astype(float)
            y_t = sh[yc][mask].values.astype(float)
            if times.size == 0:
                continue

            pos = _match_times(t_all, times, v)
            y_p = np.asarray(tvs[v], dtype=float)[pos]

            ye_col = f"MES_E:{v}"
            y_e = (sh[ye_col][mask].values.astype(float) if ye_col in sh
                   else np.full_like(y_t, std_dev[v], dtype=float))
            y_e = np.where(np.isnan(y_e), std_dev[v], y_e)    # empty MES_E cell -> 'unc'

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
            y_t = float(sh[yc].iloc[0])
            if np.isnan(y_t):
                continue
            y_p = float(np.ravel(tis.get(v, np.nan))[0])
            ecol = f"MES_E:{v}"
            sigma = (float(sh[ecol].iloc[0]) if ecol in sh and not _pd.isna(sh[ecol].iloc[0])
                     else float(std_dev[v]))
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
        all_y_true += ytl
        all_y_pred += ypl
        all_y_err += yel
        all_vars += [v] * len(ytl)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_err = np.array(all_y_err)
    all_vars = np.array(all_vars)
    N = len(all_y_true)
    if not (np.all(np.isfinite(all_y_true)) and np.all(np.isfinite(all_y_pred))):
        raise ValueError("non-finite residual in the objective (NaN/inf in the model output "
                         f"for model '{solver}', or in MES_Y)")
    if not (np.all(np.isfinite(all_y_err)) and np.all(all_y_err > 0)):
        raise ValueError("measurement sigma (MES_E column or system 'unc') must be finite and > 0")

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

        # True sums with per-point sigma (paper eqs. S5-S7); identical to uncert.
        r = yt - yp
        per_var['LS'][v] = np.sum(r ** 2)
        per_var['WLS'][v] = np.sum((r / ye) ** 2)
        per_var['MLE'][v] = 0.5 * np.sum(np.log(2 * np.pi * ye ** 2) + (r / ye) ** 2)
        # 'Chi': relative least squares sum(((y - y_hat)/max(|y|, sigma))^2). The floor at the
        # point's sigma stops a measured value near 0 (below the noise, e.g. a clipped negative
        # draw) from getting a weight ~1/eps^2 that overrides every other point.
        floor = np.maximum(np.nan_to_num(ye, nan=0.0), eps)
        per_var['Chi'][v] = np.sum((r / np.maximum(np.abs(yt), floor)) ** 2)

    LS, WLS, MLE, Chi = (float(sum(per_var[k].values())) for k in ('LS', 'WLS', 'MLE', 'Chi'))

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