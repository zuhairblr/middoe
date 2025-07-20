# import concurrent
# import pandas as pd
# import multiprocessing
# import time
# import numpy as np
# from numdifftools import Hessian
# import warnings
# from scipy.optimize import minimize, differential_evolution, BFGS
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from middoe.krnl_simula import simula
# from middoe.iden_utils import _initialize_dictionaries
# from operator import attrgetter
#
#
# warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")
#
# def _initialise_theta_parameters(theta_min, theta_max, active_models):
#     """
#     Initialise thetac by sampling within theta_min and theta_max for each model.
#     """
#     theta_parameters = {}
#     for solver in active_models:
#         theta_min_vals = theta_min[solver]
#         theta_max_vals = theta_max[solver]
#         sampled_thetac = [np.random.uniform(lo, hi) for lo, hi in zip(theta_min_vals, theta_max_vals)]
#         theta_parameters[solver] = sampled_thetac
#     return theta_parameters
#
# def _run_task_wrapper(args):
#     return _run_single_start(*args)
#
# # def parmest(system, models, iden_opt, data, case=None):
# #     _initialize_dictionaries(models, iden_opt)
# #
# #     active_models = models['can_m']
# #     bound_max = models['t_u']
# #     bound_min = models['t_l']
# #     mutation = models['mutation']
# #     method = iden_opt['meth']
# #     objf = iden_opt['ob']
# #     multi = iden_opt.get('ms', False)
# #     logging = iden_opt.get('log', False)
# #
# #     # Initialise theta_parameters by sampling inside min/max
# #     theta_parameters = _initialise_theta_parameters(bound_min, bound_max, active_models)
# #
# #     x0_dict = {solver: [1.0] * len(params) for solver, params in theta_parameters.items()}
# #
# #     total_cpus = multiprocessing.cpu_count()
# #     n_starts = max(1, int(0.7 * total_cpus)) if multi else 1
# #
# #     if multi:
# #         results = _multi_start_runner(
# #             active_models, theta_parameters, bound_max, bound_min, mutation,
# #             objf, method, data, system, models, logging, n_starts
# #         )
# #     else:
# #         results = _runner(
# #             active_models, theta_parameters, bound_max, bound_min, mutation,
# #             objf, x0_dict, method, data, system, models, logging
# #         )
# #
# #     return results
#
# def parmest(system, models, iden_opt, data, case=None):
#     _initialize_dictionaries(models, iden_opt)
#     active_models = models['can_m']
#     bound_max     = models['t_u']
#     bound_min     = models['t_l']
#     mutation      = models['mutation']
#     method        = iden_opt['meth']
#     objf          = iden_opt['ob']
#     multi         = iden_opt.get('ms', False)
#     bootstrap     = iden_opt.get('bootstrap', False)
#     nboot         = iden_opt.get('nboot', 100)
#     logging       = iden_opt.get('log', False)
#
#     theta_parameters = _initialise_theta_parameters(bound_min, bound_max, active_models)
#     x0_dict          = {sv: [1.0]*len(theta_parameters[sv]) for sv in active_models}
#
#     if bootstrap:
#         return _bootstrap_runner(
#             active_models, theta_parameters,
#             bound_max, bound_min, mutation,
#             objf, x0_dict, method,
#             data, system, models,
#             logging, nboot
#         )
#
#     if multi:
#         total_cpus = multiprocessing.cpu_count()
#         n_starts   = max(1, int(0.7*total_cpus))
#         return _multi_start_runner(
#             active_models, theta_parameters,
#             bound_max, bound_min, mutation,
#             objf, method, data,
#             system, models, logging,
#             n_starts
#         )
#
#     return _runner(
#         active_models, theta_parameters,
#         bound_max, bound_min, mutation,
#         objf, x0_dict, method,
#         data, system, models,
#         logging
#     )
#
# def _bootstrap_runner(
#     active_models, theta_parameters,
#     bound_max, bound_min, mutation,
#     objf, x0_dict, method,
#     data, system, models,
#     logging, nboot
# ):
#     # 1) reference fit
#     ref = _runner(active_models, theta_parameters, bound_max, bound_min,
#                   mutation, objf, x0_dict, method, data, system, models, logging)
#     all_params   = {sv: [] for sv in active_models}
#     boot_results = {sv: [] for sv in active_models}
#
#     def _one_boot(_):
#         boot_models = {**models, '__bootstrap__': True}
#         return _runner(active_models, theta_parameters, bound_max, bound_min,
#                        mutation, objf, x0_dict, method,
#                        data, system, boot_models, logging)
#
#     with ProcessPoolExecutor(max_workers=min(nboot, multiprocessing.cpu_count())) as exe:
#         futures = [exe.submit(_one_boot, i) for i in range(nboot)]
#         for fut in as_completed(futures):
#             res = fut.result()
#             for sv in active_models:
#                 r = res.get(sv)
#                 if r is not None:
#                     boot_results[sv].append(r)
#                     all_params[sv].append(r.scaled_x)
#
#     final = {}
#     for sv in active_models:
#         P = np.stack(all_params[sv], axis=0)
#         varcov = np.cov(P, rowvar=False, ddof=1) if P.size else None
#         final[sv] = {
#             'reference':    ref[sv],
#             'boot_results': boot_results[sv],
#             'varcov':       varcov
#         }
#     return final
#
# def _run_single_start(solver, thetac, thetamin, thetamax, thetas,
#                       objf, method, data, system, models, logging):
#     x0 =  [1.0] * len(thetac)
#     if logging:
#         print(f"[{solver}] Sampled x0: {x0}")
#
#     x0_dict = {solver: x0}
#     theta_param_new = {solver: thetac}  # Keep thetac fixed
#
#     result = _runner(
#         [solver], theta_param_new, {solver: thetamax}, {solver: thetamin}, {solver: thetas},
#         objf, x0_dict, method, data, system, models, logging
#     )
#     if solver not in result:
#         print(f"[{solver}] Optimization failed completely.")
#         return solver, None
#     return solver, result
#
# def _multi_start_runner(
#     active_models,
#     _unused_theta_parameters,  # accepted but unused
#     bound_max,
#     bound_min,
#     mutation,
#     objf,
#     method,
#     data,
#     system,
#     models,
#     logging,
#     n_starts=None
# ):
#     del _unused_theta_parameters  # clean unused argument
#
#     total_cpus = multiprocessing.cpu_count()
#     n_workers = max(1, int(0.7 * total_cpus))
#     if n_starts is None:
#         n_starts = max(2 * n_workers, 10)
#
#     all_results = {solver: [] for solver in active_models}
#     tasks = []
#
#     for solver in active_models:
#         for _ in range(n_starts):
#             thetac = [
#                 np.random.uniform(lo, hi)
#                 for lo, hi in zip(bound_min[solver], bound_max[solver])
#             ]
#             tasks.append((
#                 solver,
#                 thetac,
#                 bound_min[solver],
#                 bound_max[solver],
#                 mutation[solver],
#                 objf, method, data, system, models, logging
#             ))
#
#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         futures = [executor.submit(_run_task_wrapper, t) for t in tasks]
#         for future in as_completed(futures):
#             solver, result = future.result()
#             try:
#                 res = result.get(solver, None)
#                 if res is not None:
#                     all_results[solver].append(res)
#             except Exception as e:
#                 print(f"[{solver}] Error retrieving result for one of the iterations: {e}")
#
#
#
#
#     best_results = {}
#     for solver, res_list in all_results.items():
#         valid = [r for r in res_list if r is not None]
#         successes = [r for r in valid if r.success]
#         if successes:
#             best_results[solver] = min(successes, key=attrgetter("fun"))
#         elif valid:
#             best_results[solver] = min(valid, key=attrgetter("fun"))
#         else:
#             best_results[solver] = None
#
#     return best_results
#
#
#
# # def _objective(
# #     theta,
# #     data,
# #     active_models,
# #     thetac,
# #     system,
# #     models,
# #     **kwargs
# # ):
# #     """
# #     Objective function for optimization in parameter estimation.
# #
# #     Returns:
# #         tuple: (data, metrics)
# #     """
# #
# #
# #     theta = theta.tolist()
# #     # print(f'thetac= {thetac}')
# #
# #     global_y_true = {}
# #     global_y_pred = {}
# #     global_y_err = {}
# #     var_measurements = {}
# #
# #     tv_iphi_vars = list(system['tvi'].keys())
# #     ti_iphi_vars = list(system['tii'].keys())
# #     tv_ophi_vars = [v for v, cfg in system['tvo'].items() if cfg.get('meas', True)]
# #     ti_ophi_vars = [v for v, cfg in system['tio'].items() if cfg.get('meas', True)]
# #
# #     std_dev = {}
# #     for var, cfg in {**system['tvo'], **system['tio']}.items():
# #         if cfg.get('meas', True):
# #             sigma = cfg.get('unc', 1.0)
# #             std_dev[var] = 1.0 if sigma is None or np.isnan(sigma) else sigma
# #
# #     phisc = {v: 1 for v in ti_iphi_vars}
# #     phitsc = {v: 1 for v in tv_iphi_vars}
# #     tsc = 1
# #
# #     for sheet_name, sheet_data in data.items():
# #         t_values = {
# #             var: np.array(sheet_data[f"MES_X:{var}"])[~np.isnan(sheet_data[f"MES_X:{var}"])]
# #             for var in tv_ophi_vars if f"MES_X:{var}" in sheet_data
# #         }
# #
# #         swps_data = {}
# #         for var in tv_iphi_vars:
# #             tkey, lkey = f"{var}t", f"{var}l"
# #             if tkey in sheet_data and lkey in sheet_data:
# #                 tarr = np.array(sheet_data[tkey])[~np.isnan(sheet_data[tkey])]
# #                 larr = np.array(sheet_data[lkey])[~np.isnan(sheet_data[lkey])]
# #                 if tarr.size and larr.size:
# #                     swps_data[tkey] = tarr
# #                     swps_data[lkey] = larr
# #
# #         ti_iphi_data = {v: np.array(sheet_data.get(v, [np.nan]))[0] for v in ti_iphi_vars}
# #         tv_iphi_data = {v: np.array(sheet_data.get(v, []))[~np.isnan(sheet_data.get(v, []))] for v in tv_iphi_vars}
# #         tv_ophi_data = {v: np.array(sheet_data.get(f"MES_Y:{v}", []))[~np.isnan(sheet_data.get(f"MES_Y:{v}", []))]
# #                         for v in tv_ophi_vars if f"MES_Y:{v}" in sheet_data}
# #         ti_ophi_data = {v: np.array(sheet_data.get(f"MES_Y:{v}", [np.nan]))[0]
# #                         for v in ti_ophi_vars if f"MES_Y:{v}" in sheet_data}
# #
# #         cvp = {v: sheet_data[f"CVP:{v}"].iloc[0] for v in system['tvi']}
# #         solver_name = active_models[0]
# #
# #         if solver_name:
# #             t_all = np.unique(np.array(sheet_data["X:all"])[~np.isnan(sheet_data["X:all"])] )
# #             tv_sim, ti_sim, _ = simula(
# #                 t_all, swps_data, ti_iphi_data,
# #                 phisc, phitsc, tsc, theta, thetac,
# #                 cvp, tv_iphi_data, solver_name, system, models
# #             )
# #             # --- GUARD: make sure every tv_sim[var] is full‐length ---
# #             expected_len = t_all.shape[0]
# #             for var, arr in tv_sim.items():
# #                 arr = np.asarray(arr)
# #                 if arr.ndim != 1 or arr.shape[0] != expected_len:
# #                     # bail out early with a huge penalty:
# #                     return data, {'LS': 1e12, 'WLS': 1e12, 'MLE': 1e12, 'Chi': 1e12, 'R2_responses': {}}
# #
# #
# #             for var in tv_ophi_vars:
# #                 times = t_values.get(var)
# #                 if times is None:
# #                     continue
# #                 idx = np.isin(t_all, times)
# #                 y_true = tv_ophi_data.get(var, np.array([]))
# #                 y_pred = np.array(tv_sim[var])[idx]
# #                 if f"MES_E:{var}" in sheet_data:
# #                     y_err = np.array(sheet_data[f"MES_E:{var}"])[~np.isnan(sheet_data[f"MES_E:{var}"])]
# #                 else:
# #                     y_err = np.full_like(y_true, std_dev.get(var, 1.0))
# #
# #                 global_y_true.setdefault(var, []).extend(y_true)
# #                 global_y_pred.setdefault(var, []).extend(y_pred)
# #                 global_y_err.setdefault(var, []).extend(y_err)
# #                 for yt, yp, ye in zip(y_true, y_pred, y_err):
# #                     var_measurements.setdefault(var, []).append((yt, yp, ye))
# #
# #             for var in ti_ophi_vars:
# #                 y_true = ti_iphi_data.get(var, np.nan)
# #                 y_pred = ti_sim.get(var, np.nan)
# #                 if not np.isnan(y_true) and not np.isnan(y_pred):
# #                     sigma = std_dev.get(var, 1.0)
# #                     global_y_true.setdefault(var, []).append(y_true)
# #                     global_y_pred.setdefault(var, []).append(y_pred)
# #                     global_y_err.setdefault(var, []).append(sigma)
# #                     var_measurements.setdefault(var, []).append((y_true, y_pred, sigma))
# #
# #     all_y_true, all_y_pred, all_y_err, all_vars = [], [], [], []
# #     for var in global_y_true:
# #         y_t = np.array(global_y_true[var])
# #         y_p = np.array(global_y_pred[var])
# #         y_e = np.array(global_y_err[var])
# #         n = min(len(y_t), len(y_p), len(y_e))
# #         all_y_true.extend(y_t[:n])
# #         all_y_pred.extend(y_p[:n])
# #         all_y_err.extend(y_e[:n])
# #         all_vars.extend([var] * n)
# #
# #     all_y_true = np.array(all_y_true)
# #     all_y_pred = np.array(all_y_pred)
# #     all_y_err = np.array(all_y_err)
# #     all_vars = np.array(all_vars)
# #     N_total = len(all_y_true)
# #
# #     eps = 1e-8
# #     per_var_metrics = {'LS': {}, 'WLS': {}, 'MLE': {}, 'Chi': {}}
# #     for var in np.unique(all_vars):
# #         idx = np.where(all_vars == var)[0]
# #         y_t = all_y_true[idx]
# #         y_p = all_y_pred[idx]
# #         y_e = all_y_err[idx]
# #
# #         per_var_metrics['LS'][var] = np.sum((y_t - y_p) ** 2) / len(idx)
# #         sigma = std_dev.get(var, 1.0)
# #         per_var_metrics['WLS'][var] = np.sum(((y_t - y_p) / sigma) ** 2) / len(idx)
# #
# #         rel_err = (y_t - y_p) / np.maximum(np.abs(y_t), eps)
# #         rel_unc = y_e / np.maximum(np.abs(y_t), eps)
# #         rel_unc = np.clip(rel_unc, 1e-3, 1e2)
# #         mle_terms = 0.5 * (np.log(2 * np.pi * rel_unc**2) + (rel_err ** 2) / (rel_unc ** 2))
# #         per_var_metrics['MLE'][var] = np.mean(mle_terms)
# #
# #         chi_squared = ((y_t - y_p) / np.maximum(np.abs(y_t), eps)) ** 2
# #         per_var_metrics['Chi'][var] = np.sum(chi_squared) / len(idx)
# #
# #     LS = sum(per_var_metrics['LS'].values()) * N_total
# #     WLS = sum(per_var_metrics['WLS'].values()) * N_total
# #     MLE = sum(per_var_metrics['MLE'].values()) * N_total
# #     Chi = sum(per_var_metrics['Chi'].values()) * N_total
# #
# #     r2_responses = {}
# #     for var, meas in var_measurements.items():
# #         y_t = np.array([m[0] for m in meas])
# #         y_p = np.array([m[1] for m in meas])
# #         if len(y_t) > 1:
# #             ssr = np.sum((y_t - y_p) ** 2)
# #             sst = np.sum((y_t - np.mean(y_t)) ** 2)
# #             r2_responses[var] = 1 - ssr / sst if sst else 0.0
# #         else:
# #             r2_responses[var] = np.nan
# #
# #     metrics = {
# #         'LS': LS,
# #         'MLE': MLE,
# #         'Chi': Chi,
# #         'WLS': WLS,
# #         'R2_responses': r2_responses
# #     }
# #     return data, metrics
#
# def _objective(
#     theta,
#     data,
#     active_models,
#     thetac,
#     system,
#     models,
#     **kwargs
# ):
#     """
#     Objective function for optimization or a single pairs‐bootstrap draw.
#
#     If kwargs.get('bootstrap') is True, draws one sample of indices with replacement
#     and reindexes y_true, y_pred, y_err, and vars_ accordingly.
#     Otherwise uses the full original data.
#
#     Returns:
#         tuple: (data, metrics)
#     """
#     import numpy as np
#     from middoe.krnl_simula import simula
#
#     theta = theta.tolist()
#     bootstrap = kwargs.get('bootstrap', False)
#
#     # 1) Run simula & collect per‑point true/pred/err
#     global_y_true = {}
#     global_y_pred = {}
#     global_y_err  = {}
#     var_measurements = {}
#
#     tv_iphi = list(system['tvi'].keys())
#     ti_iphi = list(system['tii'].keys())
#     tv_ophi = [v for v,c in system['tvo'].items() if c.get('meas',True)]
#     ti_ophi = [v for v,c in system['tio'].items() if c.get('meas',True)]
#
#     std_dev = {}
#     for v, cfg in {**system['tvo'], **system['tio']}.items():
#         if cfg.get('meas',True):
#             s = cfg.get('unc',1.0)
#             std_dev[v] = 1.0 if s is None or np.isnan(s) else s
#
#     phisc  = {v:1 for v in ti_iphi}
#     phitsc = {v:1 for v in tv_iphi}
#     tsc    = 1
#
#     solver = active_models[0]
#     for sheet_name, sheet in data.items():
#         # time grid
#         t_all = np.unique(sheet["X:all"].dropna().values)
#
#         # switch‑pressure
#         swps = {}
#         for v in tv_iphi:
#             tkey,lkey = f"{v}t", f"{v}l"
#             if tkey in sheet and lkey in sheet:
#                 ta = sheet[tkey].dropna().values
#                 la = sheet[lkey].dropna().values
#                 if ta.size and la.size:
#                     swps[tkey], swps[lkey] = ta, la
#
#         # inputs
#         ti_in = {v: sheet.get(v, pd.Series([np.nan])).iloc[0] for v in ti_iphi}
#         tv_in = {v: sheet.get(v, pd.Series()).dropna().values for v in tv_iphi}
#         cvp   = {v: sheet[f"CVP:{v}"].iloc[0] for v in system['tvi']}
#
#         # simulate
#         tv_sim, ti_sim, _ = simula(
#             t_all, swps, ti_in,
#             phisc, phitsc, tsc, theta, thetac,
#             cvp, tv_in, solver, system, models
#         )
#
#         # collect errors
#         for v in tv_ophi:
#             xcol,ycol = f"MES_X:{v}", f"MES_Y:{v}"
#             if xcol not in sheet or ycol not in sheet: continue
#             mask = ~sheet[xcol].isna().values
#             times = sheet[xcol][mask].values
#             y_t = sheet[ycol][mask].values
#             idx = np.isin(t_all, times)
#             y_p = np.array(tv_sim[v])[idx]
#             ecol = f"MES_E:{v}"
#             y_e = sheet[ecol][mask].values if ecol in sheet else np.full_like(y_t, std_dev[v])
#             global_y_true.setdefault(v,[]).extend(y_t.tolist())
#             global_y_pred.setdefault(v,[]).extend(y_p.tolist())
#             global_y_err.setdefault(v,[]).extend(y_e.tolist())
#             for yt,yp,ye in zip(y_t,y_p,y_e):
#                 var_measurements.setdefault(v,[]).append((yt,yp,ye))
#
#         for v in ti_ophi:
#             ycol = f"MES_Y:{v}"
#             if ycol not in sheet: continue
#             y_t = sheet[ycol].iloc[0]
#             y_p = ti_sim.get(v, np.nan)
#             if np.isnan(y_t) or np.isnan(y_p): continue
#             sigma = std_dev[v]
#             global_y_true.setdefault(v,[]).append(y_t)
#             global_y_pred.setdefault(v,[]).append(y_p)
#             global_y_err.setdefault(v,[]).append(sigma)
#             var_measurements.setdefault(v,[]).append((y_t,y_p,sigma))
#
#     # 2) Flatten
#     all_y_true, all_y_pred, all_y_err, all_vars = [], [], [], []
#     for v, yt_list in global_y_true.items():
#         yp_list = global_y_pred[v]
#         ye_list = global_y_err[v]
#         n = min(len(yt_list), len(yp_list), len(ye_list))
#         all_y_true.extend(yt_list[:n])
#         all_y_pred.extend(yp_list[:n])
#         all_y_err.extend(ye_list[:n])
#         all_vars.extend([v]*n)
#
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
#     all_y_err  = np.array(all_y_err)
#     all_vars    = np.array(all_vars)
#     N = len(all_y_true)
#
#     # 3) Pairs bootstrap or original
#     if bootstrap:
#         idx = np.random.choice(N, size=N, replace=True)
#         y_t   = all_y_true[idx]
#         y_p   = all_y_pred[idx]
#         y_e   = all_y_err[idx]
#         vars_ = all_vars[idx]
#     else:
#         y_t, y_p, y_e, vars_ = all_y_true, all_y_pred, all_y_err, all_vars
#
#     # 4) Compute metrics
#     eps = 1e-8
#     per_var = {'LS':{}, 'WLS':{}, 'MLE':{}, 'Chi':{}}
#     for v in np.unique(vars_):
#         sel = np.where(vars_==v)[0]
#         yt, yp, ye = y_t[sel], y_p[sel], y_e[sel]
#         n = len(sel)
#         per_var['LS'][v]  = np.sum((yt-yp)**2)/n
#         sigma = std_dev[v]
#         per_var['WLS'][v] = np.sum(((yt-yp)/sigma)**2)/n
#         rel_err = (yt-yp)/np.maximum(np.abs(yt),eps)
#         rel_unc = np.clip(ye/np.maximum(np.abs(yt),eps),1e-3,1e2)
#         mle    = 0.5*(np.log(2*np.pi*rel_unc**2)+(rel_err**2)/(rel_unc**2))
#         per_var['MLE'][v] = np.mean(mle)
#         per_var['Chi'][v] = np.sum(rel_err**2)/n
#
#     LS  = sum(per_var['LS'].values())  * N
#     WLS = sum(per_var['WLS'].values()) * N
#     MLE = sum(per_var['MLE'].values()) * N
#     Chi = sum(per_var['Chi'].values()) * N
#
#     # 5) R²
#     r2 = {}
#     for v, meas in var_measurements.items():
#         yt = np.array([m[0] for m in meas])
#         yp = np.array([m[1] for m in meas])
#         if len(yt)>1:
#             ssr = np.sum((yt-yp)**2)
#             sst = np.sum((yt-yt.mean())**2)
#             r2[v] = 1 - ssr/sst if sst else np.nan
#         else:
#             r2[v] = np.nan
#
#     metrics = {'LS':LS, 'WLS':WLS, 'MLE':MLE, 'Chi':Chi, 'R2_responses':r2}
#     return data, metrics
#
#
# def _objective_function(theta, data, model, thetac, objf, system, models, logging):
#     """
#     Wrapper function for the objective function to be used in optimization.
#
#     Parameters:
#     theta (list): List of parameters to optimize.
#     data (dict): Dictionary containing the experimental data for optimization.
#     model (str): Name of the model to be used.
#     thetac (list): List of parameter scalors.
#     nd (int): Number of data points, for time spanning.
#     objf (str): Name of the objective function to minimize.
#     design_variables (dict): Dictionary containing design variables, part of the model structure.
#     run_solver (function): Model simulator function.
#
#     Returns:
#     float: Value of the objective function.
#     """
#     start_time = time.time()  # Start timer
#     # Call the objective function
#     optimized_data, metrics = _objective(
#         theta, data, [model], thetac, system, models
#     )
#     end_time = time.time()  # End timer
#     elapsed_time = end_time - start_time
#
#     if logging:
#         print(f"Objective function: '{objf}'| model '{model}' | CPU time {elapsed_time:.4f} seconds.")
#
#
#     # Extract the metrics needed for the optimization
#     LS, MLE, Chi, WLS = metrics['LS'], metrics['MLE'], metrics['Chi'], metrics['WLS']
#
#     # Determine which objective function to minimize
#     if objf == 'LS':
#         return -LS
#     elif objf == 'MLE':
#         return -MLE
#     elif objf == 'Chi':
#         return -Chi
#     elif objf == 'WLS':
#         return WLS
#     else:
#         raise ValueError(f"Unknown objective function: {objf}")
#
#
# def _runner(active_models, theta_parameters, bound_max, bound_min,
#             mutation, objf, x0_dict, method, data, system, models, logging):
#     """
#     Runner function to perform optimization using different solvers.
#     """
#     results = {}
#     x0 = {model: x0_dict[model] for model in active_models}
#
#     for solver in active_models:
#         thetac = theta_parameters[solver]
#         thetamax = [hi / sc if sc != 0 else hi for hi, sc in zip(bound_max[solver], thetac)]
#         thetamin = [lo / sc if sc != 0 else lo for lo, sc in zip(bound_min[solver], thetac)]
#         thetas = mutation[solver]
#         initial_x0 = x0[solver]
#         narrow_factor = 1e-40
#
#         bounds = [
#             (x * (1 - narrow_factor), x * (1 + narrow_factor)) if not theta
#             else (tmin, tmax)
#             for x, theta, tmax, tmin in zip(initial_x0, thetas, thetamax, thetamin)
#         ]
#
#         try:
#             args = (data, solver, thetac, objf, system, models, logging)
#
#             if method == 'SLSQP':
#                 result = minimize(
#                     _objective_function, initial_x0,
#                     args=args,
#                     method='SLSQP',
#                     bounds=bounds,
#                     options={'maxiter': int(1e7), 'disp': False},
#                     tol=1e-4
#                 )
#
#             elif method == 'SQP':
#                 result = minimize(
#                     _objective_function, initial_x0,
#                     args=args,
#                     method='trust-constr',
#                     bounds=bounds,
#                     options={
#                         'maxiter': 500,
#                         'xtol': 1e-2,
#                         'gtol': 1e-2,
#                         'verbose': 0
#                     }
#                 )
#
#             elif method == 'NM':
#                 result = minimize(
#                     _objective_function, initial_x0,
#                     args=args,
#                     method='Nelder-Mead',
#                     options={'maxiter': 100000, 'disp': False, 'fatol': 1e-6}
#                 )
#
#             elif method == 'BFGS':
#                 result = minimize(
#                     _objective_function, initial_x0,
#                     args=args,
#                     method='BFGS',
#                     options={'disp': False, 'maxiter': 1000, 'disp': True}
#                 )
#
#             elif method == 'DE':
#                 result = differential_evolution(
#                     _objective_function,
#                     bounds=bounds,
#                     args=args,
#                     maxiter=1000,
#                     popsize=18,
#                     tol=1e-8,
#                     strategy='best1bin',
#                     mutation=(0.5, 1.5),
#                     recombination=0.7,
#                     polish=False,
#                     updating='deferred',
#                     workers=-1,
#                     disp=False
#                 )
#         except Exception as e:
#             print(f"[{solver}] failed: {e}")
#             continue
#
#         # scale the optimized parameters back
#         result['scpr'] = result['x'] * np.array(thetac)
#
#
#         if method in ['SQP', 'SLSQP', 'DE', 'NM']:
#             print(f'---------')
#             def loss(theta):
#                  return _objective_function(theta, data, solver, thetac, objf, system, models, logging)
#             try:
#                 hessian = Hessian(loss, step=1e-4, method='central', order=2)(result.x)
#                 thetac = np.array(thetac, dtype=np.float64)
#                 S = np.diag(1.0 / thetac)
#                 hessian_scaled = S @ hessian @ S
#                 result['hessian'] = hessian_scaled
#             except Exception as e:
#                 print(f"[{solver}] Hessian computation failed: {e}")
#                 result['hessian'] = None
#                 hessian_scaled = np.eye(len(thetac)) * 1e6  # fallback
#
#         # --- Invert Hessian or use optimizer-provided inverse ---
#         D = np.diag(thetac)
#         try:
#             if hasattr(result, 'hess_inv'):
#                 hessian_inv = D @ result.hess_inv @ D
#             else:
#                 hessian_inv = np.linalg.pinv(hessian_scaled)
#             result['hess_inv'] = hessian_inv
#         except Exception as e:
#             print(f"[{solver}] Hessian inversion failed: {e}")
#             result['hess_inv'] = np.eye(len(thetac)) * 1e6  # fallback
#
#         # print(f'variance matrix : {hessian_inv}')
#         results[solver] = result
#
#     return results
#
#
#
#
# def _report_optimization_results(results, logging):
#     """
#     Report the optimization results.
#
#     Parameters:
#     results (dict): Dictionary containing the results of the optimization.
#     logging (bool): Flag to indicate whether to log the results.
#     """
#     if logging:
#         for solver, solver_results in results.items():
#             print(f"parameter estimation for model {solver} concluded- success: {solver_results['optimization_result'].success}")
#             print()
#
#

import concurrent, pandas as pd, multiprocessing, time, numpy as np, warnings
from numdifftools import Hessian
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed
from middoe.krnl_simula import simula
from middoe.iden_utils import _initialize_dictionaries
from operator import attrgetter

warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

def _initialise_theta_parameters(theta_min, theta_max, active_models):
    theta_parameters = {}
    for sv in active_models:
        theta_parameters[sv] = [np.random.uniform(lo,hi) for lo,hi in zip(theta_min[sv],theta_max[sv])]
    return theta_parameters

def parmest(system, models, iden_opt, data, case=None):
    _initialize_dictionaries(models,iden_opt)
    active_models, bound_max, bound_min, mutation = models['can_m'], models['t_u'], models['t_l'], models['mutation']
    method, objf, multi = iden_opt['meth'], iden_opt['ob'], iden_opt.get('ms',False)
    bootstrap, nboot, logging = (iden_opt.get('var-cov')=='B'), iden_opt.get('nboot',100), iden_opt.get('log',False)
    theta_params_r = _initialise_theta_parameters(bound_min,bound_max,active_models)
    theta_params_f = models['theta']
    x0_dict = {sv:[1.0]*len(theta_params_r[sv]) for sv in active_models}
    if bootstrap: return _bootstrap_runner(active_models,theta_params_f,bound_max,bound_min,mutation,objf,x0_dict,method,data,system,models,logging,nboot, multi)
    if multi:
        nstarts = max(1,int(0.7*multiprocessing.cpu_count()))
        return _multi_start_runner(active_models,theta_params_r,bound_max,bound_min,mutation,objf,method,data,system,models,logging,nstarts)
    return _runner(active_models,theta_params_f,bound_max,bound_min,mutation,objf,x0_dict,method,data,system,models,logging)

def _run_single_start(sv,thetac,thetamin,thetamax,thetas,objf,method,data,system,models,logging):
    if logging: print(f"[{sv}] Sampled x0: {[1.0]*len(thetac)}")
    res=_runner([sv],{sv:thetac}, {sv:thetamax},{sv:thetamin},{sv:thetas},objf,{sv:[1.0]*len(thetac)},method,data,system,models,logging)
    print(f"[{sv}] Finished with result: {res}")
    return (sv,res) if sv in res else (sv,None)

def _multi_start_runner(active_models,_unused, bmax,bmin,mutation,objf,method,data,system,models,logging,nstarts=None):
    del _unused
    nwork=max(1,int(0.7*multiprocessing.cpu_count())); nstarts=nstarts or max(2*nwork,10)
    allr={sv:[] for sv in active_models}; tasks=[]
    for sv in active_models:
        for _ in range(nstarts):
            tasks.append((sv,[np.random.uniform(lo,hi) for lo,hi in zip(bmin[sv],bmax[sv])],bmin[sv],bmax[sv],mutation[sv],objf,method,data,system,models,logging))
    with ProcessPoolExecutor(max_workers=nwork) as exe:
        for fut in as_completed([exe.submit(_run_single_start,*t) for t in tasks]):
            sv,res=fut.result();
            if res and res.get(sv): allr[sv].append(res[sv])
    return {sv:(min([r for r in allr[sv] if r.success],key=attrgetter("fun")) if any(r.success for r in allr[sv]) else None) for sv in active_models}


# module‐level helper
def _bootstrap_worker(active_models, theta_params, bmax, bmin, mutation, objf, x0, method, data, system, models, logging):
    models_boot = {**models, '__bootstrap__': True}
    nstarts = max(1, int(0.7 * multiprocessing.cpu_count()))
    theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
    ref = _multi_start_runner(active_models, theta_params_r, bmax, bmin, mutation, objf, method, data, system, models_boot,
                              logging, nstarts)
    # ref = _runner(active_models, theta_params, bmax, bmin, mutation, objf, x0, method, data, system, models_boot, logging)
    return ref

def _bootstrap_runner(active_models, theta_params, bmax, bmin, mutation, objf, x0, method, data, system, models, logging, nboot, multi):
    # reference fit (full data)
    if multi:
        nstarts = max(1,int(0.7*multiprocessing.cpu_count()))
        theta_params_r = _initialise_theta_parameters(bmin, bmax, active_models)
        ref =_multi_start_runner(active_models,theta_params_r,bmax,bmin,mutation,objf,method,data,system,models,logging,nstarts)
    else:
        _runner(active_models, theta_params, bmax, bmin, mutation, objf, x0, method, data, system, models, logging)
    allP = {sv: [] for sv in active_models}; bootR = {sv: [] for sv in active_models}
    args = (active_models, theta_params, bmax, bmin, mutation, objf, x0, method, data, system, models, logging)
    with ProcessPoolExecutor(max_workers=min(nboot, multiprocessing.cpu_count())) as exe:
        futures = [exe.submit(_bootstrap_worker, *args) for _ in range(nboot)]
        for fut in as_completed(futures):
            res = fut.result()
            for sv in active_models:
                r = res.get(sv)
                if r and getattr(r, 'success', True):
                    bootR[sv].append(r)
                    allP[sv].append(r.scpr)
    for sv in active_models:
        ref[sv].samples = bootR[sv]
        ref[sv].v = np.cov(np.stack(allP[sv]), rowvar=False, ddof=1) if allP[sv] else None

    return ref


def _runner(active_models, theta_params, bmax, bmin, mutation, objf, x0_dict, method, data, system, models, logging):
    results = {}
    for sv in active_models:
        thetac, x0 = np.array(theta_params[sv], float), np.array(x0_dict[sv], float)
        thetamax, thetamin = np.array(bmax[sv]) / thetac, np.array(bmin[sv]) / thetac
        mask, narrow = np.array(mutation[sv], bool), 1e-40
        bounds = [((x*(1-narrow), x*(1+narrow)) if not m else (tmin, tmax))
                  for x, m, tmin, tmax in zip(x0, mask, thetamin, thetamax)]
        bootstrap_flag = models.get('__bootstrap__', False)
        args = (data, [sv], thetac, system, models, logging, objf, bootstrap_flag)
        try:
            if method == 'SLSQP':
                res = minimize(_objective_function, x0, args=args, method='SLSQP',
                               bounds=bounds, options={'maxiter':100000, 'ftol':1e-1, 'disp':True})
            elif method == 'SQP':
                res = minimize(_objective_function, x0, args=args, method='trust-constr',
                               bounds=bounds, options={'maxiter':500, 'xtol':1e-2, 'gtol':1e-2})
            elif method == 'NM':
                res = minimize(_objective_function, x0, args=args, method='Nelder-Mead',
                               options={'maxiter':1e5, 'fatol':1e-6, 'disp':False})
            elif method == 'BFGS':
                res = minimize(_objective_function, x0, args=args, method='BFGS',
                               options={'maxiter':1e3, 'disp':False})
            elif method == 'DE':
                res = differential_evolution(_objective_function, bounds, args=args,
                                             maxiter=1e3, popsize=18, tol=1e-8,
                                             strategy='best1bin', mutation=(0.5,1.5),
                                             recombination=0.7, polish=False,
                                             updating='deferred', workers=-1)
            else:
                raise ValueError(f"Unknown method '{method}'")
        except Exception as e:
            if logging: print(f"[{sv}] optimize error: {e}")
            continue

        res.scpr = res.x * thetac

        # compute Hessian & inverse for the reference (non-bootstrap) fit
        if not bootstrap_flag and method in ['SLSQP','SQP','NM','DE']:
            try:
                loss = lambda t: _objective_function(t, data, [sv], thetac, system, models, logging, objf, False)
                H = Hessian(loss, step=1e-4, method='central', order=2)(res.x)
                S = np.diag(1.0 / thetac.astype(float))
                Hs = S @ H @ S
                res.hessian = Hs
            except Exception as e:
                if logging: print(f"[{sv}] Hessian computation failed: {e}")
                Hs = np.eye(len(thetac)) * 1e6
                res.hessian = None
            D = np.diag(thetac.astype(float))
            try:
                inv = D @ (res.hess_inv if hasattr(res, 'hess_inv') else np.linalg.pinv(Hs)) @ D
                res.hess_inv = inv
            except Exception as e:
                if logging: print(f"[{sv}] Hessian inversion failed: {e}")
                res.hess_inv = np.eye(len(thetac)) * 1e6

        results[sv] = res
    return results

def _objective_function(theta,data,active, thetac,system,models,logging,objf,bootstrap):
    start=time.time();_,m=_objective(theta,data,active,thetac,system,models,bootstrap=bootstrap)
    if logging: print(f"Obj '{objf}'|{active[0]}|{time.time()-start:.3f}s")
    return {'LS':m['LS'],'WLS':m['WLS'],'MLE':m['MLE'],'Chi':m['Chi']}[objf]

def _objective(theta,data,active,thetac,system,models,bootstrap=False):
    import pandas as _pd
    theta=theta.tolist()
    tv_i,ti_i=list(system['tvi']),list(system['tii'])
    tv_o=[v for v,c in system['tvo'].items() if c.get('meas',True)]
    ti_o=[v for v,c in system['tio'].items() if c.get('meas',True)]
    std_dev={v:(cfg.get('unc',1.0) if not np.isnan(cfg.get('unc',1.0)) else 1.0) for v,cfg in {**system['tvo'],**system['tio']}.items() if cfg.get('meas',True)}
    phisc,phitsc,tsc={v:1 for v in ti_i},{v:1 for v in tv_i},1
    solver=active[0]
    global_y_true,global_y_pred,global_y_err,varm={}, {}, {}, {}

    for sh in data.values():
        t_all=np.unique(sh["X:all"].dropna().values)
        swps={}
        for v in tv_i:
            tk,lk=f"{v}t",f"{v}l"
            if tk in sh and lk in sh:
                ta,la=sh[tk].dropna().values,sh[lk].dropna().values
                if ta.size and la.size: swps[tk],swps[lk]=ta,la
        ti_in={v:sh.get(v,_pd.Series([np.nan])).iloc[0] for v in ti_i}
        tv_in={v:sh.get(v,_pd.Series()).dropna().values for v in tv_i}
        cvp={v:sh[f"CVP:{v}"].iloc[0] for v in system['tvi']}
        tvs,tis,_=simula(t_all,swps,ti_in,phisc,phitsc,tsc,theta,thetac,cvp,tv_in,solver,system,models)

        for v in tv_o:
            xc,yc=f"MES_X:{v}",f"MES_Y:{v}"
            if xc not in sh or yc not in sh: continue
            mask=~sh[xc].isna().values; times,y_t=sh[xc][mask].values,sh[yc][mask].values
            idx=np.isin(t_all,times); y_p=np.array(tvs[v])[idx]
            ye_col=f"MES_E:{v}"
            y_e=sh[ye_col][mask].values if ye_col in sh else np.full_like(y_t,std_dev[v])
            global_y_true.setdefault(v,[]).extend(y_t.tolist())
            global_y_pred.setdefault(v,[]).extend(y_p.tolist())
            global_y_err.setdefault(v,[]).extend(y_e.tolist())
            for yt,yp,ye in zip(y_t,y_p,y_e): varm.setdefault(v,[]).append((yt,yp,ye))

        for v in ti_o:
            yc=f"MES_Y:{v}"
            if yc not in sh: continue
            y_t,y_p=sh[yc].iloc[0],tis.get(v,np.nan)
            if np.isnan(y_t) or np.isnan(y_p): continue
            sigma=std_dev[v]
            global_y_true.setdefault(v,[]).append(y_t)
            global_y_pred.setdefault(v,[]).append(y_p)
            global_y_err.setdefault(v,[]).append(sigma)
            varm.setdefault(v,[]).append((y_t,y_p,sigma))

    # flatten
    all_y_true,all_y_pred,all_y_err,all_vars=[],[],[],[]
    for v,ytl in global_y_true.items():
        ypl,yel=global_y_pred[v],global_y_err[v]; n=min(len(ytl),len(ypl),len(yel))
        all_y_true+=ytl[:n]; all_y_pred+=ypl[:n]; all_y_err+=yel[:n]; all_vars+=[v]*n
    all_y_true,all_y_pred,all_y_err,all_vars=np.array(all_y_true),np.array(all_y_pred),np.array(all_y_err),np.array(all_vars)
    N=len(all_y_true)

    if bootstrap:
        idx=np.random.choice(N,size=N,replace=True)
        y_t,y_p,y_e,vars_=all_y_true[idx],all_y_pred[idx],all_y_err[idx],all_vars[idx]
    else:
        y_t,y_p,y_e,vars_=all_y_true,all_y_pred,all_y_err,all_vars

    eps=1e-8; per_var={'LS':{},'WLS':{},'MLE':{},'Chi':{}}
    for v in np.unique(vars_):
        sel=np.where(vars_==v)[0]; yt,yp,ye=y_t[sel],y_p[sel],y_e[sel]; n=len(sel)
        per_var['LS'][v]=np.sum((yt-yp)**2)/n
        per_var['WLS'][v]=np.sum(((yt-yp)/std_dev[v])**2)/n
        rel_err=(yt-yp)/np.maximum(np.abs(yt),eps)
        rel_unc=np.clip(ye/np.maximum(np.abs(yt),eps),1e-3,1e2)
        per_var['MLE'][v]=np.mean(0.5*(np.log(2*np.pi*rel_unc**2)+(rel_err**2)/(rel_unc**2)))
        per_var['Chi'][v]=np.sum(rel_err**2)/n

    LS=sum(per_var['LS'].values())*N;WLS=sum(per_var['WLS'].values())*N;MLE=sum(per_var['MLE'].values())*N;Chi=sum(per_var['Chi'].values())*N
    if bootstrap:
        print(f'WLS={WLS:.5f}')
    r2={}
    for v,me in varm.items():
        yta,ypa=np.array([m[0] for m in me]),np.array([m[1] for m in me])
        if len(yta)>1: r2[v]=1-np.sum((yta-ypa)**2)/np.sum((yta-yta.mean())**2)
        else: r2[v]=np.nan

    return data,{'LS':LS,'WLS':WLS,'MLE':MLE,'Chi':Chi,'R2_responses':r2}
