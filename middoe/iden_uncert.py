# # iden_uncert.py
# import matplotlib.pyplot as plt
# from pathlib import Path
# from middoe.krnl_simula import simula
# import logging
# import numpy as np
# import pandas as pd
# from scipy import stats
# from middoe.log_utils import  read_excel
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
# def uncert(resultpr, system, models, iden_opt, case=None):
#     """
#     Perform uncertainty analysis on the optimization results.
#
#     This function evaluates the uncertainty in the optimization results by analyzing
#     the variance-covariance structure of the estimated parameters. It supports multiple
#     methods for sensitivity analysis and variance-covariance computation.
#
#     Parameters
#     ----------
#     data : dict
#         Experimental data used for the analysis.
#     resultpr : dict
#         Dictionary containing the optimization results for each solver.
#     system : dict
#         System configuration, including variable definitions and constraints.
#     models : dict
#         Model definitions and settings, including mutation masks and parameter bounds.
#     iden_opt : dict
#         Identification options, including sensitivity method and variance-covariance type.
#         - 'sens_m': str, optional
#             Sensitivity method ('central' or 'forward'). Default is 'central'.
#         - 'varcov': str, optional
#             Variance-covariance type ('M', 'H', or 'B'). Default is 'H'.
#     case : str, optional
#         Specifies the analysis case. Default is None.
#
#     Returns
#     -------
#     dict
#         A dictionary containing the uncertainty analysis results, including:
#         - 'results': dict
#             Detailed results for each solver.
#         - 'obs': Any
#             Observed values from the analysis.
#
#     Notes
#     -----
#     If `iden_opt['varcov'] == 'B'`, the function uses the bootstrap variance-covariance
#     matrix stored in `resultpr[solver]['varcov']`.
#     """
#     # Sensitivity method and varcov selection
#     data = read_excel()
#     sens_method = iden_opt.get('sens_m', 'central')
#     varcov_key  = iden_opt.get('var-cov', 'H')  # 'M', 'H', or 'B'
#
#     # Identify measurement variables
#     tv_iphi_vars = list(system.get('tvi', {}))
#     ti_iphi_vars = list(system.get('tii', {}))
#     tv_ophi_vars = [v for v,c in system.get('tvo', {}).items() if c.get('meas', True)]
#     ti_ophi_vars = [v for v,c in system.get('tio', {}).items() if c.get('meas', True)]
#
#     # Collect measurement uncertainties (not directly used here)
#     std_dev = {}
#     for grp in ('tvo','tio'):
#         for v,cfg in system.get(grp, {}).items():
#             if cfg.get('meas', True):
#                 s = cfg.get('unc', 1.0)
#                 std_dev[v] = 1.0 if s is None or np.isnan(s) else s
#
#     # Prepare epsilon settings
#     eps_in = iden_opt.get('eps', None)
#     if not isinstance(eps_in, dict):
#         epsf = {sv: eps_in for sv in resultpr}
#     else:
#         epsf = eps_in.copy()
#
#     resultun = {}
#     observed_values = None
#
#     # Loop over each solver's optimization result
#     for solver, solver_results in resultpr.items():
#         thetac = solver_results['scpr']
#         thetas = models.get('mutation', {}).get(solver, [])
#         # If epsilon not provided, run mesh-independency test
#         if epsf.get(solver) is None:
#             epsf[solver] = _fdm_mesh_independency(
#                 theta=np.ones_like(thetac),
#                 thetac=thetac,
#                 solver=solver,
#                 system=system,
#                 models=models,
#                 tv_iphi_vars=tv_iphi_vars,
#                 ti_iphi_vars=ti_iphi_vars,
#                 tv_ophi_vars=tv_ophi_vars,
#                 ti_ophi_vars=ti_ophi_vars,
#                 data=data,
#                 sens_method=sens_method,
#                 varcov=varcov_key,
#                 resultpr=resultpr
#             )
#
#         # Compute uncertainty metrics
#         (
#             optimized_data,
#             LS, MLE, Chi,
#             LSA, Z, V_matrix,
#             CI, t_values,
#             t_m, tv_input_m, ti_input_m,
#             tv_output_m, ti_output_m,
#             WLS, obs, MSE,
#             R2_responses, R2_total, M
#         ) = _uncert_metrics(
#             theta=np.ones_like(thetac),
#             data=data,
#             active_solvers=[solver],
#             thetac=thetac,
#             eps=epsf,
#             thetas=thetas,
#             ti_iphi_vars=ti_iphi_vars,
#             tv_iphi_vars=tv_iphi_vars,
#             tv_ophi_vars=tv_ophi_vars,
#             ti_ophi_vars=ti_ophi_vars,
#             system=system,
#             models=models,
#             varcov=varcov_key,
#             resultpr=resultpr,
#             sens_method=sens_method
#         )
#         observed_values = obs
#
#         resultun[solver] = {
#             'optimization_result': solver_results,
#             'data': optimized_data,
#             'LS': LS,
#             'MLE': MLE,
#             'MSE': MSE,
#             'Chi': Chi,
#             'LSA': LSA,
#             'Z': Z,
#             'WLS': WLS,
#             'R2_responses': R2_responses,
#             'R2_total': R2_total,
#             'V_matrix': V_matrix,
#             'CI': CI,
#             't_values': t_values,
#             't_m': t_m,
#             'tv_input_m': tv_input_m,
#             'ti_input_m': ti_input_m,
#             'tv_output_m': tv_output_m,
#             'ti_output_m': ti_output_m,
#             'estimations': thetac,
#             'found_eps': epsf[solver],
#             'M': M
#         }
#
#     # Store back eps
#     iden_opt['eps'] = epsf
#
#     # Summarize and report
#     (
#         scaled_params,
#         solver_parameters,
#         solver_cov_matrices,
#         solver_confidence_intervals,
#         resultun2
#     ) = _report(resultun, models.get('mutation', {}), models, iden_opt.get('log', False))
#
#     # Optionally update models in‐place
#     if case is None:
#         for sv in models['can_m']:
#             if 'thetastart' not in models:
#                 models['thetastart'] = {}
#                 for sv in models['can_m']:
#                     if sv not in models['thetastart']:
#                         models['thetastart'][sv] = models['theta'][sv]
#
#             models['theta'][sv]    = scaled_params[sv]
#             models['V_matrix'][sv] = resultun[sv]['V_matrix']
#             # Initialize 'LSA' as a dict if it does not exist; otherwise keep existing
#             if 'LSA' not in models or not isinstance(models['LSA'], dict):
#                 models['LSA'] = {}
#
#             # Update (or add) the LSA values for each solver key
#             for sv in models['can_m']:
#                 models['theta'][sv] = scaled_params.get(sv)
#                 models['V_matrix'][sv] = resultun.get(sv, {}).get('V_matrix')
#
#                 lsa_value = resultun.get(sv, {}).get('LSA')
#                 if lsa_value is not None:
#                     models['LSA'][sv] = lsa_value
#
#     return {'results': resultun2, 'obs': observed_values}
#
# def _uncert_metrics(
#     theta, data, active_solvers, thetac, eps, thetas,
#     ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
#     system, models, varcov, resultpr, sens_method='forward'
# ):
#     """
#     Calculate detailed uncertainty metrics. Honors varcov=='B' to pull
#     bootstrap covariance from resultpr.
#
#     Assumes heteroskedastic (relative) measurement errors stored in MES_E:*.
#     Each MES_E value is interpreted as a *fractional* standard deviation and
#     is converted to an absolute sigma by multiplying with the signal magnitude
#     (|y_pred|). A tiny floor avoids zero weights.
#
#     Adds ill-conditioning diagnostics (rank/condition numbers/eigenvalues)
#     without changing the original signature or return tuple. The diagnostic
#     summary is logged (INFO to avoid colored 'red' outputs) and stored in
#     resultpr[solver]['ill_report'] when possible.
#     """
#     # ---- diagnostic thresholds (adjust here if needed) -----------------
#     _SLOPPY_THRESH = 1e3     # 'sloppy' if cond(M) >= this
#     _COND_THRESH   = 1e8     # ill-conditioned if cond(M) >= this
#     _TINY_EIG      = 1e-12   # ill if min eigenvalue < this
#     _REPORT_LEVEL  = logging.INFO  # keep it white in most color schemes
#     _SIG_FLOOR     = 1e-12   # floor for absolute sigmas
#
#     # print(f'theta: {theta}')
#     # print(f'thetac: {thetac}')
#
#     theta_full = np.array(theta, dtype=float)
#     active_idx = np.where(thetas)[0]
#     print(f'active_idx: {active_idx}')
#     n_active   = len(active_idx)
#     thetac_arr = np.array(thetac, dtype=float)
#
#     # Containers
#     var_measurements = {}                     # {var: [(y_true, y_pred, sigma_abs), ...]}
#     Q_accum = {v: [] for v in (tv_ophi_vars + ti_ophi_vars)}
#     t_m = {}
#     tv_input_m, ti_input_m = {}, {}
#     tv_output_m, ti_output_m = {}, {}
#
#     solver = active_solvers[0]
#     h = eps[solver]
#
#     # Loop over data sheets
#     for sheet_name, sheet in data.items():
#         # Time grid
#         t_all = np.unique(sheet.get("X:all", pd.Series()).dropna().values)
#
#         # Switch-pressure inputs
#         swps = {}
#         for v in tv_iphi_vars:
#             tkey, lkey = f"{v}t", f"{v}l"
#             if tkey in sheet and lkey in sheet:
#                 ta = sheet[tkey].dropna().values
#                 la = sheet[lkey].dropna().values
#                 if ta.size and la.size:
#                     swps[tkey], swps[lkey] = ta, la
#
#         # Input data
#         ti_in = {v: sheet.get(v, pd.Series([np.nan])).iloc[0] for v in ti_iphi_vars}
#         tv_in = {v: sheet.get(v, pd.Series()).dropna().values for v in tv_iphi_vars}
#         cvp   = {v: sheet[f"CVP:{v}"].iloc[0] for v in system.get('tvi', {})}
#
#         # Reference simulation
#         tv_ref, ti_ref, _ = simula(
#             t_all, swps, ti_in,
#             {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#             theta_full, thetac_arr, cvp, tv_in, solver, system, models
#         )
#
#         # Store simulation outputs
#         t_m[sheet_name]         = t_all.tolist()
#         tv_input_m[sheet_name]  = tv_in
#         ti_input_m[sheet_name]  = ti_in
#         tv_output_m[sheet_name] = tv_ref
#         ti_output_m[sheet_name] = ti_ref
#
#         # Build perturbed sims for FD sensitivities
#         perturbed = {}
#         for p in active_idx:
#             if sens_method == 'forward':
#                 thp = theta_full.copy(); thp[p] += h
#                 tvp, tip, _ = simula(
#                     t_all, swps, ti_in,
#                     {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                     thp, thetac_arr, cvp, tv_in, solver, system, models
#                 )
#                 perturbed[p] = ('forward', (tvp, tip))
#             else:
#                 thp = theta_full.copy(); thp[p] += h
#                 thm = theta_full.copy(); thm[p] -= h
#                 tvp, tip, _ = simula(
#                     t_all, swps, ti_in,
#                     {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                     thp, thetac_arr, cvp, tv_in, solver, system, models
#                 )
#                 tvm, tim, _ = simula(
#                     t_all, swps, ti_in,
#                     {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                     thm, thetac_arr, cvp, tv_in, solver, system, models
#                 )
#                 perturbed[p] = ('central', (tvp, tip, tvm, tim))
#
#         # Collect residuals and sensitivities
#         for var in tv_ophi_vars + ti_ophi_vars:
#             if var in tv_ophi_vars:
#                 # time-varying measurement
#                 xcol = f"MES_X:{var}"
#                 ycol = f"MES_Y:{var}"
#                 ecol = f"MES_E:{var}"
#                 mask = ~sheet[xcol].isna().values
#                 times = sheet[xcol][mask].values
#                 y_t   = sheet[ycol][mask].values
#                 idx   = np.isin(t_all, times)
#                 y_p   = np.array(tv_ref[var])[idx]
#
#                 # relative sigma from sheet, convert to absolute using |y_p|
#                 if ecol in sheet:
#                     rel = sheet[ecol][mask].values.astype(float)
#                     y_e = rel
#                 else:
#                     y_e = np.full_like(y_t, 1.0, dtype=float)
#             else:
#                 # time-independent measurement
#                 ycol = f"MES_Y:{var}"
#                 ecol = f"MES_E:{var}"
#                 y_t  = np.array([sheet[ycol].iloc[0]])
#                 y_p  = np.array([ti_ref[var]])
#                 if ecol in sheet:
#                     rel = float(sheet[ecol].iloc[0])
#                     # y_e = np.array([rel * np.abs(y_p[0]) + _SIG_FLOOR], dtype=float)
#                     y_e = np.array([rel], dtype=float)
#                 else:
#                     y_e = np.array([1.0], dtype=float)
#
#             var_measurements.setdefault(var, []).extend(zip(y_t, y_p, y_e))
#
#             # Sensitivity block
#             cols = []
#             for p in active_idx:
#                 mode, sims = perturbed[p]
#                 if mode == 'forward':
#                     tvp, tip = sims
#                     if var in tv_ophi_vars:
#                         y2 = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
#                     else:
#                         y2 = np.array([tip[var]])
#                     d = (y2 - y_p) / (h * thetac_arr[p])
#                 else:
#                     tvp, tip, tvm, tim = sims
#                     if var in tv_ophi_vars:
#                         y_p_p = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
#                         y_p_m = np.array([tvm[var][np.where(t_all==t)[0][0]] for t in times])
#                     else:
#                         y_p_p = np.array([tip[var]])
#                         y_p_m = np.array([tim[var]])
#                     d = (y_p_p - y_p_m) / (2*h*thetac_arr[p])
#                 cols.append(d)
#             if len(cols):
#                 Q_accum[var].append(np.stack(cols, axis=1))
#
#     # Flatten Q into LSA
#     LSA = np.vstack([np.vstack(Q_accum[v]) for v in Q_accum])
#     obs = sum(len(vals) for vals in var_measurements.values())
#     dof = max(obs - n_active, 1)
#     tcrit = stats.t.ppf(0.975, dof)
#
#     # --- Variance–Covariance Selection ---
#     if varcov == 'M':
#         # Using absolute sigmas we just constructed
#         sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]])
#                                for v in Q_accum])
#         # Avoid forming giant diagonal: scale rows instead
#         scaled_J = LSA / sigs[:, None]
#         M = scaled_J.T @ scaled_J
#         V = np.linalg.pinv(M)
#
#
#
#     elif varcov == 'H':
#         hess_inv = resultpr[solver].get('hess_inv')
#         err = resultpr[solver].get('fun', 1.0) / dof
#         V = err * np.array(hess_inv)
#         M = np.linalg.pinv(V)
#
#     elif varcov == 'B':
#         V = resultpr[solver].get('v')
#         if V is None:
#             raise ValueError(f"No bootstrap v matrix found for model '{solver}'")
#         M = np.linalg.pinv(V)
#         logger.info(f"Using bootstrap var–cov for {solver}")
#
#     else:
#         raise ValueError(f"Unknown varcov option '{varcov}'")
#
#     # --- Ill-conditioning diagnostics ----------------------------------
#     ill_report = {}
#     try:
#         s = np.linalg.svd(LSA, compute_uv=False)
#         ill_report['Q_singular_values'] = s
#         ill_report['Q_rank'] = int(np.sum(s > s[0] * np.finfo(float).eps))
#         ill_report['Q_cond'] = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
#     except Exception as e:
#         ill_report['Q_error'] = str(e)
#
#     try:
#         eig = np.linalg.eigvalsh(M)
#         ill_report['M_eigenvalues'] = eig
#         ill_report['M_min_eig'] = float(eig.min())
#         ill_report['M_max_eig'] = float(eig.max())
#         ill_report['M_cond'] = float(eig.max() / eig.min()) if eig.min() > 0 else np.inf
#     except Exception as e:
#         ill_report['M_error'] = str(e)
#
#     ill_report['is_rank_deficient'] = ill_report.get('Q_rank', n_active) < n_active
#     ill_report['is_sloppy'] = ill_report.get('M_cond', np.inf) >= _SLOPPY_THRESH
#     ill_report['is_ill_conditioned'] = (
#         ill_report['is_rank_deficient'] or
#         ill_report.get('M_cond', 0) >= _COND_THRESH or
#         ill_report.get('M_min_eig', _TINY_EIG) < _TINY_EIG
#     )
#     ill_report['summary'] = (
#         f"Rank(Q)={ill_report.get('Q_rank','?')}/{n_active}, "
#         f"cond(Q)≈{ill_report.get('Q_cond',np.nan):.2e}, "
#         f"cond(M)≈{ill_report.get('M_cond',np.nan):.2e}, "
#         f"λ_min(M)≈{ill_report.get('M_min_eig',np.nan):.2e}; "
#         f"{'ILL-CONDITIONED' if ill_report['is_ill_conditioned'] else 'OK'}"
#     )
#
#     logger.log(_REPORT_LEVEL, "Conditioning diagnostics: %s", ill_report['summary'])
#
#     # Stash report into resultpr without changing the API
#     try:
#         if isinstance(resultpr.get(solver, None), dict):
#             resultpr[solver]['ill_report'] = ill_report
#     except Exception:
#         pass
#
#     # Confidence intervals and t-values
#     if varcov == 'B':
#         CI = tcrit * np.sqrt(np.diag(V))[active_idx]
#         theta_std = np.sqrt(np.diag(V))[active_idx]
#     else:
#         CI = tcrit * np.sqrt(np.diag(V))
#         theta_std = np.sqrt(np.diag(V))
#     print(f'CI: {CI}')
#
#     # t_values = (theta_full[active_idx] * thetac_arr[active_idx]) / CI
#     with np.errstate(divide='ignore', invalid='ignore'):
#         numer = theta_full[active_idx] * thetac_arr[active_idx]
#         t_values = np.where(theta_std != 0, numer / theta_std, np.inf)
#
#     # Z-scores (normalized sensitivities)
#     resp_sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]]) for v in Q_accum])
#     Z = LSA * theta_std / resp_sigs[:, None]
#
#     # Compute LS, WLS, MLE, Chi, MSE, R2
#     LS = WLS = MLE = Chi = 0.0
#     total_err = 0.0
#     R2_data = {v: {'y':[], 'yp':[]} for v in Q_accum}
#     for v, vals in var_measurements.items():
#         for y, y_pred, s in vals:
#             r = y - y_pred
#             LS  += r**2
#             WLS += (r/s)**2
#             MLE += 0.5*(np.log(2*np.pi*s**2)+(r/s)**2)
#             Chi += r**2/s**2
#             total_err += r**2
#             R2_data[v]['y'].append(y)
#             R2_data[v]['yp'].append(y_pred)
#
#     R2_responses = {
#         v: 1 - np.sum((np.array(d['y'])-np.array(d['yp']))**2) /
#               np.sum((np.array(d['y'])-np.mean(d['y']))**2)
#         if len(d['y']) > 1 else np.nan
#         for v, d in R2_data.items()
#     }
#     MSE = total_err/obs if obs else np.nan
#
#     # Aggregate all y and y_pred across variables for total R2
#     all_y, all_yp = [], []
#     for v_vals in var_measurements.values():
#         for y, y_pred, _ in v_vals:
#             all_y.append(y)
#             all_yp.append(y_pred)
#
#     if len(all_y) > 1:
#         y_arr = np.array(all_y)
#         yp_arr = np.array(all_yp)
#         R2_total = 1 - np.sum((y_arr - yp_arr) ** 2) / np.sum((y_arr - np.mean(y_arr)) ** 2)
#     else:
#         R2_total = np.nan
#
#     return (
#         data, LS, MLE, Chi,
#         LSA, Z, V, CI, t_values,
#         t_m, tv_input_m, ti_input_m,
#         tv_output_m, ti_output_m,
#         WLS, obs, MSE, R2_responses, R2_total, M
#     )
#
# def _report(result, mutation, models, logging_flag):
#     """
#     Summarize and optionally log the uncertainty results.
#     """
#     scaled_params = {}
#     solver_parameters = {}
#     solver_cov_matrices = {}
#     solver_confidence_intervals = {}
#     sum_of_chi_squared = sum(1 / (res['Chi']) ** 2 for res in result.values())
#     for solver, res in result.items():
#         opt = res['optimization_result']
#         scpr = opt['scpr']
#         scaled_params[solver] = scpr.tolist()
#         solver_parameters[solver] = opt['x']
#         solver_cov_matrices[solver] = res['V_matrix']
#         solver_confidence_intervals[solver] = {
#             i: ci for i, ci in enumerate(res['CI'])
#         }
#         res['P'] = ((1 / (res['Chi']) ** 2) / sum_of_chi_squared) * 100
#         if logging_flag:
#             logger.info(f"Solver {solver}: Estimated θ = {scpr}")
#             logger.info(f"Solver {solver}: t-val = {res['t_values']}")
#             # logger.info(f"Solver {solver}: active = {res['optimization_result'][solver]['activeparams']}")
#     return (
#         scaled_params,
#         solver_parameters,
#         solver_cov_matrices,
#         solver_confidence_intervals,
#         result
#     )
#
# def _fdm_mesh_independency(
#     theta, thetac, solver, system, models,
#     tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars,
#     data, sens_method, varcov, resultpr
# ):
#     """
#     FDM mesh-dependency test to choose epsilon for sensitivities.
#     """
#     logger.info(f"Performing mesh-independency test for {solver}")
#     eps_values = np.logspace(-12, -1, 50)
#     eigs = []
#     varcov = 'M'
#     for eps in eps_values:
#         try:
#             out = _uncert_metrics(
#                 theta, data, [solver], thetac, {solver:eps}, [True]*len(theta),
#                 ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
#                 system, models, varcov, resultpr, sens_method=sens_method
#             )
#             V = out[6]  # returned V_matrix
#             eigs.append(np.linalg.eigvalsh(V))
#         except Exception:
#             eigs.append([np.nan]*len(theta))
#     eigs = np.array(eigs)
#
#     # Plot and save
#     folder = Path.cwd() / "meshindep_plots"
#     folder.mkdir(exist_ok=True)
#     base = folder / f"{solver}_eps_test"
#     plt.figure(figsize=(8,5))
#     for i in range(eigs.shape[1]):
#         plt.plot(eps_values, eigs[:,i], marker='o', label=f'eig{i+1}')
#     plt.xscale('log'); plt.yscale('log')
#     plt.xlabel('Epsilon'); plt.ylabel('Eigenvalues')
#     plt.legend(); plt.tight_layout()
#     plt.savefig(f"{base}.png"); plt.close()
#     pd.DataFrame(eigs, index=eps_values).to_excel(f"{base}.xlsx")
#
#     # Choose epsilon minimizing variation
#     valid = np.isfinite(eigs).all(axis=1)
#     idx = np.where(valid)[0]
#     win = 5
#     half = win//2
#     best, best_var = eps_values[0], np.inf
#     for i in idx[half:-half]:
#         var = eigs[i-half:i+half+1].std(axis=0).sum()
#         if var<best_var:
#             best_var, best = var, eps_values[i]
#     logger.info(f"Selected epsilon={best:.1e} for {solver}")
#     return best
#


# iden_uncert.py

import matplotlib.pyplot as plt
from pathlib import Path
from middoe.krnl_simula import simula
import logging
import numpy as np
import pandas as pd
from scipy import stats
from middoe.log_utils import  read_excel
from middoe.iden_parmest import _normalize_iden_opt, _match_times, _check_sigma_fallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def uncert(resultpr, system, models, iden_opt, case=None, data=None):
    r"""
    Perform comprehensive uncertainty and sensitivity analysis on parameter estimation results.

    This function evaluates parameter uncertainties using variance-covariance matrices,
    computes confidence intervals, performs sensitivity analysis via finite-difference
    methods, and assesses model quality through multiple statistical metrics (R², MSE,
    likelihood). It supports Hessian-based, Jacobian-based, and bootstrap-based
    covariance estimation.

    Parameters
    ----------
    resultpr : dict[str, scipy.optimize.OptimizeResult]
        Parameter estimation results from parmest(). Each OptimizeResult should contain:
            - 'scpr' : np.ndarray
                Optimized parameters.
            - 'hess_inv' : np.ndarray, optional
                Inverse Hessian (for var-cov='H').
            - 'v' : np.ndarray, optional
                Bootstrap covariance matrix (for var-cov='B').
            - 'fun' : float
                Final objective function value.

    system : dict
        System configuration including:
            - 'tvi' : dict
                Time-variant input definitions.
            - 'tii' : dict
                Time-invariant input definitions.
            - 'tvo' : dict
                Time-variant output definitions:
                    * 'meas' : bool — include in analysis
                    * 'unc' : float — measurement standard deviation
            - 'tio' : dict
                Time-invariant output definitions.

    models : dict
        Model definitions:
            - 'can_m' : list[str]
                Active model/solver names.
            - 'mutation' : dict[str, list[bool]]
                Parameter masks (True=free, False=fixed).
            - 'theta' : dict[str, list[float]]
                Nominal parameter values (updated in-place if case=None).
            - 'V_matrix' : dict[str, np.ndarray]
                Covariance matrices (updated in-place if case=None).
            - 'LSA' : dict[str, np.ndarray]
                Local sensitivity analysis matrices (updated in-place if case=None).

    iden_opt : dict
        Identification options:
            - 'sens_m' : str, optional
                Sensitivity method (from paper Table S3):
                    * 'central': Central finite difference (second-order accuracy,
                                 requires 2 evaluations per parameter)
                    * 'forward': Forward finite difference (first-order accuracy,
                                 requires 1 evaluation per parameter)

                Default: 'central'. Any other value raises ValueError.

            - 'var-cov' : str, optional
                Covariance method (from paper Table S3):
                    * 'H': Hessian-based (numerical Hessian of the objective computed
                           by parmest with var-cov='H'; scaled by the objective:
                           WLS 2*H^-1, MLE H^-1, LS/Chi 2*(fun/dof)*H^-1)
                    * 'M': Jacobian/FIM-based, (Q^T W Q)^-1 (alias 'J')
                    * 'B': Bootstrap-based (global, requires var-cov='B' in parmest)

                Default: 'H' if not specified.

            - 'ob' : str, optional
                Objective used in parmest ('LS', 'WLS', 'MLE', 'Chi'; alias 'CS').
                Needed for var-cov='H'.

            - 'eps' : float or dict[str, float], optional
                Finite-difference step size. If None, automatically determined via
                mesh-independency test for each model.
            - 'log' : bool, optional
                Enable verbose logging (default: False).

    case : str, optional
        Analysis mode:
            - None: Update models dictionary in-place with results.
            - Any other value: Return results without modifying models.

    data : dict, optional
        Experimental data ``{sheet_name: DataFrame}``. If None (default) the data
        are read from ``data.xlsx`` in the working directory.

    Returns
    -------
    analysis_results : dict
        Dictionary with:
            - 'results' : dict[str, dict]
                Detailed uncertainty analysis for each model/solver:
                    * 'optimization_result' : OptimizeResult
                        Original parameter estimation result.
                    * 'data' : dict
                        Experimental data (unchanged).
                    * 'LS' : float
                        Least Squares objective.
                    * 'WLS' : float
                        Weighted Least Squares objective.
                    * 'MLE' : float
                        Maximum Likelihood Estimation objective.
                    * 'MSE' : float
                        Mean Squared Error.
                    * 'Chi' : float
                        Relative least squares, ``sum(((y - y_hat)/max(|y|, sigma))^2)``
                        (the same 'Chi' as minimised by parmest).
                    * 'sigma_rows' : np.ndarray
                        Per-row measurement sigma used (rows of 'LSA').
                    * 'LSA' : np.ndarray, shape (n_observations, n_active_params)
                        Local Sensitivity Analysis matrix (Jacobian).
                    * 'Z' : np.ndarray
                        Normalized sensitivities (Z-scores).
                    * 'V_matrix' : np.ndarray, shape (n_active_params, n_active_params)
                        Variance-covariance matrix.
                    * 'CI' : np.ndarray
                        95% confidence intervals (half-widths).
                    * 't_values' : np.ndarray
                        t-statistics for parameter significance tests.
                    * 'R2_responses' : dict[str, float]
                        R² for each response variable.
                    * 'R2_total' : float
                        Overall R² across all responses.
                    * 'M' : np.ndarray
                        Fisher Information Matrix.
                    * 't_m' : dict
                        Time vectors for each sheet.
                    * 'tv_input_m' : dict
                        Time-variant inputs for each sheet.
                    * 'ti_input_m' : dict
                        Time-invariant inputs for each sheet.
                    * 'tv_output_m' : dict
                        Simulated time-variant outputs for each sheet.
                    * 'ti_output_m' : dict
                        Simulated time-invariant outputs for each sheet.
                    * 'estimations' : np.ndarray
                        Final parameter estimates (same as 'scpr').
                    * 'found_eps' : float
                        Finite-difference step size used.
                    * 'P' : float
                        Relative probability (model weight) based on Chi-squared.
            - 'obs' : int
                Total number of observations used.

    Notes
    -----
    **Sensitivity Analysis**:
    The Local Sensitivity Analysis (LSA) matrix is computed via finite differences:

        - **Forward**: \( \frac{\partial y}{\partial \theta_i} \approx \frac{y(\theta + h e_i) - y(\theta)}{h} \)
        - **Central**: \( \frac{\partial y}{\partial \theta_i} \approx \frac{y(\theta + h e_i) - y(\theta - h e_i)}{2h} \)

    Central differences are more accurate (second-order) but require twice as many simulations.

    **Variance-Covariance Estimation Methods** (from paper Table S3):

        - **'H' (Hessian-based)**: Local uncertainty from the Hessian \( \mathbf{H} \) of the
          fitted objective at the optimum (computed by ``parmest`` with finite differences):
          \( \mathbf{V} = 2\mathbf{H}^{-1} \) for 'WLS', \( \mathbf{V} = \mathbf{H}^{-1} \) for
          'MLE', and \( \mathbf{V} = 2 (f/\nu) \mathbf{H}^{-1} \) for 'LS' and 'Chi', where
          \( f \) is the objective value and \( \nu \) the degrees of freedom.
          Advantages: includes the curvature of the real objective.
          Disadvantages: needs a well-converged fit (use a tight 'tol').

        - **'M' (Fisher-matrix based, alias 'J')**: Local uncertainty from the Fisher Information Matrix.
          \[
          \mathbf{M} = \mathbf{Q}^T \mathbf{W} \mathbf{Q}, \quad \mathbf{V} = \mathbf{M}^{-1}
          \]
          where \( \mathbf{Q} \) is the sensitivity matrix and \( \mathbf{W} = \text{diag}(1/\sigma_i^2) \).
          Advantages: More robust than Hessian for parameter-sensitive models
          Disadvantages: Requires explicit sensitivity computation

        - **'B' (Bootstrap)**: Uncertainty from refitting resampled data sets (pairs bootstrap).
          Accounts for parameter bounds via truncated normal correction.
          Advantages: Most robust, accounts for non-linearity
          Disadvantages: Computationally expensive, requires running parmest with var-cov='B'

    **Confidence Intervals**:
    95% confidence intervals are computed using t-distribution:

        \[
        CI_i = t_{0.975, \nu} \cdot \sqrt{V_{ii}}
        \]

    where \( \nu = n_{obs} - n_{params} \) is degrees of freedom.

    **t-Statistics**:
    Parameter significance is tested via:

        \[
        t_i = \frac{\theta_i}{\sqrt{V_{ii}}}
        \]

    Large :math:`|t_i|` (typically > 1.96 for 95% confidence) indicates the parameter
    is significantly different from zero.

    **Normalized Sensitivities (Z-scores)**:
    Sensitivities are normalized by parameter uncertainties and measurement errors:

        \[
        Z_{ij} = \frac{\partial y_i}{\partial \theta_j} \cdot \frac{\sqrt{V_{jj}}}{\sigma_i}
        \]

    This quantifies the relative influence of parameter \( j \) on observable \( i \).

    **Model Weights (P-test)**:
    When multiple models are analyzed, relative probabilities are computed via P-test
    (Galvanin et al. 2016; paper eq. 2 without the misprinted leading "1 -"):

        \[
        P_k = \frac{1/\chi_k^2}{\sum_m 1/\chi_m^2} \times 100\%
        \]

    with \( \chi^2 \) the 'Chi' value. A model with \( \chi^2 = 0 \) gets 100%.
    This provides model comparison based on goodness-of-fit (higher probability → better model).

    **Mesh-Independency Test**:
    If epsilon is not provided, it is automatically determined by varying \( h \) over
    \( [10^{-12}, 10^{-1}] \) and selecting the value that minimizes sensitivity matrix
    variance. Results are saved in './meshindep_plots/'.

    **Ill-Conditioning Diagnostics**:
    The function automatically logs conditioning metrics:

        - Rank of sensitivity matrix \( \mathbf{Q} \)
        - Condition numbers of \( \mathbf{Q} \) and \( \mathbf{M} \)
        - Minimum eigenvalue of \( \mathbf{M} \)
        - Warnings if system is ill-conditioned (condition number > \( 10^8 \))

    **In-Place Updates** (case=None):
    When case=None, the function updates models dictionary:

        - models['theta']: Updated with final estimates
        - models['V_matrix']: Updated with covariance matrices
        - models['LSA']: Updated with sensitivity matrices
        - models['thetastart']: Preserves pre-uncertainty estimates

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
    parmest : Parameter estimation (prerequisite for uncertainty analysis).
    _uncert_metrics : Core uncertainty computation routine.
    _fdm_mesh_independency : Epsilon selection via mesh-independency test.

    Examples
    --------
    >>> # After parameter estimation
    >>> results_pe = parmest(system, models, iden_opt={'meth': 'SLSQP', 'ob': 'WLS'})
    >>>
    >>> # Uncertainty analysis with Fisher-matrix covariance
    >>> iden_opt_ua = {'sens_m': 'central', 'var-cov': 'M', 'log': True}
    >>> uncert_results = uncert(results_pe, system, models, iden_opt_ua)
    >>>
    >>> # Access results
    >>> for model in uncert_results['results']:
    ...     res = uncert_results['results'][model]
    ...     print(f"Model {model}:")
    ...     print(f"  R² = {res['R2_total']:.4f}")
    ...     print(f"  Parameters: {res['estimations']}")
    ...     print(f"  CI (95%): {res['CI']}")
    ...     print(f"  t-values: {res['t_values']}")
    ...     print(f"  Significant (|t|>1.96): {np.abs(res['t_values']) > 1.96}")

    >>> # Bootstrap-based uncertainty (requires bootstrap in parmest)
    >>> results_boot = parmest(system, models,
    ...                       iden_opt={'meth': 'SLSQP', 'ob': 'WLS',
    ...                                 'var-cov': 'B', 'nboot': 500})
    >>> uncert_boot = uncert(results_boot, system, models, {'var-cov': 'B'})
    >>> print(f"Bootstrap covariance:\n{uncert_boot['results']['M1']['V_matrix']}")
    """

    # Sensitivity method and varcov selection
    # validate options and map aliases ('J'->'M', 'CS'->'Chi', ...); written back
    _normalize_iden_opt(iden_opt, need_meth=False)
    data = data if data is not None else read_excel()
    _check_sigma_fallback(data, system)   # raise/warn where 'unc' replaces a missing MES_E
    sens_method = iden_opt.get('sens_m') or 'central'
    varcov_key  = iden_opt['var-cov']  # 'M', 'H', or 'B'

    # Identify measurement variables
    tv_iphi_vars = list(system.get('tvi', {}))
    ti_iphi_vars = list(system.get('tii', {}))
    tv_ophi_vars = [v for v,c in system.get('tvo', {}).items() if c.get('meas', True)]
    ti_ophi_vars = [v for v,c in system.get('tio', {}).items() if c.get('meas', True)]

    # Collect measurement uncertainties (not directly used here)
    std_dev = {}
    for grp in ('tvo','tio'):
        for v,cfg in system.get(grp, {}).items():
            if cfg.get('meas', True):
                s = cfg.get('unc', 1.0)
                std_dev[v] = 1.0 if s is None or np.isnan(s) else s

    # Prepare epsilon settings
    eps_in = iden_opt.get('eps', None)
    if not isinstance(eps_in, dict):
        epsf = {sv: eps_in for sv in resultpr}
    else:
        epsf = eps_in.copy()

    resultun = {}
    observed_values = None

    # Loop over each solver's optimization result
    for solver, solver_results in resultpr.items():
        thetac = solver_results['scpr']
        thetas = models.get('mutation', {}).get(solver, [])
        # If epsilon not provided, run mesh-independency test
        if epsf.get(solver) is None:
            epsf[solver] = _fdm_mesh_independency(
                theta=np.ones_like(thetac),
                thetac=thetac,
                solver=solver,
                system=system,
                models=models,
                tv_iphi_vars=tv_iphi_vars,
                ti_iphi_vars=ti_iphi_vars,
                tv_ophi_vars=tv_ophi_vars,
                ti_ophi_vars=ti_ophi_vars,
                data=data,
                sens_method=sens_method,
                varcov=varcov_key,
                resultpr=resultpr
            )

        # Compute uncertainty metrics
        (
            optimized_data,
            LS, MLE, Chi,
            LSA, Z, V_matrix,
            CI, t_values,
            t_m, tv_input_m, ti_input_m,
            tv_output_m, ti_output_m,
            WLS, obs, MSE,
            R2_responses, R2_total, M
        ) = _uncert_metrics(
            theta=np.ones_like(thetac),
            data=data,
            active_solvers=[solver],
            thetac=thetac,
            eps=epsf,
            thetas=thetas,
            ti_iphi_vars=ti_iphi_vars,
            tv_iphi_vars=tv_iphi_vars,
            tv_ophi_vars=tv_ophi_vars,
            ti_ophi_vars=ti_ophi_vars,
            system=system,
            models=models,
            varcov=varcov_key,
            resultpr=resultpr,
            sens_method=sens_method,
            objf=iden_opt.get('ob')
        )
        observed_values = obs
        sigma_rows = resultpr[solver].get('sigma_rows') if isinstance(resultpr[solver], dict) else None

        resultun[solver] = {
            'optimization_result': solver_results,
            'data': optimized_data,
            'LS': LS,
            'MLE': MLE,
            'MSE': MSE,
            'Chi': Chi,
            'LSA': LSA,
            'Z': Z,
            'WLS': WLS,
            'R2_responses': R2_responses,
            'R2_total': R2_total,
            'V_matrix': V_matrix,
            'CI': CI,
            't_values': t_values,
            't_m': t_m,
            'tv_input_m': tv_input_m,
            'ti_input_m': ti_input_m,
            'tv_output_m': tv_output_m,
            'ti_output_m': ti_output_m,
            'estimations': thetac,
            'found_eps': epsf[solver],
            'M': M,
            'sigma_rows': sigma_rows
        }

    # Store back eps
    iden_opt['eps'] = epsf

    # Summarize and report
    (
        scaled_params,
        solver_parameters,
        solver_cov_matrices,
        solver_confidence_intervals,
        resultun2
    ) = _report(resultun, models.get('mutation', {}), models, iden_opt.get('log', False))

    # Optionally update models in‐place
    if case is None:
        models.setdefault('V_matrix', {})
        for sv in models['can_m']:
            if 'thetastart' not in models:
                models['thetastart'] = {}
                for sv in models['can_m']:
                    if sv not in models['thetastart']:
                        models['thetastart'][sv] = models['theta'][sv]

            models['theta'][sv]    = scaled_params[sv]
            models['V_matrix'][sv] = resultun[sv]['V_matrix']
            # Initialize 'LSA' as a dict if it does not exist; otherwise keep existing
            if 'LSA' not in models or not isinstance(models['LSA'], dict):
                models['LSA'] = {}

            # Update (or add) the LSA values for each solver key
            for sv in models['can_m']:
                models['theta'][sv] = scaled_params.get(sv)
                models['V_matrix'][sv] = resultun.get(sv, {}).get('V_matrix')

                lsa_value = resultun.get(sv, {}).get('LSA')
                if lsa_value is not None:
                    models['LSA'][sv] = lsa_value

    return {'results': resultun2, 'obs': observed_values}


def _uncert_metrics(
    theta, data, active_solvers, thetac, eps, thetas,
    ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
    system, models, varcov, resultpr, sens_method='forward', objf=None
):
    r"""
    Compute comprehensive uncertainty metrics including sensitivities, covariance, and quality measures.

    This is the core computational routine for uncertainty analysis. It performs forward
    simulations with perturbed parameters to compute finite-difference sensitivities,
    constructs variance-covariance matrices using various methods, computes confidence
    intervals and t-statistics, and calculates model quality metrics (R², MSE, likelihood).
    It also includes automatic ill-conditioning diagnostics.

    Parameters
    ----------
    theta : np.ndarray
        Normalized parameter vector (typically np.ones_like(thetac) for uncertainty analysis).
    data : dict
        Experimental data sheets (from read_excel). Each sheet contains:
            - 'X:all' : Time vector
            - '{var}t', '{var}l' : Switching times/levels for time-variant inputs
            - '{var}' : Time-invariant input values
            - 'MES_X:{var}', 'MES_Y:{var}' : Measurement times and values
            - 'MES_E:{var}' : Measurement uncertainties (fractional std dev)
            - 'CVP:{var}' : Control variable parameterisation flags
    active_solvers : list[str]
        Active model/solver names (typically single element).
    thetac : np.ndarray
        Parameter values at which to evaluate sensitivities (from optimization).
    eps : dict[str, float]
        Finite-difference step sizes for each solver.
    thetas : list[bool]
        Parameter activity mask (True=free, False=fixed).
    ti_iphi_vars : list[str]
        Time-invariant input variable names.
    tv_iphi_vars : list[str]
        Time-variant input variable names.
    tv_ophi_vars : list[str]
        Time-variant output variable names (to be measured).
    ti_ophi_vars : list[str]
        Time-invariant output variable names (to be measured).
    system : dict
        System configuration.
    models : dict
        Model definitions.
    varcov : str
        Covariance method: 'M' (measurement), 'H' (Hessian), or 'B' (bootstrap).
    resultpr : dict
        Parameter estimation results (used for 'H' and 'B' methods).
    sens_method : str, optional
        Sensitivity method: 'forward' or 'central' (default: 'forward').
    objf : str, optional
        Objective minimised by parmest ('LS', 'WLS', 'MLE', 'Chi'); sets the
        scaling of the Hessian-based covariance (varcov='H').

    Returns
    -------
    data : dict
        Input data (unchanged, returned for consistency).
    LS : float
        Least Squares objective: \( \sum_i (y_i - \hat{y}_i)^2 \).
    MLE : float
        Negative log-likelihood: \( \sum_i [\log(2\pi\sigma_i^2) + (r_i/\sigma_i)^2]/2 \).
    Chi : float
        Relative least squares: \( \sum_i (r_i/|y_i|)^2 \) (|y| floored at 1e-8).
    LSA : np.ndarray, shape (n_observations, n_active_params)
        Local Sensitivity Analysis matrix (Jacobian): \( Q_{ij} = \partial y_i / \partial \theta_j \).
    Z : np.ndarray, shape (n_observations, n_active_params)
        Normalized sensitivities (Z-scores): \( Z_{ij} = Q_{ij} \cdot \sqrt{V_{jj}} / \sigma_i \).
    V_matrix : np.ndarray, shape (n_active_params, n_active_params)
        Variance-covariance matrix of parameters.
    CI : np.ndarray, shape (n_active_params,)
        95% confidence interval half-widths: \( t_{0.975, \nu} \cdot \sqrt{V_{ii}} \).
    t_values : np.ndarray, shape (n_active_params,)
        t-statistics for parameter significance: \( t_i = \theta_i / \sqrt{V_{ii}} \).
    t_m : dict[str, list[float]]
        Time vectors for each data sheet.
    tv_input_m : dict[str, dict]
        Time-variant inputs for each sheet.
    ti_input_m : dict[str, dict]
        Time-invariant inputs for each sheet.
    tv_output_m : dict[str, dict]
        Simulated time-variant outputs for each sheet.
    ti_output_m : dict[str, dict]
        Simulated time-invariant outputs for each sheet.
    WLS : float
        Weighted Least Squares objective: \( \sum_i (r_i/\sigma_i)^2 \).
    obs : int
        Total number of observations (measurements).
    MSE : float
        Mean Squared Error: \( \sum_i r_i^2 / n_{obs} \).
    R2_responses : dict[str, float]
        Coefficient of determination for each response variable.
    R2_total : float
        Overall R² across all response variables.
    M : np.ndarray, shape (n_active_params, n_active_params)
        Fisher Information Matrix: \( M = Q^T W Q \) (for varcov='M').

    Notes
    -----
    **Finite-Difference Sensitivity Computation**:
        - **Forward**: \( \frac{\partial y}{\partial \theta_j} \approx \frac{y(\theta + h e_j) - y(\theta)}{h \cdot \theta_c[j]} \)
        - **Central**: \( \frac{\partial y}{\partial \theta_j} \approx \frac{y(\theta + h e_j) - y(\theta - h e_j)}{2h \cdot \theta_c[j]} \)

    Central differences provide second-order accuracy but require twice as many simulations.

    **Measurement Error Handling**:
    Measurement uncertainties ('MES_E:{var}') are absolute standard deviations.
    If the column is missing, system['tvo'|'tio'][var]['unc'] is used (default 1.0).
    Sigmas must be finite and > 0 (ValueError otherwise). Rows with NaN in MES_X
    or MES_Y are skipped.

    **Variance-Covariance Methods**:
        1. **'M' (Measurement-based)**:
            - Constructs weighted sensitivity matrix: \( \tilde{Q} = W^{1/2} Q \)
            - Fisher Information: \( M = \tilde{Q}^T \tilde{Q} \)
            - Covariance: \( V = M^{-1} \)
            - Assumes measurement errors dominate parameter uncertainties

        2. **'H' (Hessian-based)**:
            - Uses the physical inverse Hessian of the objective from parmest
            - WLS: \( V = 2 H^{-1} \); MLE: \( V = H^{-1} \);
              LS and Chi: \( V = 2 (f_{opt}/\nu) H^{-1} \), \( \nu = n_{obs} - n_{active} \)
            - V is restricted to the active (free) parameters

        3. **'B' (Bootstrap-based)**:
            - Uses pre-computed bootstrap covariance from parmest()
            - Accounts for parameter bounds via truncated normal correction
            - Most robust but computationally expensive

    **Ill-Conditioning Diagnostics**:
    The function automatically computes and logs:
        - **Rank(Q)**: Number of linearly independent sensitivity directions
        - **cond(Q)**: Condition number of sensitivity matrix (ratio of largest/smallest singular values)
        - **cond(M)**: Condition number of Fisher Information Matrix
        - **λ_min(M)**: Minimum eigenvalue (near-zero indicates non-identifiability)

    **Thresholds**:
        - Sloppy: cond(M) ≥ \( 10^3 \) (some parameter combinations poorly determined)
        - Ill-conditioned: cond(M) ≥ \( 10^8 \) or λ_min < \( 10^{-12} \)

    Diagnostic summary is logged at INFO level and stored in resultpr[solver]['ill_report'].

    **Confidence Intervals**:
    95% confidence intervals use t-distribution with \( \nu = n_{obs} - n_{params} \) degrees of freedom:
        \[
        [\theta_i - t_{0.975, \nu} \sqrt{V_{ii}}, \ \theta_i + t_{0.975, \nu} \sqrt{V_{ii}}]
        \]

    **t-Statistics**:
    Tests null hypothesis \( H_0: \theta_i = 0 \):
        \[
        t_i = \frac{\theta_i}{\sqrt{V_{ii}}}
        \]
    Typically, \( |t_i| > 2 \) indicates significance at 95% level (approximate).

    **Normalized Sensitivities (Z-scores)**:
    Quantify relative parameter influence accounting for uncertainties:
        \[
        Z_{ij} = \frac{\partial y_i}{\partial \theta_j} \cdot \frac{\sqrt{V_{jj}}}{\sigma_i}
        \]
    Large \( |Z_{ij}| \) means parameter \( j \) significantly affects observable \( i \).

    **Model Quality Metrics**:
        - **R²**: Fraction of variance explained (1 = perfect, < 0 = worse than mean)
        - **MSE**: Average squared prediction error
        - **Chi²**: Weighted sum of squared normalized residuals

    References
    ----------
    .. [1] Bates, D. M., & Watts, D. G. (1988).
       *Nonlinear Regression Analysis and Its Applications*. Wiley.

    .. [2] Seber, G. A. F., & Wild, C. J. (2003).
       *Nonlinear Regression*. Wiley-Interscience.

    .. [3] Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2011).
       Geometry of nonlinear least squares with applications to sloppy models and optimization.
       *Physical Review E*, 83(3), 036701.

    See Also
    --------
    uncert : Main entry point that calls this function.
    _fdm_mesh_independency : Epsilon selection for finite differences.

    Examples
    --------
    >>> # Typically called internally by uncert()
    >>> out = _uncert_metrics(
    ...     theta=np.ones(3), data=data, active_solvers=['M1'],
    ...     thetac=theta_opt, eps={'M1': 1e-6}, thetas=[True]*3,
    ...     ti_iphi_vars=ti_vars, tv_iphi_vars=tv_vars,
    ...     tv_ophi_vars=outputs, ti_ophi_vars=[],
    ...     system=system, models=models, varcov='M',
    ...     resultpr=results, sens_method='central'
    ... )
    >>> LSA, V, CI, t_vals = out[4], out[6], out[7], out[8]
    >>> print(f"Sensitivities shape: {LSA.shape}")
    >>> print(f"Confidence intervals: {CI}")
    """
    # ---- diagnostic thresholds (adjust here if needed) -----------------
    _SLOPPY_THRESH = 1e3     # 'sloppy' if cond(M) >= this
    _COND_THRESH   = 1e8     # ill-conditioned if cond(M) >= this
    _TINY_EIG      = 1e-12   # ill if min eigenvalue < this
    _REPORT_LEVEL  = logging.INFO  # keep it white in most color schemes
    _SIG_FLOOR     = 1e-12   # floor for absolute sigmas

    if sens_method not in ('forward', 'central'):
        raise ValueError(f"Unknown sens_m={sens_method!r}; valid values: ['central', 'forward']")
    theta_full = np.array(theta, dtype=float)
    active_idx = np.where(thetas)[0]
    print(f'active_idx: {active_idx}')
    n_active   = len(active_idx)
    thetac_arr = np.array(thetac, dtype=float)

    # Containers
    var_measurements = {}                     # {var: [(y_true, y_pred, sigma_abs), ...]}
    Q_accum = {v: [] for v in (tv_ophi_vars + ti_ophi_vars)}
    t_m = {}
    tv_input_m, ti_input_m = {}, {}
    tv_output_m, ti_output_m = {}, {}

    solver = active_solvers[0]
    h = eps[solver]

    # Loop over data sheets
    for sheet_name, sheet in data.items():
        # Time grid
        t_all = np.unique(sheet.get("X:all", pd.Series()).dropna().values)

        # Switch-pressure inputs
        swps = {}
        for v in tv_iphi_vars:
            tkey, lkey = f"{v}t", f"{v}l"
            if tkey in sheet and lkey in sheet:
                ta = sheet[tkey].dropna().values
                la = sheet[lkey].dropna().values
                if ta.size and la.size:
                    swps[tkey], swps[lkey] = ta, la

        # Input data
        ti_in = {v: sheet.get(v, pd.Series([np.nan])).iloc[0] for v in ti_iphi_vars}
        tv_in = {v: sheet.get(v, pd.Series()).dropna().values for v in tv_iphi_vars}
        cvp   = {v: sheet[f"CVP:{v}"].iloc[0] for v in system.get('tvi', {})}

        # Reference simulation
        tv_ref, ti_ref, _ = simula(
            t_all, swps, ti_in,
            {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
            theta_full, thetac_arr, cvp, tv_in, solver, system, models
        )

        # Store simulation outputs
        t_m[sheet_name]         = t_all.tolist()
        tv_input_m[sheet_name]  = tv_in
        ti_input_m[sheet_name]  = ti_in
        tv_output_m[sheet_name] = tv_ref
        ti_output_m[sheet_name] = ti_ref

        # Build perturbed sims for FD sensitivities
        perturbed = {}
        for p in active_idx:
            if sens_method == 'forward':
                thp = theta_full.copy(); thp[p] += h
                tvp, tip, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thp, thetac_arr, cvp, tv_in, solver, system, models
                )
                perturbed[p] = ('forward', (tvp, tip))
            else:
                thp = theta_full.copy(); thp[p] += h
                thm = theta_full.copy(); thm[p] -= h
                tvp, tip, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thp, thetac_arr, cvp, tv_in, solver, system, models
                )
                tvm, tim, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thm, thetac_arr, cvp, tv_in, solver, system, models
                )
                perturbed[p] = ('central', (tvp, tip, tvm, tim))

        # Collect residuals and sensitivities
        for var in tv_ophi_vars + ti_ophi_vars:
            if var in tv_ophi_vars:
                # time-varying measurement
                xcol = f"MES_X:{var}"
                ycol = f"MES_Y:{var}"
                ecol = f"MES_E:{var}"
                if xcol not in sheet or ycol not in sheet:
                    continue
                mask = ~sheet[xcol].isna().values & ~sheet[ycol].isna().values
                times = sheet[xcol][mask].values.astype(float)
                y_t   = sheet[ycol][mask].values.astype(float)
                if times.size == 0:
                    continue
                pos   = _match_times(t_all, times, var)
                y_p   = np.asarray(tv_ref[var], dtype=float)[pos]

                # absolute sigma from sheet; fallback: system 'unc' (same as parmest)
                if ecol in sheet:
                    y_e = pd.to_numeric(sheet[ecol][mask], errors='raise').values.astype(float)
                else:
                    y_e = np.full_like(y_t, float(system['tvo'][var].get('unc', 1.0)), dtype=float)
                if not np.all(np.isfinite(y_e)) or np.any(y_e <= 0):
                    raise ValueError(f"Sheet '{sheet_name}', {ecol}: sigma must be finite and > 0")
            else:
                # time-independent measurement
                ycol = f"MES_Y:{var}"
                ecol = f"MES_E:{var}"
                if ycol not in sheet or pd.isna(sheet[ycol].iloc[0]):
                    continue
                y_t  = np.array([float(sheet[ycol].iloc[0])])
                y_p  = np.array([float(np.ravel(ti_ref[var])[0])])
                if ecol in sheet and not pd.isna(sheet[ecol].iloc[0]):
                    y_e = np.array([float(pd.to_numeric(sheet[ecol].iloc[0], errors='raise'))], dtype=float)
                else:
                    y_e = np.array([float(system['tio'][var].get('unc', 1.0))], dtype=float)
                if not np.all(np.isfinite(y_e)) or np.any(y_e <= 0):
                    raise ValueError(f"Sheet '{sheet_name}', {ecol}: sigma must be finite and > 0")

            var_measurements.setdefault(var, []).extend(zip(y_t, y_p, y_e))

            # Sensitivity block
            cols = []
            for p in active_idx:
                mode, sims = perturbed[p]
                if mode == 'forward':
                    tvp, tip = sims
                    if var in tv_ophi_vars:
                        y2 = np.asarray(tvp[var], dtype=float)[pos]
                    else:
                        y2 = np.array([float(np.ravel(tip[var])[0])])
                    d = (y2 - y_p) / (h * thetac_arr[p])
                else:
                    tvp, tip, tvm, tim = sims
                    if var in tv_ophi_vars:
                        y_p_p = np.asarray(tvp[var], dtype=float)[pos]
                        y_p_m = np.asarray(tvm[var], dtype=float)[pos]
                    else:
                        y_p_p = np.array([float(np.ravel(tip[var])[0])])
                        y_p_m = np.array([float(np.ravel(tim[var])[0])])
                    d = (y_p_p - y_p_m) / (2*h*thetac_arr[p])
                cols.append(d)
            if len(cols):
                Q_accum[var].append(np.stack(cols, axis=1))

    # Flatten Q into LSA
    Q_accum = {v: q for v, q in Q_accum.items() if q}
    if not Q_accum:
        raise ValueError("uncert: no valid measurements (MES_X/MES_Y) found in the data")
    LSA = np.vstack([np.vstack(Q_accum[v]) for v in Q_accum])
    obs = sum(len(vals) for vals in var_measurements.values())
    dof = max(obs - n_active, 1)
    tcrit = stats.t.ppf(0.975, dof)

    # --- Variance–Covariance Selection ---
    if varcov == 'M':
        # Using absolute sigmas we just constructed
        sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]])
                               for v in Q_accum])
        # Avoid forming giant diagonal: scale rows instead
        scaled_J = LSA / sigs[:, None]
        M = scaled_J.T @ scaled_J
        V = np.linalg.pinv(M)



    elif varcov == 'H':
        hess_inv = resultpr[solver].get('hess_inv')
        try:
            hess_inv = np.asarray(hess_inv, dtype=float)
        except (TypeError, ValueError):
            hess_inv = None
        n_all = len(thetac_arr)
        if hess_inv is None or hess_inv.shape != (n_all, n_all):
            raise ValueError(f"var-cov='H' needs the Hessian-based covariance from parmest "
                             f"(run parmest with iden_opt['var-cov']='H'); model '{solver}' has none")
        fun = float(resultpr[solver].get('fun', 1.0))
        # hess_inv is the physical inverse Hessian of the objective (plain sums)
        if objf == 'WLS':
            V = 2.0 * hess_inv
        elif objf == 'MLE':
            V = hess_inv
        elif objf in ('LS', 'Chi'):
            V = 2.0 * (fun / dof) * hess_inv
        else:   # objective unknown: residual-variance scaling (legacy)
            V = (fun / dof) * hess_inv
        V = V[np.ix_(active_idx, active_idx)]   # active parameters only (as 'M')
        bad = np.where(~(np.diag(V) > 0))[0]
        if bad.size:
            logger.warning(
                f"Model '{solver}': the Hessian of the objective is not positive definite at the "
                f"reported optimum (non-positive variance for active parameter(s) {bad.tolist()}). "
                f"Their CI/t-values will be NaN. The optimiser has probably stopped before a true "
                f"minimum (e.g. a loose 'tol'); tighten it or use iden_opt['var-cov']='M'.")
        M = np.linalg.pinv(V)

    elif varcov == 'B':
        V = resultpr[solver].get('v')
        if V is None:
            raise ValueError(f"No bootstrap v matrix found for model '{solver}'")
        # V is full size with zero rows/columns for fixed parameters; the information
        # matrix is that of the active parameters (as for 'M' and 'H')
        M = np.linalg.pinv(np.asarray(V, float)[np.ix_(active_idx, active_idx)])
        logger.info(f"Using bootstrap var–cov for {solver}")

    else:
        raise ValueError(f"Unknown varcov option '{varcov}'")

    # --- Ill-conditioning diagnostics ----------------------------------
    # Computed on dimensionless quantities, so that parameter units do not decide the result:
    #   Q_s = LSA * |theta| / sigma   (relative-parameter, sigma-weighted sensitivities)
    #   M_s = D M D, D = diag(|theta|) (information on relative parameter changes;
    #         1/sqrt(eig) = relative standard deviation along an eigenvector)
    ill_report = {}
    th_abs = np.abs(theta_full[active_idx] * thetac_arr[active_idx])
    sig_rows = np.concatenate([np.array([s for _, _, s in var_measurements[v]]) for v in Q_accum])
    try:
        Qs = LSA * th_abs[None, :] / sig_rows[:, None]
        s = np.linalg.svd(Qs, compute_uv=False)
        ill_report['Q_singular_values'] = s
        ill_report['Q_rank'] = int(np.sum(s > s[0] * max(Qs.shape) * np.finfo(float).eps))
        ill_report['Q_cond'] = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    except Exception as e:
        ill_report['Q_error'] = str(e)

    try:
        eig = np.linalg.eigvalsh(th_abs[:, None] * np.asarray(M, float) * th_abs[None, :])
        ill_report['M_eigenvalues'] = eig
        ill_report['M_min_eig'] = float(eig.min())
        ill_report['M_max_eig'] = float(eig.max())
        ill_report['M_cond'] = float(eig.max() / eig.min()) if eig.min() > 0 else np.inf
    except Exception as e:
        ill_report['M_error'] = str(e)

    ill_report['is_rank_deficient'] = ill_report.get('Q_rank', n_active) < n_active
    ill_report['is_sloppy'] = ill_report.get('M_cond', np.inf) >= _SLOPPY_THRESH
    ill_report['is_ill_conditioned'] = (
        ill_report['is_rank_deficient'] or
        ill_report.get('M_cond', 0) >= _COND_THRESH or
        ill_report.get('M_min_eig', _TINY_EIG) < _TINY_EIG
    )
    ill_report['summary'] = (
        f"(scaled) Rank(Q)={ill_report.get('Q_rank','?')}/{n_active}, "
        f"cond(Q)≈{ill_report.get('Q_cond',np.nan):.2e}, "
        f"cond(M)≈{ill_report.get('M_cond',np.nan):.2e}, "
        f"λ_min(M)≈{ill_report.get('M_min_eig',np.nan):.2e}; "
        f"{'ILL-CONDITIONED' if ill_report['is_ill_conditioned'] else 'OK'}"
    )

    logger.log(_REPORT_LEVEL, "Conditioning diagnostics: %s", ill_report['summary'])

    # Stash report into resultpr without changing the API
    try:
        if isinstance(resultpr.get(solver, None), dict):
            resultpr[solver]['ill_report'] = ill_report
    except Exception:
        pass

    # Confidence intervals and t-values
    if varcov == 'B':
        CI = tcrit * np.sqrt(np.diag(V))[active_idx]
        theta_std = np.sqrt(np.diag(V))[active_idx]
    else:
        CI = tcrit * np.sqrt(np.diag(V))
        theta_std = np.sqrt(np.diag(V))
    print(f'CI: {CI}')

    with np.errstate(divide='ignore', invalid='ignore'):
        numer = theta_full[active_idx] * thetac_arr[active_idx]
        # no sensitivity / zero variance -> t is undefined (NaN), not infinite
        t_values = np.where(theta_std != 0, numer / theta_std, np.nan)

    # Z-scores (normalized sensitivities)
    resp_sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]]) for v in Q_accum])
    Z = LSA * theta_std / resp_sigs[:, None]
    # per-row sigma (rows of LSA), exported for sc_estima's scaled sensitivities
    try:
        if isinstance(resultpr.get(solver, None), dict):
            resultpr[solver]['sigma_rows'] = resp_sigs
    except Exception:
        pass

    # Compute LS, WLS, MLE, Chi, MSE, R2
    LS = WLS = MLE = Chi = 0.0
    total_err = 0.0
    R2_data = {v: {'y':[], 'yp':[]} for v in Q_accum}
    for v, vals in var_measurements.items():
        for y, y_pred, s in vals:
            r = y - y_pred
            LS  += r**2
            WLS += (r/s)**2
            MLE += 0.5*(np.log(2*np.pi*s**2)+(r/s)**2)
            s_floor = s if np.isfinite(s) else 0.0
            Chi += (r / max(abs(y), s_floor, 1e-8))**2   # relative LS, |y| floored at sigma; same as parmest 'Chi'
            total_err += r**2
            R2_data[v]['y'].append(y)
            R2_data[v]['yp'].append(y_pred)

    R2_responses = {
        v: 1 - np.sum((np.array(d['y'])-np.array(d['yp']))**2) /
              np.sum((np.array(d['y'])-np.mean(d['y']))**2)
        if len(d['y']) > 1 else np.nan
        for v, d in R2_data.items()
    }
    MSE = total_err/obs if obs else np.nan

    # Aggregate all y and y_pred across variables for total R2
    all_y, all_yp = [], []
    for v_vals in var_measurements.values():
        for y, y_pred, _ in v_vals:
            all_y.append(y)
            all_yp.append(y_pred)

    if len(all_y) > 1:
        y_arr = np.array(all_y)
        yp_arr = np.array(all_yp)
        R2_total = 1 - np.sum((y_arr - yp_arr) ** 2) / np.sum((y_arr - np.mean(y_arr)) ** 2)
    else:
        R2_total = np.nan

    return (
        data, LS, MLE, Chi,
        LSA, Z, V, CI, t_values,
        t_m, tv_input_m, ti_input_m,
        tv_output_m, ti_output_m,
        WLS, obs, MSE, R2_responses, R2_total, M
    )



def _report(result, mutation, models, logging_flag):
    r"""
    Summarize and optionally log uncertainty analysis results.

    This function processes the uncertainty analysis results from _uncert_metrics(),
    computes model weights based on Chi-squared values, and optionally logs parameter
    estimates, t-values, and confidence intervals. It prepares data structures for
    returning to the user or updating the models dictionary.

    Parameters
    ----------
    result : dict[str, dict]
        Uncertainty analysis results for each model/solver. Each dict contains:
            - 'optimization_result' : dict
                Original parameter estimation result with 'scpr', 'x', etc.
            - 'Chi' : float
                Chi-squared statistic for the model.
            - 'V_matrix' : np.ndarray
                Variance-covariance matrix.
            - 'CI' : np.ndarray
                Confidence interval half-widths.
            - 't_values' : np.ndarray
                t-statistics for parameters.
    mutation : dict[str, list[bool]]
        Parameter activity masks for each solver.
    models : dict
        Model definitions (passed for potential future use, currently unused).
    logging_flag : bool
        If True, log parameter estimates and t-values at INFO level.

    Returns
    -------
    scaled_params : dict[str, list[float]]
        Final parameter estimates for each solver (from 'scpr').
    solver_parameters : dict[str, np.ndarray]
        Normalized parameter vectors for each solver (from 'x').
    solver_cov_matrices : dict[str, np.ndarray]
        Variance-covariance matrices for each solver.
    solver_confidence_intervals : dict[str, dict[int, float]]
        Confidence intervals indexed by parameter number for each solver.
    result : dict[str, dict]
        Updated result dictionary with added 'P' field (model weights in percent).

    Notes
    -----
    **Model Weights (P)**:
    Relative probabilities are computed based on Chi-squared goodness-of-fit:
        \[
        P_k = \frac{w_k}{\sum_m w_m} \times 100\%, \quad w_k = \frac{1}{\chi_k^2}
        \]
    Lower Chi-squared indicates better fit → higher weight. This is a simplified
    model comparison metric (not rigorous AIC/BIC, but useful for quick assessment).

    **Logging Output**:
    When logging_flag=True, the function logs:
        - Estimated parameters (θ) for each solver
        - t-statistics for parameter significance

    This provides a quick summary of which parameters are well-determined (high |t|)
    vs. poorly determined (low |t|).

    **Return Structure**:
    The function returns five objects:
        1. scaled_params: Final estimates in original scale (for updating models['theta'])
        2. solver_parameters: Normalized estimates (diagnostic use)
        3. solver_cov_matrices: Covariance matrices (for updating models['V_matrix'])
        4. solver_confidence_intervals: CI by parameter index (plotting/reporting)
        5. result: Enriched result dict with model weights

    See Also
    --------
    uncert : Main entry point that calls this function.
    _uncert_metrics : Computes the metrics that are reported here.

    Examples
    --------
    >>> # Typically called internally by uncert()
    >>> scaled_p, norm_p, cov_mats, cis, enriched = _report(
    ...     result=uncert_results, mutation=models['mutation'],
    ...     models=models, logging_flag=True
    ... )
    >>> print(scaled_p['M1'])  # Final parameter estimates
    >>> print(enriched['M1']['P'])  # Model weight (%)
    """
    scaled_params = {}
    solver_parameters = {}
    solver_cov_matrices = {}
    solver_confidence_intervals = {}
    # P-test: P_i = (1/chi2_i) / sum_j(1/chi2_j) * 100, chi2 = 'Chi'; chi2 = 0 -> 100 %
    chi = {s: float(res['Chi']) for s, res in result.items()}
    zero = [s for s, c in chi.items() if c <= 0]
    if zero:
        weights = {s: (1.0 if s in zero else 0.0) for s in chi}
    else:
        weights = {s: 1.0 / c for s, c in chi.items()}
    sum_of_chi_squared = sum(weights.values())
    for solver, res in result.items():
        opt = res['optimization_result']
        scpr = opt['scpr']
        scaled_params[solver] = scpr.tolist()
        solver_parameters[solver] = opt['x']
        solver_cov_matrices[solver] = res['V_matrix']
        solver_confidence_intervals[solver] = {
            i: ci for i, ci in enumerate(res['CI'])
        }
        res['P'] = (weights[solver] / sum_of_chi_squared) * 100
        if logging_flag:
            logger.info(f"Solver {solver}: Estimated θ = {scpr}")
            logger.info(f"Solver {solver}: t-val = {res['t_values']}")
    return (
        scaled_params,
        solver_parameters,
        solver_cov_matrices,
        solver_confidence_intervals,
        result
    )


def _fdm_mesh_independency(
    theta, thetac, solver, system, models,
    tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars,
    data, sens_method, varcov, resultpr
):
    r"""
    Determine optimal finite-difference step size via mesh-independency test.

    This function performs a systematic sweep of epsilon values (finite-difference
    step sizes) to identify the value that produces the most stable and converged
    covariance matrix. It evaluates eigenvalues of the covariance matrix across
    a logarithmic range of epsilons and selects the value that minimizes local
    variance, indicating numerical convergence.

    Parameters
    ----------
    theta : np.ndarray
        Normalized parameter vector (typically np.ones_like(thetac)).
    thetac : np.ndarray
        Parameter values at which to evaluate sensitivities.
    solver : str
        Model/solver name.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    tv_iphi_vars : list[str]
        Time-variant input variable names.
    ti_iphi_vars : list[str]
        Time-invariant input variable names.
    tv_ophi_vars : list[str]
        Time-variant output variable names.
    ti_ophi_vars : list[str]
        Time-invariant output variable names.
    data : dict
        Experimental data sheets.
    sens_method : str
        Sensitivity method ('forward' or 'central').
    varcov : str
        Covariance method (forced to 'M' for this test).
    resultpr : dict
        Parameter estimation results (passed to _uncert_metrics).

    Returns
    -------
    best_eps : float
        Optimal epsilon value that minimizes eigenvalue variance.

    Notes
    -----
    **Algorithm**:
        1. Generate 50 epsilon values logarithmically spaced in \( [10^{-12}, 10^{-1}] \).
        2. For each epsilon:
            - Call _uncert_metrics() to compute covariance matrix V.
            - Extract eigenvalues of V.
        3. For each epsilon, compute variance of eigenvalues in a sliding window (width=5).
        4. Select epsilon with minimum total eigenvalue variance (most stable region).

    **Why This Works**:
        - **Too small ε**: Finite differences suffer from catastrophic cancellation
          (numerical noise dominates) → eigenvalues fluctuate wildly.
        - **Too large ε**: Finite differences incur truncation error (nonlinearity)
          → eigenvalues vary systematically.
        - **Optimal ε**: In between these extremes, eigenvalues plateau (mesh-independent).

    **Heuristic Selection**:
    The sliding window approach finds the epsilon where eigenvalues are least sensitive
    to perturbations, indicating convergence. Window size of 5 balances noise averaging
    with resolution.

    **Output Files**:
    Results are saved in './meshindep_plots/' directory:
        - '{solver}_eps_test.png': Plot of eigenvalues vs epsilon
        - '{solver}_eps_test.xlsx': Tabulated eigenvalues for all tested epsilons

    **Forced Covariance Method**:
    This test always uses varcov='M' (measurement-based) regardless of input, because
    it requires repeated covariance computation with varying epsilon. Hessian ('H') or
    bootstrap ('B') methods do not depend on epsilon in the same way.

    **Typical Results**:
    For well-conditioned problems, optimal epsilon is usually \( 10^{-6} \) to \( 10^{-4} \).
    For ill-conditioned problems, you may see a narrower plateau or no clear optimum
    (indicating fundamental identifiability issues).

    References
    ----------
    .. [1] Gill, P. E., Murray, W., & Wright, M. H. (1981).
       *Practical Optimization*. Academic Press.

    .. [2] Dennis, J. E., & Schnabel, R. B. (1996).
       *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*.
       SIAM.

    See Also
    --------
    uncert : Main entry point that calls this when eps is not provided.
    _uncert_metrics : Core routine called repeatedly during the sweep.

    Examples
    --------
    >>> # Typically called internally by uncert() when eps is None
    >>> optimal_eps = _fdm_mesh_independency(
    ...     theta=np.ones(3), thetac=theta_opt, solver='M1',
    ...     system=system, models=models,
    ...     tv_iphi_vars=tv_vars, ti_iphi_vars=ti_vars,
    ...     tv_ophi_vars=outputs, ti_ophi_vars=[],
    ...     data=data, sens_method='central', varcov='M', resultpr=results
    ... )
    >>> print(f"Selected epsilon: {optimal_eps:.1e}")
    """
    logger.info(f"Performing mesh-independency test for {solver}")
    eps_values = np.logspace(-12, -1, 50)
    eigs = []
    varcov = 'M'
    for eps in eps_values:
        try:
            out = _uncert_metrics(
                theta, data, [solver], thetac, {solver:eps}, [True]*len(theta),
                ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
                system, models, varcov, resultpr, sens_method=sens_method
            )
            V = out[6]  # returned V_matrix
            eigs.append(np.linalg.eigvalsh(V))
        except Exception:
            eigs.append([np.nan]*len(theta))
    eigs = np.array(eigs)

    # Plot and save
    folder = Path.cwd() / "meshindep_plots"
    folder.mkdir(exist_ok=True)
    base = folder / f"{solver}_eps_test"
    plt.figure(figsize=(8,5))
    for i in range(eigs.shape[1]):
        plt.plot(eps_values, eigs[:,i], marker='o', label=f'eig{i+1}')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Epsilon'); plt.ylabel('Eigenvalues')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{base}.png"); plt.close()
    pd.DataFrame(eigs, index=eps_values).to_excel(f"{base}.xlsx")

    # Choose epsilon minimising the variation of the eigenvalues over a window of 5 steps.
    # The variation is measured on log10(eigenvalue): relative, so every eigenvalue counts
    # alike whatever the parameter units (on a linear scale the largest variance, i.e. the
    # sloppiest direction, would decide alone). Windows with a non-finite or non-positive
    # eigenvalue are skipped.
    # Eigenvalues that are 0 for most steps (directions without sensitivity) are left out.
    with np.errstate(divide='ignore', invalid='ignore'):
        logeig = np.where(eigs > 0, np.log10(np.where(eigs > 0, eigs, 1.0)), np.nan)
    cols = np.where(np.mean(np.isfinite(logeig), axis=0) >= 0.8)[0]
    win = 5
    half = win//2
    best, best_var = eps_values[0], np.inf
    for i in range(half, len(eps_values) - half):
        if cols.size:
            block = logeig[i-half:i+half+1][:, cols]
        else:   # no usable log scale: linear variation (previous rule)
            block = eigs[i-half:i+half+1]
        if not np.isfinite(block).all():
            continue
        var = block.std(axis=0).sum()
        if var < best_var:
            best_var, best = var, eps_values[i]
    logger.info(f"Selected epsilon={best:.1e} for {solver}")
    return best
