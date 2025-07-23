# import numpy as np
# from middoe.krnl_simula import simula
# import scipy.stats as stats
# import logging
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path
#
#
# # Configure the logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# def uncert(data, resultpr, system, models, iden_opt, case=None):
#     """
#     Perform uncertainty analysis on the optimization results.
#
#     Parameters
#     ----------
#     data : dict
#         Dictionary containing the experimental data.
#     resultpr : dict
#         Dictionary containing the results of the optimization from parameter estimation.
#     system : dict
#         User-provided dictionary containing the model structure.
#     models : dict
#         User-provided dictionary containing the modelling settings.
#     iden_opt : dict
#         User-provided dictionary containing the estimation settings.
#     run_solver : function
#         Simulator function to run the model.
#
#     Returns
#     ----------
#     tuple
#         (resultun2, theta_parameters, solver_parameters, scaled_params, observed_values)
#
#         Where resultun2 is a dictionary keyed by model name, containing:
#           'LS', 'MLE', 'MSE', 'Chi', 'LSA', 'JWLS', 'R2_responses', etc.
#     """
#
#
#     # Gather variable names
#     sens_method = iden_opt.get('sens_m', 'central')  # Default to central difference
#     tv_iphi_vars = list(system.get('tvi', {}).keys())
#     ti_iphi_vars = list(system.get('tii', {}).keys())
#     tv_ophi_vars = [
#         var for var in system.get('tvo', {}).keys()
#         if system['tvo'][var].get('meas', True)
#     ]
#     ti_ophi_vars = [
#         var for var in system.get('tio', {}).keys()
#         if system['tio'][var].get('meas', True)
#     ]
#     varcov= iden_opt.get('var-cov', 'H')
#     # Attempt to retrieve standard deviations for each measured variable.
#     # If not found or NaN, default to 1.0.
#     # (Although we don't use them directly here, we show how you'd gather them.)
#     std_dev = {}
#     for var in system.get('tvo', {}):
#         if system['tvo'][var].get('meas', True):
#             sigma = system['tvo'][var].get('unc', None)
#             if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
#                 sigma = 1.0
#             std_dev[var] = sigma
#     for var in system.get('tio', {}):
#         if system['tio'][var].get('meas', True):
#             sigma = system['tio'][var].get('unc', None)
#             if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
#                 sigma = 1.0
#             std_dev[var] = sigma
#
#     # Estimation settings
#     eps = iden_opt.get('eps', None)
#     if isinstance(eps, dict):
#         epsf = eps.copy()  # user provided dict
#     else:
#         epsf = {solver: eps for solver in resultpr}  # use same float for all
#     logging = iden_opt.get('log', False)
#
#     # Modelling settings
#     mutation = models.get('mutation', {})
#
#     resultun = {}
#
#     # We'll keep a placeholder to store the last-run 'observed_values'
#     # so that we can return it at the end.
#     observed_values = None
#
#     # Loop over solvers in the results dictionary
#     for solver, solver_results in resultpr.items():
#
#         thetac = solver_results['scpr']
#         thetas = mutation.get(solver, [])
#         # initial_x0 = solver_results['optimization_result'].x
#         theta = np.ones_like(thetac, dtype=float)
#
#         # If eps is not provided, perform (optional) FDM mesh dependency test
#         # in your own custom function:
#         if epsf[solver] is None:
#             epsf[solver] = _fdm_mesh_independency(theta, thetac, solver, system, models, tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars, data, sens_method, varcov, resultpr)
#
#
#         # ---------------------------------------------------------------------
#         # Call the core metrics function (updated to handle Weighted LS, MLE, etc.)
#         # ---------------------------------------------------------------------
#         (
#             optimized_data,
#             LS,
#             MLE,
#             Chi,
#             LSA,
#             Z,
#             V_matrix,
#             CI,
#             t_values,
#             t_m,
#             tv_input_m,
#             ti_input_m,
#             tv_output_m,
#             ti_output_m,
#             WLS,
#             obs,
#             MSE,
#             R2_responses,
#             M
#         ) = _uncert_metrics(
#             theta,
#             data,
#             [solver],
#             thetac,
#             epsf,
#             thetas,
#             ti_iphi_vars,
#             tv_iphi_vars,
#             tv_ophi_vars,
#             ti_ophi_vars,
#             system,
#             models,
#             varcov,
#             resultpr,
#             sens_method=sens_method
#         )
#         observed_values = obs  # store how many measurements used, etc.
#
#         resultun[solver] = {
#             'optimization_result': solver_results,
#             'data': optimized_data,
#             'LS': LS,                     # Overall R² across all scaled data
#             'MLE': MLE,                   # Weighted negative log-likelihood
#             'MSE': MSE,                   # Mean squared error (unweighted)
#             'Chi': Chi,                   # Sum((res^2) / abs(pred))
#             'LSA': LSA,                   # Jacobian (sensitivities)
#             'Z': Z,                       # Jacobian normalized by std deviations
#             'WLS': WLS,                   # Weighted least squares
#             'R2_responses': R2_responses, # dictionary of per-variable R²
#             'V_matrix': np.array(V_matrix),
#             'CI': CI,
#             't_values': t_values,
#             't_m': t_m,
#             'tv_input_m': tv_input_m,
#             'ti_input_m': ti_input_m,
#             'tv_output_m': tv_output_m,
#             'ti_output_m': ti_output_m,
#             'estimations': thetac,
#             'found_eps': epsf,
#             'M': M,
#         }
#
#     iden_opt['eps'] = epsf
#     # Summarize results for all solvers
#     (
#         scaled_params,
#         solver_parameters,
#         solver_cov_matrices,
#         solver_confidence_intervals,
#         resultun2
#     ) = _report(
#         resultun,
#         mutation,
#         models,
#         logging
#     )
#
#     if case == None:
#         for solver in models['can_m']:
#             models['theta'][solver] = scaled_params[solver]
#
#         # ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
#         for solver in models['can_m']:
#             models['V_matrix'][solver] = resultun[solver]['V_matrix']
#
#     # Construct return dictionary
#     uncert_output = {
#         'results': resultun2,
#         'obs': observed_values
#     }
#
#     return uncert_output
#
#
#
#
#
# def _uncert_metrics(
#     theta,
#     data,
#     active_solvers,
#     thetac,
#     eps,
#     thetas,
#     ti_iphi_vars,
#     tv_iphi_vars,
#     tv_ophi_vars,
#     ti_ophi_vars,
#     system,
#     models,
#     varcov,
#     resultpr,
#     sens_method='forward'
# ):
#     """
#     Calculate uncertainty metrics for parameter estimation results.
#     Uses pointwise heteroscedastic uncertainties from 'MES_E:{var}' arrays.
#
#     Returns:
#       data, LS, sum_mle, sum_chi,
#       LSA (sensitivity matrix), Z (Z-scores),
#       V (covariance matrix), CI (confidence intervals), t_values,
#       t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m,
#       sum_wls, obs_count, MSE, R2
#     """
#
#
#     theta_full = np.array(theta)
#     active_idx = np.where(thetas)[0]
#     n_active = len(active_idx)
#     thetac = np.array(thetac)
#
#     var_measurements = {}
#     Q_accum = {v: [] for v in tv_ophi_vars + ti_ophi_vars}
#     t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m = {}, {}, {}, {}, {}
#
#     if not active_solvers:
#         raise ValueError("No active solver provided.")
#     solver = active_solvers[0]
#     h = eps[solver]
#
#     for sheet, sheet_data in data.items():
#         t_all = np.unique(np.array(sheet_data.get('X:all', []))[~np.isnan(sheet_data.get('X:all', []))])
#         swps = {}
#         for v in tv_iphi_vars:
#             tkey, lkey = f"{v}t", f"{v}l"
#             if tkey in sheet_data and lkey in sheet_data:
#                 ta = np.array(sheet_data[tkey])[~np.isnan(sheet_data[tkey])]
#                 la = np.array(sheet_data[lkey])[~np.isnan(sheet_data[lkey])]
#                 if ta.size and la.size:
#                     swps[tkey], swps[lkey] = ta, la
#         ti_data = {v: float(sheet_data.get(v, [np.nan])[0]) for v in ti_iphi_vars}
#         tv_data = {v: np.array(sheet_data.get(v, []))[~np.isnan(sheet_data.get(v, []))] for v in tv_iphi_vars}
#         cvp = {v: sheet_data[f"CVP:{v}"].iloc[0] for v in system.get('tvi', {}) if f"CVP:{v}" in sheet_data}
#
#         tv_pred, ti_pred, _ = simula(
#             t_all, swps, ti_data,
#             {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#             theta_full, thetac, cvp, tv_data, solver, system, models
#         )
#
#         t_m[sheet] = t_all.tolist()
#         tv_input_m[sheet] = tv_data
#         ti_input_m[sheet] = ti_data
#         tv_output_m[sheet] = tv_pred
#         ti_output_m[sheet] = ti_pred
#
#         perturbed = {}
#         for p in active_idx:
#             if sens_method == 'forward':
#                 th = theta_full.copy(); th[p] += h
#                 tv2, ti2, _ = simula(t_all, swps, ti_data, {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                                      th, thetac, cvp, tv_data, solver, system, models)
#                 perturbed[p] = {'plus':(tv2, ti2)}
#             elif sens_method == 'central':
#                 th_p = theta_full.copy(); th_p[p] += h
#                 th_m = theta_full.copy(); th_m[p] -= h
#                 tv_p, ti_p, _ = simula(t_all, swps, ti_data, {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                                        th_p, thetac, cvp, tv_data, solver, system, models)
#                 tv_m, ti_m, _ = simula(t_all, swps, ti_data, {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
#                                        th_m, thetac, cvp, tv_data, solver, system, models)
#                 perturbed[p] = {'plus':(tv_p, ti_p), 'minus':(tv_m, ti_m)}
#             elif sens_method in ('five','five_point'):
#                 ths = {}
#                 for k, factor in zip(('p1','m1','p2','m2'), (1,-1,2,-2)):
#                     tsh = theta_full.copy(); tsh[p] += factor*h
#                     ths[k] = tsh
#                 sims = {k: simula(t_all, swps, ti_data, {v:1 for v in ti_iphi_vars},{v:1 for v in tv_iphi_vars},1,
#                                   ths[k], thetac, cvp, tv_data, solver, system, models)[:2]
#                         for k in ths}
#                 perturbed[p] = sims
#             else:
#                 raise ValueError(f"Unknown sens_method '{sens_method}'")
#
#         for var in tv_ophi_vars + ti_ophi_vars:
#             if var in tv_ophi_vars:
#                 yt = np.array(sheet_data.get(f"MES_Y:{var}",[]))[~np.isnan(sheet_data.get(f"MES_Y:{var}",[]))]
#                 tx = np.array(sheet_data.get(f"MES_X:{var}",[]))[~np.isnan(sheet_data.get(f"MES_X:{var}",[]))]
#                 yp = np.array([tv_pred[var][np.where(t_all==t)[0][0]] for t in tx]) if tx.size else np.array([])
#                 sigs_raw = sheet_data.get(f"MES_E:{var}", [])
#                 sigs = np.array(sigs_raw)[~np.isnan(sigs_raw)]
#                 if sigs.size != yp.size:
#                     raise ValueError(f"Mismatch between MES_E and predictions for {var} in {sheet}")
#
#             else:
#                 yt = np.array([sheet_data.get(f"MES_Y:{var}",[np.nan])[0]])
#                 yp = np.array([ti_pred[var]])
#                 sigs = np.array([1.0])
#
#             sigs = np.maximum(sigs, 1e-6)
#             var_measurements.setdefault(var, []).extend(zip(yt, yp, sigs))
#             if yt.size == 0: continue
#
#             cols = []
#             for p in active_idx:
#                 if sens_method == 'forward':
#                     tv2, ti2 = perturbed[p]['plus']
#                     y2 = (np.array([tv2[var][np.where(t_all==t)[0][0]] for t in tx])
#                           if var in tv_ophi_vars else np.array([ti2[var]]))
#                     d = (y2 - yp) / (h * thetac[p])
#                 elif sens_method == 'central':
#                     tv_p, ti_p = perturbed[p]['plus']; tv_m, ti_m = perturbed[p]['minus']
#                     y_p = (np.array([tv_p[var][np.where(t_all==t)[0][0]] for t in tx])
#                            if var in tv_ophi_vars else np.array([ti_p[var]]))
#                     y_m = (np.array([tv_m[var][np.where(t_all==t)[0][0]] for t in tx])
#                            if var in tv_ophi_vars else np.array([ti_m[var]]))
#                     d = (y_p - y_m) / (2 * h * thetac[p])
#                 else:
#                     tv_p1, ti_p1 = perturbed[p]['p1']; tv_m1, ti_m1 = perturbed[p]['m1']
#                     tv_p2, ti_p2 = perturbed[p]['p2']; tv_m2, ti_m2 = perturbed[p]['m2']
#                     def fetch(sim): return (np.array([sim[var][np.where(t_all==t)[0][0]] for t in tx])
#                                             if var in tv_ophi_vars else np.array([sim[var]]))
#                     y_p1, y_m1, y_p2, y_m2 = [fetch(sim) for sim in (tv_p1, tv_m1, tv_p2, tv_m2)]
#                     d = (-y_p2 + 8*y_p1 - 8*y_m1 + y_m2) / (12 * h * thetac[p])
#                 cols.append(d)
#             Q_accum[var].append(np.stack(cols, axis=1))
#
#     Q_full = {v: np.vstack(Q_accum[v]) for v in Q_accum}
#     vars_list = list(Q_full)
#     obs = sum(len(v) for v in var_measurements.values())
#     dof = max(obs - n_active, 1)
#     tcrit = stats.t.ppf(0.975, dof)
#     LSA = np.vstack([Q_full[v] for v in vars_list])
#     sigs_all = np.concatenate([[sig for _, _, sig in var_measurements[v]] for v in vars_list])
#     sigs_all = np.clip(sigs_all, 1e-6, np.inf)
#
#     # --- Variance-covariance matrix selection ---
#     if varcov == 'M':
#         W_full = np.diag(1.0 / sigs_all ** 2)
#         M = LSA.T @ W_full @ LSA
#         V = np.linalg.pinv(M)
#         logger = logging.getLogger(__name__)
#         logger.info(f"Fisher matrix condition number: {np.linalg.cond(M):.2e}")
#         if np.linalg.cond(M) > 1e10:
#             logger.warning("Fisher matrix is ill-conditioned. Confidence intervals may be unreliable.")
#     elif varcov == 'H':
#         hess_inv = resultpr.get(solver, {}).get('hess_inv', None)
#         if hess_inv is None:
#             raise ValueError("Hessian inverse not found in resultpr for solver: " + solver)
#         Ve = np.array(hess_inv)
#         err = resultpr.get(solver, {}).get('fun', None)
#         err=err/dof
#         V= err * Ve  # scale Hessian inverse by error variance
#         M = np.linalg.pinv(V)  # if you want to return M as well
#     else:
#         raise ValueError(f"Unknown varcov option '{varcov}'")
#
#     CI = tcrit * np.sqrt(np.diag(V))
#     tvals = (theta_full[active_idx] * thetac[active_idx]) / (CI)
#     resp_sigs = np.concatenate([[sig for _, _, sig in var_measurements[v]] for v in vars_list])
#     theta_std = np.sqrt(np.diag(V))
#     Z = LSA * theta_std / resp_sigs[:, None]
#
#     LS = sum_wls = sum_mle = sum_chi = total_err = 0.0
#     R2_data = {v: {'y': [], 'yp': []} for v in vars_list}
#     for v, vals in var_measurements.items():
#         for y, y_pred, sig in vals:
#             r = y - y_pred
#             LS += r**2
#             sum_wls += (r / sig)**2
#             sum_mle += 0.5 * (np.log(2 * np.pi * sig**2) + (r / sig)**2)
#             sum_chi += r**2 / sig**2
#             R2_data[v]['y'].append(y)
#             R2_data[v]['yp'].append(y_pred)
#             total_err += r**2
#
#     R2 = {v: 1 - np.sum((np.array(d['y']) - np.array(d['yp']))**2) / np.sum((np.array(d['y']) - np.mean(d['y']))**2)
#           if len(d['y']) > 1 else np.nan
#           for v, d in R2_data.items()}
#     MSE = total_err / obs if obs else np.nan
#
#     return (
#         data, LS, sum_mle, sum_chi, LSA, Z, V, CI, tvals,
#         t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m,
#         sum_wls, obs, MSE, R2, M
#     )
#
#
#
#
#
#
# def _report(result, mutation, models, logging):
#     """
#     Report the uncertainty and parameter estimation results.
#
#     Parameters:
#     ----------
#     result : dict
#         Dictionary containing the results of the optimization.
#     mutation : dict
#         Dictionary indicating which parameters are to be optimized for each model.
#     theta_parameters : dict
#         Dictionary of theta parameters for each model.
#     models : dict
#         User-provided dictionary containing the modelling settings.
#     logging : bool
#         Flag to indicate whether to log the results.
#
#     Returns:
#     ----------
#     tuple
#         (scaled_params, solver_parameters, solver_cov_matrices,
#          solver_confidence_intervals, result)
#     """
#     solver_parameters = {}
#     solver_cov_matrices = {}
#     solver_confidence_intervals = {}
#     scaled_thetaest = {}
#     scaled_params = {}
#     sum_of_chi_squared = sum(1 / (res['Chi']) ** 2 for res in result.values())
#
#     for solver, solver_results in result.items():
#         # Extract mutation mask and positions of active parameters
#         mutation_mask = mutation.get(solver, [True] * len(solver_results['optimization_result'].x))
#         original_positions = [i for i, active in enumerate(mutation_mask) if active]
#
#         # Scale estimated parameters
#         scaled_thetaest[solver] = solver_results['optimization_result']['scpr']
#
#         if logging:
#             scaled_params[solver] = [float(param) for param in scaled_thetaest[solver]]
#             print(f"Estimated parameters of {solver}: {scaled_params[solver]}")
#             print(f"LS objective function value for {solver}: {solver_results['LS']}")
#
#         # Update modelling settings with V_matrix for the model
#         models['V_matrix'][solver] = solver_results['V_matrix']
#
#         # Map CI, t-values, and CI ratios to their original parameter positions
#         CI_mapped = {pos: solver_results['CI'][i] for i, pos in enumerate(original_positions)}
#         t_values_mapped = {pos: solver_results['t_values'][i] for i, pos in enumerate(original_positions)}
#         CI_ratio_mapped = {
#             pos: ci * 100 / param if param != 0 else float('inf')
#             for pos, ci, param in zip(original_positions, solver_results['CI'], solver_results['optimization_result'].x)
#         }
#
#         # Log t-values if requested
#         if logging:
#             print(f"T-values of model {solver}: {solver_results['t_values']}")
#
#         # Store parameters, covariance matrices, and confidence intervals
#         solver_parameters[solver] = solver_results['optimization_result'].x
#         solver_cov_matrices[solver] = solver_results['V_matrix']
#         solver_confidence_intervals[solver] = CI_mapped
#         solver_results['P'] = ((1 / (solver_results['Chi']) ** 2) / sum_of_chi_squared) * 100
#         print(f"P-value of model:{solver} is {solver_results['P']} for model discrimination")
#
#         # Log R² for responses
#         if 'R2_responses' in solver_results:
#             r2_responses = solver_results['R2_responses']
#             if logging:
#                 print(f"R2 values for responses in model {solver}:")
#                 for var, r2_value in r2_responses.items():
#                     print(f"  {var}: {r2_value:.4f}")
#
#             # Optionally store R² values in result for external use
#             solver_results['R2_responses_summary'] = {
#                 var: round(r2_value, 4) for var, r2_value in r2_responses.items()
#             }
#
#     return scaled_params, solver_parameters, solver_cov_matrices, solver_confidence_intervals, result
#
# def _fdm_mesh_independency(theta, thetac, solver, system, models,
#                            tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars, data, sens_method, varcov, resultpr):
#     """
#     Perform FDM mesh dependency test to determine a suitable epsilon for sensitivity analysis.
#
#     Parameters:
#     theta (list): Initial parameter estimates.
#     thetac (list): Scaling factors for parameters.
#     solver (str): Solver name.
#     system (dict): Model structure.
#     models (dict): Modelling settings.
#     tv_iphi_vars (list): Time-variant input variable names.
#     ti_iphi_vars (list): Time-invariant input variable names.
#     tv_ophi_vars (list): Time-variant output variable names.
#     ti_ophi_vars (list): Time-invariant output variable names.
#     data (dict): Experimental data.
#
#     Returns:
#     float: Optimal epsilon value for sensitivity analysis.
#     """
#     logger.info(f"Performing FDM mesh dependency test for model {solver}.")
#
#     eps_values = np.logspace(-12, -1, 50)
#     eig_matrix = []
#
#     for eps in eps_values:
#         try:
#             result = _uncert_metrics(
#                 theta, data, [solver], thetac, {solver: eps}, [True] * len(theta),
#                 ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars, system, models, varcov,resultpr, sens_method=sens_method
#             )
#             V_matrix = result[6]  # variance-covariance matrix
#             eigvals = np.linalg.eigvalsh(V_matrix)
#             eig_matrix.append(eigvals)
#         except np.linalg.LinAlgError:
#             logger.warning(f"LinAlgError at eps={eps:.1e} for {solver}")
#             eig_matrix.append([np.nan] * len(theta))
#
#     eig_matrix = np.array(eig_matrix)  # shape = (eps values, parameters)
#
#     # Plot eigenvalues across epsilons
#     folder = Path.cwd() / "meshindep_plots"
#     folder.mkdir(exist_ok=True)
#     filename_base = folder / f"{solver}_eps_eigenvalue_test"
#
#     plt.figure(figsize=(10, 6))
#     for i in range(eig_matrix.shape[1]):
#         plt.plot(eps_values, eig_matrix[:, i], marker='o', label=f'Eigenvalue {i+1}')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Epsilon')
#     plt.ylabel('Eigenvalues of V')
#     plt.title(f'Eigenvalue Spectrum vs Epsilon (Model: {solver})')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.savefig(f"{filename_base}.png", dpi=300)
#     if plt.get_backend() != 'agg':
#         plt.show()
#     plt.close()
#
#     # Save eigenvalues to Excel
#     df = pd.DataFrame(eig_matrix, columns=[f'Eigen_{i+1}' for i in range(eig_matrix.shape[1])])
#     df.insert(0, 'Epsilon', eps_values)
#     df.to_excel(f"{filename_base}.xlsx", index=False)
#
#     # Select epsilon based on region of minimal total eigenvalue variation
#     stable_region = np.where(np.isfinite(eig_matrix).all(axis=1))[0]
#     if stable_region.size == 0:
#         raise ValueError(f"Could not determine a stable epsilon region for model {solver}.")
#
#     window = 5  # must be odd
#     half_w = window // 2
#     min_total_var = np.inf
#     best_center_idx = None
#
#     for i in range(half_w, len(stable_region) - half_w):
#         eps_candidate = eps_values[stable_region[i]]
#         if not (1e-7 <= eps_candidate <= 1e-2):
#             continue  # skip out-of-range eps
#
#         window_indices = stable_region[i - half_w:i + half_w + 1]
#         window_eigs = eig_matrix[window_indices]
#         variation = np.sum(np.std(window_eigs, axis=0))
#         if variation < min_total_var:
#             min_total_var = variation
#             best_center_idx = i
#
#     # If no preferred-range candidate found, fall back to full region
#     if best_center_idx is None:
#         for i in range(half_w, len(stable_region) - half_w):
#             window_indices = stable_region[i - half_w:i + half_w + 1]
#             window_eigs = eig_matrix[window_indices]
#             variation = np.sum(np.std(window_eigs, axis=0))
#             if variation < min_total_var:
#                 min_total_var = variation
#                 best_center_idx = i
#
#     optimal_eps = eps_values[stable_region[best_center_idx]]
#     logger.info(f"Optimal epsilon selected for model {solver}: {optimal_eps:.2e}")
#
#     return optimal_eps



import numpy as np
import scipy.stats as stats
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from middoe.krnl_simula import simula

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def uncert(data, resultpr, system, models, iden_opt, case=None):
    """
    Perform uncertainty analysis on the optimization results.

    If iden_opt['varcov']=='B', will use the bootstrap variance–covariance
    stored in resultpr[solver]['varcov'].
    """
    # Sensitivity method and varcov selection
    sens_method = iden_opt.get('sens_m', 'central')
    varcov_key  = iden_opt.get('var-cov', 'H')  # 'M', 'H', or 'B'

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
            R2_responses, M
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
            sens_method=sens_method
        )
        observed_values = obs

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
            'M': M
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
        for sv in models['can_m']:
            models['theta'][sv]    = scaled_params[sv]
            models['V_matrix'][sv] = resultun[sv]['V_matrix']

    return {'results': resultun2, 'obs': observed_values}


# def _uncert_metrics(
#     theta, data, active_solvers, thetac, eps, thetas,
#     ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
#     system, models, varcov, resultpr, sens_method='forward'
# ):
#     """
#     Calculate detailed uncertainty metrics.  Honors varcov=='B' to pull
#     bootstrap covariance from resultpr.
#     """
#     theta_full = np.array(theta, dtype=float)
#     active_idx = np.where(thetas)[0]
#     n_active   = len(active_idx)
#     thetac_arr = np.array(thetac, dtype=float)
#
#     # Containers
#     var_measurements = {}
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
#         # Switch‐pressure inputs
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
#             # Extract y_true, y_pred, sigma
#             if var in tv_ophi_vars:
#                 # time‐varying
#                 xcol = f"MES_X:{var}"
#                 ycol = f"MES_Y:{var}"
#                 ecol = f"MES_E:{var}"
#                 mask = ~sheet[xcol].isna().values
#                 times = sheet[xcol][mask].values
#                 y_t = sheet[ycol][mask].values
#                 idx = np.isin(t_all, times)
#                 y_p = np.array(tv_ref[var])[idx]
#                 if ecol in sheet:
#                     y_e = sheet[ecol][mask].values
#                 else:
#                     y_e = np.full_like(y_t, 1.0)
#             else:
#                 # time‐independent
#                 ycol = f"MES_Y:{var}"
#                 y_t = np.array([sheet[ycol].iloc[0]])
#                 y_p = np.array([ti_ref[var]])
#                 y_e = np.array([1.0])
#
#             var_measurements.setdefault(var, []).extend(zip(y_t, y_p, y_e))
#
#             # Build sensitivity row
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
#         # Fisher-based
#         sigs = np.concatenate([np.array([s for _,_,s in var_measurements[v]]) for v in Q_accum])
#         W = np.diag(1.0/sigs**2)
#         M = LSA.T @ W @ LSA
#         V = np.linalg.inv(M)
#
#     elif varcov == 'H':
#         # Hessian-inverse
#         hess_inv = resultpr[solver].get('hess_inv')
#         err = resultpr[solver].get('fun', 1.0) / dof
#         V = err * np.array(hess_inv)
#         M = np.linalg.pinv(V)
#
#     elif varcov == 'B':
#         # **Bootstrap**-based
#         V = resultpr[solver].get('v')
#         if V is None:
#             raise ValueError(f"No bootstrap v matrix found for model '{solver}'")
#         M = np.linalg.pinv(V)
#         logger.info(f"Using bootstrap var–cov for {solver}")
#
#     else:
#         raise ValueError(f"Unknown varcov option '{varcov}'")
#
#     # Confidence intervals and t-values
#     CI = tcrit * np.sqrt(np.diag(V))
#     theta_std = np.sqrt(np.diag(V))
#     t_values = (theta_full[active_idx] * thetac_arr[active_idx]) / CI
#
#     # Z-scores (normalized sensitivities)
#     resp_sigs = np.concatenate([np.array([s for _,_,s in var_measurements[v]]) for v in Q_accum])
#     Z = LSA * theta_std / resp_sigs[:, None]
#
#     # Compute LS, WLS, MLE, Chi, MSE, R2
#     LS = WLS = MLE = Chi = 0.0
#     total_err = 0.0
#     R2_data = {v: {'y':[], 'yp':[]} for v in Q_accum}
#     for v, vals in var_measurements.items():
#         for y, y_pred, s in vals:
#             r = y - y_pred
#             LS += r**2
#             WLS += (r/s)**2
#             MLE += 0.5*(np.log(2*np.pi*s**2)+(r/s)**2)
#             Chi += r**2/s**2
#             total_err += r**2
#             R2_data[v]['y'].append(y)
#             R2_data[v]['yp'].append(y_pred)
#     R2_responses = {
#         v: 1 - np.sum((np.array(d['y'])-np.array(d['yp']))**2)/np.sum((np.array(d['y'])-np.mean(d['y']))**2)
#         if len(d['y'])>1 else np.nan
#         for v,d in R2_data.items()
#     }
#     MSE = total_err/obs if obs else np.nan
#
#     return (
#         data, LS, MLE, Chi,
#         LSA, Z, V, CI, t_values,
#         t_m, tv_input_m, ti_input_m,
#         tv_output_m, ti_output_m,
#         WLS, obs, MSE, R2_responses, M
#     )

import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def _uncert_metrics(
    theta, data, active_solvers, thetac, eps, thetas,
    ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
    system, models, varcov, resultpr, sens_method='forward'
):
    """
    Calculate detailed uncertainty metrics. Honors varcov=='B' to pull
    bootstrap covariance from resultpr.

    Assumes heteroskedastic (relative) measurement errors stored in MES_E:*.
    Each MES_E value is interpreted as a *fractional* standard deviation and
    is converted to an absolute sigma by multiplying with the signal magnitude
    (|y_pred|). A tiny floor avoids zero weights.

    Adds ill-conditioning diagnostics (rank/condition numbers/eigenvalues)
    without changing the original signature or return tuple. The diagnostic
    summary is logged (INFO to avoid colored 'red' outputs) and stored in
    resultpr[solver]['ill_report'] when possible.
    """
    # ---- diagnostic thresholds (adjust here if needed) -----------------
    _SLOPPY_THRESH = 1e3     # 'sloppy' if cond(M) >= this
    _COND_THRESH   = 1e8     # ill-conditioned if cond(M) >= this
    _TINY_EIG      = 1e-12   # ill if min eigenvalue < this
    _REPORT_LEVEL  = logging.INFO  # keep it white in most color schemes
    _SIG_FLOOR     = 1e-12   # floor for absolute sigmas

    theta_full = np.array(theta, dtype=float)
    active_idx = np.where(thetas)[0]
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
                mask = ~sheet[xcol].isna().values
                times = sheet[xcol][mask].values
                y_t   = sheet[ycol][mask].values
                idx   = np.isin(t_all, times)
                y_p   = np.array(tv_ref[var])[idx]

                # relative sigma from sheet, convert to absolute using |y_p|
                if ecol in sheet:
                    rel = sheet[ecol][mask].values.astype(float)
                    y_e = rel * np.abs(y_p) + _SIG_FLOOR
                else:
                    y_e = np.full_like(y_t, 1.0, dtype=float)
            else:
                # time-independent measurement
                ycol = f"MES_Y:{var}"
                ecol = f"MES_E:{var}"
                y_t  = np.array([sheet[ycol].iloc[0]])
                y_p  = np.array([ti_ref[var]])
                if ecol in sheet:
                    rel = float(sheet[ecol].iloc[0])
                    y_e = np.array([rel * np.abs(y_p[0]) + _SIG_FLOOR], dtype=float)
                else:
                    y_e = np.array([1.0], dtype=float)

            var_measurements.setdefault(var, []).extend(zip(y_t, y_p, y_e))

            # Sensitivity block
            cols = []
            for p in active_idx:
                mode, sims = perturbed[p]
                if mode == 'forward':
                    tvp, tip = sims
                    if var in tv_ophi_vars:
                        y2 = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
                    else:
                        y2 = np.array([tip[var]])
                    d = (y2 - y_p) / (h * thetac_arr[p])
                else:
                    tvp, tip, tvm, tim = sims
                    if var in tv_ophi_vars:
                        y_p_p = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
                        y_p_m = np.array([tvm[var][np.where(t_all==t)[0][0]] for t in times])
                    else:
                        y_p_p = np.array([tip[var]])
                        y_p_m = np.array([tim[var]])
                    d = (y_p_p - y_p_m) / (2*h*thetac_arr[p])
                cols.append(d)
            if len(cols):
                Q_accum[var].append(np.stack(cols, axis=1))

    # Flatten Q into LSA
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
        err = resultpr[solver].get('fun', 1.0) / dof
        V = err * np.array(hess_inv)
        M = np.linalg.pinv(V)

    elif varcov == 'B':
        V = resultpr[solver].get('v')
        if V is None:
            raise ValueError(f"No bootstrap v matrix found for model '{solver}'")
        M = np.linalg.pinv(V)
        logger.info(f"Using bootstrap var–cov for {solver}")

    else:
        raise ValueError(f"Unknown varcov option '{varcov}'")

    # --- Ill-conditioning diagnostics ----------------------------------
    ill_report = {}
    try:
        s = np.linalg.svd(LSA, compute_uv=False)
        ill_report['Q_singular_values'] = s
        ill_report['Q_rank'] = int(np.sum(s > s[0] * np.finfo(float).eps))
        ill_report['Q_cond'] = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    except Exception as e:
        ill_report['Q_error'] = str(e)

    try:
        eig = np.linalg.eigvalsh(M)
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
        f"Rank(Q)={ill_report.get('Q_rank','?')}/{n_active}, "
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
    CI = tcrit * np.sqrt(np.diag(V))
    theta_std = np.sqrt(np.diag(V))
    t_values = (theta_full[active_idx] * thetac_arr[active_idx]) / CI

    # Z-scores (normalized sensitivities)
    resp_sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]]) for v in Q_accum])
    Z = LSA * theta_std / resp_sigs[:, None]

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
            Chi += r**2/s**2
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

    return (
        data, LS, MLE, Chi,
        LSA, Z, V, CI, t_values,
        t_m, tv_input_m, ti_input_m,
        tv_output_m, ti_output_m,
        WLS, obs, MSE, R2_responses, M
    )



def _report(result, mutation, models, logging_flag):
    """
    Summarize and optionally log the uncertainty results.
    """
    scaled_params = {}
    solver_parameters = {}
    solver_cov_matrices = {}
    solver_confidence_intervals = {}

    for solver, res in result.items():
        opt = res['optimization_result']
        scpr = opt['scpr']
        scaled_params[solver] = scpr.tolist()
        solver_parameters[solver] = opt['x']
        solver_cov_matrices[solver] = res['V_matrix']
        solver_confidence_intervals[solver] = {
            i: ci for i, ci in enumerate(res['CI'])
        }
        if logging_flag:
            logger.info(f"Solver {solver}: Estimated θ = {scpr}")
            logger.info(f"Solver {solver}: CI = {res['CI']}")
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
    """
    FDM mesh-dependency test to choose epsilon for sensitivities.
    """
    logger.info(f"Performing mesh-independency test for {solver}")
    eps_values = np.logspace(-12, -1, 50)
    eigs = []
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

    # Choose epsilon minimizing variation
    valid = np.isfinite(eigs).all(axis=1)
    idx = np.where(valid)[0]
    win = 5
    half = win//2
    best, best_var = eps_values[0], np.inf
    for i in idx[half:-half]:
        var = eigs[i-half:i+half+1].std(axis=0).sum()
        if var<best_var:
            best_var, best = var, eps_values[i]
    logger.info(f"Selected epsilon={best:.1e} for {solver}")
    return best

