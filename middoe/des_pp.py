# # mbdoe_pp_complete.py
# from middoe.des_utils import _slicer, _reporter, configure_logger
# from middoe.krnl_simula import simula
# from collections import defaultdict
# from functools import partial
# import numpy as np
# from pymoo.algorithms.soo.nonconvex.de import DE
# from pymoo.operators.sampling.lhs import LHS
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize as pymoo_minimize
# from multiprocessing import Pool
# import traceback
# from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# from typing import Dict, Any, Union
# from middoe.sc_estima import parameter_ranking
#
#
# # ==================== MAIN ENTRY POINT ====================
# def mbdoe_pp(
#         des_opt: Dict[str, Any],
#         system: Dict[str, Any],
#         models: Dict[str, Any],
#         round: int,
#         num_parallel_runs: int = 1
# ) -> Dict[str, Union[Dict[str, Any], float]]:
#     """
#     Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).
#     """
#     if num_parallel_runs > 1:
#         with Pool(num_parallel_runs) as pool:
#             results_list = pool.map(
#                 _safe_run,
#                 [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
#             )
#
#         successful = [res for res in results_list if res is not None]
#         if not successful:
#             raise RuntimeError(
#                 "All MBDOE-PP optimisation runs failed. "
#                 "Try increasing 'maxpp', adjusting bounds, or relaxing constraints."
#             )
#
#         best_design_decisions, best_pp_obj, best_swps = max(successful, key=lambda x: x[1])
#
#     else:
#         try:
#             best_design_decisions, best_pp_obj, best_swps = _run_single_pp(
#                 des_opt, system, models, core_number=0, round=round
#             )
#         except Exception as e:
#             raise RuntimeError(f"Single-core optimisation failed: {e}")
#         print("Design your experiment based on:")
#         print("  tii   :", best_design_decisions['tii'])
#         print("  tvi   :", best_design_decisions['tvi'])
#         print("  swps  :", best_design_decisions['swps'])
#         print("  St    :", best_design_decisions['St'])
#         print("  pp_obj:", best_design_decisions['pp_obj'])
#     return best_design_decisions
#
#
# # ==================== UTILITY FUNCTIONS ====================
# def _safe_run(args):
#     des_opt, system, models, core_num, round = args
#     configure_logger()
#     try:
#         return _run_single_pp(des_opt, system, models, core_num, round)
#     except Exception as e:
#         print(f"[Core {core_num}] FAILED: {e}")
#         traceback.print_exc()
#         return None
#
#
# def _create_index_dict(ti_iphi_vars, tv_iphi_vars, tv_iphi_seg, tv_ophi_vars, tv_ophi_seg,
#                        tv_ophi_sampling, tv_ophi_forcedsamples, ti, tf):
#     """Create the index mapping dictionary for design variables."""
#     index_dict = {'ti': {}, 'swps': {}, 'st': {}}
#     idx_counter = 0
#
#     # Time-invariant parameters
#     for i, name in enumerate(ti_iphi_vars):
#         index_dict['ti'][name] = [idx_counter]
#         idx_counter += 1
#
#     # Control variable levels
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         level_idxs = list(range(idx_counter, idx_counter + seg - 1))
#         index_dict['swps'][name + 'l'] = level_idxs
#         idx_counter += seg - 1
#
#     # Control variable times
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         time_idxs = list(range(idx_counter, idx_counter + seg - 2))
#         index_dict['swps'][name + 't'] = time_idxs
#         idx_counter += seg - 2
#
#     # Sampling times (grouped by sampling strategy)
#     sampling_groups = defaultdict(list)
#     for var in tv_ophi_vars:
#         group_id = tv_ophi_sampling[var]
#         sampling_groups[group_id].append(var)
#
#     for group_id, group_vars in sampling_groups.items():
#         var = group_vars[0]
#         i = tv_ophi_vars.index(var)
#         seg = tv_ophi_seg[i]
#         num_forced = len(tv_ophi_forcedsamples[var])
#         num_free = seg - num_forced
#
#         idxs = list(range(idx_counter, idx_counter + num_free))
#         for var_in_group in group_vars:
#             index_dict['st'][var_in_group] = idxs
#         idx_counter += num_free
#
#     return index_dict
#
# def _get_initial_odr_ranking(models, active_solvers, mutation):
#     """
#     Gets the initial ODR ranking from a pre-computed LSA in models.
#     If LSA is not available, raises an error suggesting a different method.
#     """
#     # Check if LSA data is available for the first active solver
#     primary_solver = active_solvers[0]
#     if 'LSA' not in models or primary_solver not in models['LSA']:
#         raise ValueError(
#             "No pre-computed LSA information found in models['LSA'][solver_name']. "
#             "Cannot perform ODR-based design. "
#             "Please use a different method (e.g., 'D', 'A') for the first experiment design."
#         )
#
#     # Extract the sensitivity matrix Z from the models dict
#     Z = models['LSA'][primary_solver]
#
#     # Check which parameters are free (mutable) for this solver
#     free_params_indices = [i for i, is_free in enumerate(mutation[primary_solver]) if is_free]
#
#     # Slice the Z matrix to only include columns for free parameters
#     Z_free = Z[:, free_params_indices]
#
#     if Z_free is None or Z_free.size == 0 or Z_free.shape[1] == 0:
#         n_free_params = len(free_params_indices)
#         initial_ranking = list(range(n_free_params))
#         print(
#             f"Warning: LSA matrix for {primary_solver} is empty for free parameters, using natural order: {initial_ranking}")
#         return initial_ranking
#
#     # Use the robust parameter_ranking function to get the initial ranking
#     # This returns a list of indices relative to Z_free (i.e., the free parameters)
#     initial_ranking_free_indices = parameter_ranking(Z_free)
#
#     # The ranking is now based on the free parameter subspace.
#     # We can keep it like this. The ODR objective function will also work on Z_current_free.
#     print(f"Initial ODR ranking determined from pre-computed LSA: {initial_ranking_free_indices}")
#     return initial_ranking_free_indices
#
# def _pp_runner_odr_only(
#         x,
#         tv_iphi_vars, tv_iphi_max,
#         ti_iphi_vars, ti_iphi_max,
#         tv_ophi_vars, ti_ophi_vars,
#         active_solvers, theta_parameters,
#         tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#         tf, eps, mutation, V_matrix,
#         index_dict, system, models
# ):
#     """
#     Simplified version that only returns the sensitivity matrix Z for ODR ranking.
#     """
#     if x is None:
#         return None, None, None, None, None, None, None, None
#
#     x = x.tolist() if not isinstance(x, list) else x
#
#     dt_real = system['t_r']
#     nodes = int(round(tf / dt_real)) + 1
#     tlin = np.linspace(0, 1, nodes)
#     ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
#     St = {var: np.array(sorted(St[var])) for var in St}
#     t_values_flat = [tp for times in St.values() for tp in times]
#     t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()
#
#     LSA = defaultdict(lambda: defaultdict(dict))
#     indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars if var in St}
#
#     for solver_name in active_solvers:
#         thetac = theta_parameters[solver_name]
#         theta = np.array([1.0] * len(thetac))
#         ti_iphi_data = ti
#         swps_data = swps
#         phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
#         phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
#         tsc = tf
#
#         try:
#             tv_out, ti_out, interp_data = simula(t_values, swps_data, ti_iphi_data,
#                                                  phisc, phitsc, tsc,
#                                                  theta, thetac,
#                                                  tv_iphi_cvp, {}, solver_name,
#                                                  system, models)
#         except Exception as e:
#             print(f"Simulation failed in ODR preliminary: {e}")
#             return None, None, None, None, None, None, None, None
#
#         y_values_dict = {var: np.array(tv_out[var]) for var in tv_ophi_vars}
#         y_values_dict.update({var: np.array([ti_out[var]]) for var in ti_ophi_vars})
#         y_combined = dict(y_values_dict)
#
#         free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
#
#         for para_idx in free_params_indices:
#             modified_theta = theta.copy()
#             modified_theta[para_idx] += eps
#
#             try:
#                 tv_out_mod, ti_out_mod, _ = simula(t_values, swps_data, ti_iphi_data,
#                                                    phisc, phitsc, tsc,
#                                                    modified_theta, thetac,
#                                                    tv_iphi_cvp, {}, solver_name,
#                                                    system, models)
#             except Exception as e:
#                 print(f"Perturbed simulation failed: {e}")
#                 continue
#
#             y_i_values_dict = {var: np.array(tv_out_mod[var]) for var in tv_ophi_vars}
#             y_i_values_dict.update({var: np.array([ti_out_mod[var]]) for var in ti_ophi_vars})
#             y_modified_combined = dict(y_i_values_dict)
#
#             for var in indices:
#                 if var in y_modified_combined and var in y_combined:
#                     sensitivity = (y_modified_combined[var] - y_combined[var]) / eps
#                     LSA[var][solver_name][para_idx] = sensitivity
#
#         solver_LSA_list = []
#         for var in indices:
#             if var in LSA and solver_name in LSA[var]:
#                 row = [LSA[var][solver_name][i] for i in free_params_indices if i in LSA[var][solver_name]]
#                 if row:
#                     solver_LSA_list.append(np.array(row))
#
#         if solver_LSA_list:
#             solver_LSA = np.array(solver_LSA_list)
#             return ti, swps, St, None, t_values, tv_out, ti_out, solver_LSA
#
#     return ti, swps, St, None, t_values, {}, {}, None
#
#
# def _run_single_pp(des_opt, system, models, core_number=0, round=round):
#     # --------------------------- EXTRACT DATA --------------------------- #
#     tf = system['t_s'][1]
#     ti = system['t_d']
#
#     # Time-variant inputs
#     tv_iphi_vars = list(system['tvi'].keys())
#     tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
#     tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
#     tv_iphi_seg = [system['tvi'][var]['stps'] + 1 for var in tv_iphi_vars]
#     tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
#     tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
#     tv_iphi_offsetl = [
#         system['tvi'][var]['offl'] / system['tvi'][var]['max']
#         for var in tv_iphi_vars
#     ]
#     tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}
#
#     # Time-invariant inputs
#     ti_iphi_vars = list(system['tii'].keys())
#     ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
#     ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]
#
#     # Time-variant outputs
#     tv_ophi_vars = [
#         var for var in system['tvo'].keys()
#         if system['tvo'][var].get('meas', True)
#     ]
#     tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
#     tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
#     tv_ophi_sampling = {var: system['tvo'][var]['samp_s'] for var in tv_ophi_vars}
#     tv_ophi_forcedsamples = {var: [v / tf for v in system['tvo'][var]['samp_f']] for var in tv_ophi_vars}
#
#     # Time-invariant outputs
#     ti_ophi_vars = [
#         var for var in system['tio'].keys()
#         if system['tio'][var].get('meas', True)
#     ]
#
#     # Solver and optimization settings
#     active_solvers = models['can_m']
#     theta_parameters = models['theta']
#     design_criteria = des_opt['pp_ob']
#     maxpp = des_opt['itr']['maxpp']
#     tolpp = des_opt['itr']['tolpp']
#     population_size = des_opt['itr']['pps']
#     eps = des_opt['eps']
#
#     # Ensure V_matrix and mutation are properly formatted
#     if 'V_matrix' in models:
#         V_matrix = {
#             solver: np.array(models['V_matrix'][solver])
#             for solver in models['can_m']
#         }
#     else:
#         V_matrix = {
#             solver: np.eye(len(models['theta'][solver])) * 1e5
#             for solver in models['can_m']
#         }
#
#     if 'mutation' in models:
#         mutation = models['mutation']
#     else:
#         mutation = {
#             solver: [True] * len(models['theta'][solver])
#             for solver in models['can_m']
#         }
#
#     pltshow = des_opt['plt']
#     optmethod = des_opt.get('meth', 'L')
#
#     # ------------------------ CREATE INDEX_DICT FIRST ------------------------ #
#     index_dict = _create_index_dict(
#         ti_iphi_vars, tv_iphi_vars, tv_iphi_seg,
#         tv_ophi_vars, tv_ophi_seg, tv_ophi_sampling,
#         tv_ophi_forcedsamples, ti, tf
#     )
#
#     # ------------------------ GET ODR RANKING IF NEEDED ------------------------ #
#     odr_target_ranking = None
#     if design_criteria == 'odr':
#         odr_target_ranking = _get_initial_odr_ranking(models, active_solvers, mutation)
#         des_opt['odr_target_ranking'] = odr_target_ranking
#         print(f"ODR target ranking set: {odr_target_ranking}")
#
#     # ------------------------ OPTIMIZATION ------------------------ #
#     result, _ = _optimiser(
#         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#         ti_ophi_vars,
#         tf, ti,
#         active_solvers, theta_parameters,
#         eps, maxpp, tolpp, population_size,
#         mutation, V_matrix, design_criteria,
#         system, models, optmethod, des_opt, index_dict
#     )
#
#     x_final = getattr(result, 'X', getattr(result, 'x', None))
#     if x_final is None:
#         raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
#
#     # ------------------- USE FINAL SOLUTION ------------------- #
#     try:
#         phi, swps, St, pp_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _pp_runner(
#             x_final,
#             tv_iphi_vars, tv_iphi_max,
#             ti_iphi_vars, ti_iphi_max,
#             tv_ophi_vars, ti_ophi_vars,
#             active_solvers, theta_parameters,
#             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#             tf, eps, mutation, V_matrix, design_criteria,
#             index_dict,
#             des_opt,
#             system,
#             models
#         )
#     except ValueError as e:
#         if "MBDoE optimiser kernel was unsuccessful" in str(e):
#             print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
#             return None
#         else:
#             raise
#
#     # --------------------------- REPORT & SAVE --------------------------- #
#     phi, phit, swps, St = _reporter(
#         phi, phit, swps, St,
#         pp_obj,
#         t_values,
#         tv_ophi_dict, ti_ophi_dict,
#         tv_iphi_vars, tv_iphi_max,
#         ti_iphi_vars, ti_iphi_max,
#         tf,
#         design_criteria,
#         round, pltshow,
#         core_number
#     )
#
#     # ------------------------- RETURN FINAL DATA ------------------------- #
#     design_decisions = {
#         'tii': phi,
#         'tvi': phit,
#         'swps': swps,
#         'St': St,
#         'pp_obj': pp_obj,
#         't_values': t_values
#     }
#
#     return design_decisions, pp_obj, swps
#
#
# def _optimiser(
#         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#         ti_ophi_vars,
#         tf, ti,
#         active_models, theta_parameters,
#         eps, maxpp, tolpp, population_size,
#         mutation, V_matrix, design_criteria,
#         system, models, optmethod, des_opt, index_dict
# ):
#     bounds = []
#     x0 = []
#
#     # Use the pre-created index_dict
#     for i, (name, mn, mx) in enumerate(zip(ti_iphi_vars, ti_iphi_min, ti_iphi_max)):
#         lo, hi = mn / mx, 1.0
#         bounds.append((lo, hi))
#         x0.append((lo + hi) / 2)
#
#     start = len(index_dict['ti'])
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         mn, mx = tv_iphi_min[i], tv_iphi_max[i]
#         lo = mn / mx
#         level_idxs = index_dict['swps'][name + 'l']
#         for _ in range(seg - 1):
#             bounds.append((lo, 1.0))
#             x0.append((lo + 1.0) / 2)
#         start += seg - 1
#
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         lo, hi = ti / tf, 1 - ti / tf
#         time_idxs = index_dict['swps'][name + 't']
#         for _ in range(seg - 2):
#             bounds.append((lo, hi))
#             x0.append((lo + hi) / 2)
#         start += seg - 2
#
#     sampling_groups = defaultdict(list)
#     for var in tv_ophi_vars:
#         group_id = tv_ophi_sampling[var]
#         sampling_groups[group_id].append(var)
#
#     for group_id, group_vars in sampling_groups.items():
#         var = group_vars[0]
#         i = tv_ophi_vars.index(var)
#         seg = tv_ophi_seg[i]
#         num_forced = len(tv_ophi_forcedsamples[var])
#         num_free = seg - num_forced
#         lo, hi = ti / tf, 1 - ti / tf
#
#         idxs = index_dict['st'][var]
#         for _ in range(num_free):
#             bounds.append((lo, hi))
#             x0.append((lo + hi) / 2)
#         start += num_free
#
#     lower = np.array([b[0] for b in bounds])
#     upper = np.array([b[1] for b in bounds])
#     x0 = np.array(x0)
#
#     constraint_index_list = []
#     for i, name in enumerate(tv_iphi_vars):
#         const = tv_iphi_const[i]
#         if const != 'rel':
#             idxs = index_dict['swps'][name + 'l']
#             for j in range(len(idxs) - 1):
#                 constraint_index_list.append(('lvl', i, idxs[j], idxs[j + 1]))
#
#     for i, name in enumerate(tv_iphi_vars):
#         idxs = index_dict['swps'][name + 't']
#         for j in range(len(idxs) - 1):
#             constraint_index_list.append(('t', i, idxs[j], idxs[j + 1]))
#
#     for i, name in enumerate(tv_ophi_vars):
#         idxs = index_dict['st'][name]
#         for j in range(len(idxs) - 1):
#             constraint_index_list.append(('st', i, idxs[j], idxs[j + 1]))
#
#     total_constraints = len(constraint_index_list)
#
#     local_obj = partial(
#         _pp_of,
#         tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
#         ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
#         tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
#         tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
#         tv_ophi_sampling=tv_ophi_sampling,
#         active_solvers=active_models, theta_parameters=theta_parameters,
#         tf=tf, eps=eps, mutation=mutation,
#         V_matrix=V_matrix, design_criteria=design_criteria,
#         index_dict=index_dict, des_opt=des_opt, system=system, models=models
#     )
#
#     class DEProblem(ElementwiseProblem):
#         def __init__(self):
#             super().__init__(
#                 n_var=len(bounds),
#                 n_obj=1,
#                 n_constr=total_constraints,
#                 xl=lower,
#                 xu=upper
#             )
#
#         def _evaluate(self, x, out, *args, **kwargs):
#             f_val = local_obj(x)
#             g = []
#             for kind, i, i1, i2 in constraint_index_list:
#                 if kind == 'lvl':
#                     offs = tv_iphi_offsetl[i]
#                     const = tv_iphi_const[i]
#                     diff = x[i2] - x[i1] if const == 'inc' else x[i1] - x[i2]
#                     g.append(offs - diff)
#                 elif kind == 't':
#                     offs = tv_iphi_offsett[i]
#                     g.append(offs - (x[i2] - x[i1]))
#                 elif kind == 'st':
#                     offs = tv_ophi_offsett_ophi[i]
#                     g.append(offs - (x[i2] - x[i1]))
#             out['F'] = f_val
#             out['G'] = np.array(g, dtype=np.float64)
#
#     problem = DEProblem()
#
#     if optmethod == 'PS':
#         algorithm = PatternSearch()
#         res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxpp), seed=None, verbose=True)
#
#     elif optmethod == 'DE':
#         algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
#         res_refine = pymoo_minimize(
#             problem,
#             algorithm,
#             termination=('n_gen', maxpp),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolpp,
#             save_history=True
#         )
#
#     elif optmethod == 'DEPS':
#         algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
#         res_de = pymoo_minimize(
#             problem,
#             algorithm_de,
#             termination=('n_gen', maxpp / 2),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolpp,
#             save_history=True
#         )
#         algorithm_refine = PatternSearch()
#         res_refine = pymoo_minimize(
#             problem,
#             algorithm_refine,
#             termination=('n_gen', int(maxpp / 2)),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolpp,
#             x0=res_de.X
#         )
#
#     else:
#         raise ValueError("optmethod must be 'PS', 'DE', or 'DEPS'")
#
#     return res_refine, index_dict
#
#
# def _pp_of(
#         x,
#         tv_iphi_vars, tv_iphi_max,
#         ti_iphi_vars, ti_iphi_max,
#         tv_ophi_vars, ti_ophi_vars,
#         active_solvers, theta_parameters,
#         tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#         tf, eps, mutation, V_matrix, design_criteria,
#         index_dict, des_opt, system, models
# ):
#     """
#     Objective function wrapper for MBDOE-PP.
#     """
#     try:
#         _, _, _, pp_obj, _, _, _, _ = _pp_runner(
#             x,
#             tv_iphi_vars, tv_iphi_max,
#             ti_iphi_vars, ti_iphi_max,
#             tv_ophi_vars, ti_ophi_vars,
#             active_solvers, theta_parameters,
#             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#             tf, eps, mutation, V_matrix, design_criteria,
#             index_dict,
#             des_opt,
#             system,
#             models
#         )
#         if np.isnan(pp_obj):
#             return 1e6
#         return -pp_obj  # negative => maximize pp_obj
#     except Exception as e:
#         print(f"Exception in pp_objective_function: {e}")
#         return 1e6
#
#
# def _pp_runner(
#         x,
#         tv_iphi_vars,
#         tv_iphi_max,
#         ti_iphi_vars,
#         ti_iphi_max,
#         tv_ophi_vars,
#         ti_ophi_vars,
#         active_solvers,
#         theta_parameters,
#         tv_iphi_cvp,
#         tv_ophi_forcedsamples,
#         tv_ophi_sampling,
#         tf,
#         eps,
#         mutation,
#         V_matrix,
#         MBDOE_PP_criterion,
#         index_dict,
#         des_opt,
#         system,
#         models
# ):
#     """
#     Simulate models and evaluate the PP objective.
#     Returns the sliced inputs, objective value, time vector, and outputs.
#     """
#     if x is None:
#         raise ValueError("MBDoE optimiser kernel was unsuccessful, increase iteration to avoid constraint violations.")
#     x = x.tolist() if not isinstance(x, list) else x
#
#     dt_real = system['t_r']
#     nodes = int(round(tf / dt_real)) + 1
#     tlin = np.linspace(0, 1, nodes)
#     ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
#     St = {var: np.array(sorted(St[var])) for var in St}
#     t_values_flat = [tp for times in St.values() for tp in times]
#     t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()
#
#     LSA = defaultdict(lambda: defaultdict(dict))
#     J_dot_matrix = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
#                                                  len(theta_parameters[active_solvers[0]]))))
#     indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars if var in St}
#     M0 = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
#                                        len(theta_parameters[active_solvers[0]]))))
#     M = {}
#     tv_ophi, ti_ophi, phit_interp = {}, {}, {}
#
#     for solver_name in active_solvers:
#         thetac = theta_parameters[solver_name]
#         theta = np.array([1.0] * len(thetac))
#         ti_iphi_data = ti
#         swps_data = swps
#         phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
#         phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
#         tsc = tf
#
#         tv_out, ti_out, interp_data = simula(t_values, swps_data, ti_iphi_data,
#                                              phisc, phitsc, tsc,
#                                              theta, thetac,
#                                              tv_iphi_cvp, {}, solver_name,
#                                              system, models)
#         tv_ophi[solver_name] = tv_out
#         ti_ophi[solver_name] = ti_out
#         phit_interp = interp_data
#
#         y_values_dict = {var: np.array(tv_out[var]) for var in tv_ophi_vars}
#         y_values_dict.update({var: np.array([ti_out[var]]) for var in ti_ophi_vars})
#         y_combined = dict(y_values_dict)
#
#         free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
#         y_i_values_dict = {}
#
#         for para_idx in free_params_indices:
#             modified_theta = theta.copy()
#             modified_theta[para_idx] += eps
#
#             tv_out_mod, ti_out_mod, _ = simula(t_values, swps_data, ti_iphi_data,
#                                                phisc, phitsc, tsc,
#                                                modified_theta, thetac,
#                                                tv_iphi_cvp, {}, solver_name,
#                                                system, models)
#
#             y_i_values_dict = {var: np.array(tv_out_mod[var]) for var in tv_ophi_vars}
#             y_i_values_dict.update({var: np.array([ti_out_mod[var]]) for var in ti_ophi_vars})
#             y_modified_combined = dict(y_i_values_dict)
#
#             for var in indices:
#                 LSA[var][solver_name][para_idx] = (
#                                                           y_modified_combined[var] - y_combined[var]
#                                                   ) / eps
#
#         solver_LSA_list = []
#         for var in indices:
#             row = [LSA[var][solver_name][i].mean() for i in free_params_indices]
#             solver_LSA_list.append(np.array(row))
#         solver_LSA = np.array(solver_LSA_list)
#
#         # Build variance-covariance matrix Sigma_y
#         std_dev = {}
#         for grp, keys in [('tvo', tv_ophi_vars), ('tio', ti_ophi_vars)]:
#             for var in keys:
#                 unc = system.get(grp, {}).get(var, {}).get('unc', 1.0)
#                 if unc is None or np.isnan(unc):
#                     unc = 1.0
#                 std_dev[var] = unc
#
#         vars_order = list(indices.keys())
#         N_r = len(vars_order)
#         Sigma_y = np.zeros((N_r, N_r))
#         for i, var in enumerate(vars_order):
#             Sigma_y[i, i] = std_dev.get(var, 1.0) ** 2
#         Sigma_y_inv = np.linalg.inv(Sigma_y)
#
#         # Fisher Information Matrix
#         J = solver_LSA
#         M_fisher = J.T @ Sigma_y_inv @ J
#
#         if J_dot_matrix[solver_name].shape != M_fisher.shape:
#             J_dot_matrix[solver_name] = np.zeros_like(M_fisher)
#         J_dot_matrix[solver_name] += M_fisher
#         M0[solver_name] = J_dot_matrix[solver_name]
#
#         reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
#         M[solver_name] = M0[solver_name] + np.linalg.inv(reduced_V_matrix)
#
#     # Evaluate criterion
#     solver_name = active_solvers[0]
#     if MBDOE_PP_criterion == 'D':
#         pp_obj = np.linalg.det(M[solver_name])
#     elif MBDOE_PP_criterion == 'A':
#         pp_obj = np.trace(M[solver_name])
#     elif MBDOE_PP_criterion == 'E':
#         pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))
#     elif MBDOE_PP_criterion == 'ME':
#         pp_obj = -np.linalg.cond(M[solver_name])
#     # elif MBDOE_PP_criterion == 'odr':
#     #     # GET PREDETERMINED TARGET RANKING (of free parameters)
#     #     target_ranking = des_opt.get('odr_target_ranking', list(range(solver_LSA.shape[1])))
#     #     # solver_LSA is already the matrix for free parameters only
#     #
#     #     Z = solver_LSA  # Sensitivity matrix for the current candidate design (free params only)
#     #
#     #     # Apply the robust ranking algorithm to get the CURRENT ranking
#     #     # This returns a list of indices relative to the free parameters
#     #     current_ranking = parameter_ranking(Z)
#     #
#     #     # We want to REVERSE the initial target ranking
#     #     reverse_target_ranking = list(reversed(target_ranking))
#     #
#     #     # PRINT RANKING EVERY ITERATION
#     #     print(f"ODR - Current: {current_ranking}")
#     #     print(f"ODR - Target (reversed): {reverse_target_ranking}")
#     #
#     #     # Calculate how close current ranking is to reversed target
#     #     ranking_penalty = 0
#     #     n_params = len(current_ranking)
#     #     for i, param in enumerate(current_ranking):
#     #         try:
#     #             target_position = reverse_target_ranking.index(param)
#     #         except ValueError:
#     #             # This shouldn't happen if rankings are consistent, but handle gracefully
#     #             target_position = n_params  # assign worst-case penalty
#     #         ranking_penalty += abs(i - target_position)
#     #
#     #     # Normalize penalty (0 = perfect reverse, higher = worse)
#     #     max_penalty = sum(range(n_params))
#     #     reverse_ranking_score = 1 - (ranking_penalty / max_penalty) if max_penalty > 0 else 0
#     #
#     #     # Also ensure we have good information content
#     #     information_content = np.log(np.linalg.det(M[solver_name] + np.eye(n_params) * 1e-10))
#     #
#     #     # Combine objectives: good reverse ranking AND good information
#     #     pp_obj = reverse_ranking_score * information_content
#     #
#     #     print(f"ODR: reverse_score={reverse_ranking_score:.3f}, info={information_content:.3e}, obj={pp_obj:.3e}")
#     #
#     # else:
#     #     pp_obj = None
#     elif MBDOE_PP_criterion == 'odr':
#         # GET PREDETERMINED TARGET RANKING (of free parameters)
#         target_ranking = des_opt.get('odr_target_ranking', list(range(solver_LSA.shape[1])))
#         # solver_LSA is already the matrix for free parameters only
#
#         Z = solver_LSA  # Sensitivity matrix for the current candidate design (free params only)
#
#         # Apply the robust ranking algorithm to get the CURRENT ranking
#         # This returns a list of indices relative to the free parameters
#         current_ranking = parameter_ranking(Z)
#
#         # We want to REVERSE the initial target ranking
#         reverse_target_ranking = list(reversed(target_ranking))
#
#         print(f"ODR - Current: {current_ranking}")
#         print(f"ODR - Target (reversed): {reverse_target_ranking}")
#
#         # Create a rank lookup for reversed target
#         reverse_rank_dict = {param: rank for rank, param in enumerate(reverse_target_ranking)}
#
#         # Compute squared differences penalty
#         n_params = len(current_ranking)
#         squared_diffs = [(i - reverse_rank_dict.get(param, n_params)) ** 2 for i, param in enumerate(current_ranking)]
#         penalty = sum(squared_diffs)
#         # Maximum possible penalty for normalization: n * (n^2 - 1) / 6 (sum of squared first n-1 integers)
#         max_penalty = n_params * (n_params ** 2 - 1) / 6
#         normalized_penalty = penalty / max_penalty if max_penalty > 0 else 0
#
#         # Convert penalty to a score: 1 = perfect match, 0 = worst
#         reverse_ranking_score = 1 - normalized_penalty
#
#         # Regularize information content for numerical stability
#         reg_lambda = 1e-6
#         M_reg = M[solver_name] + reg_lambda * np.eye(n_params)
#
#         try:
#             information_content = np.log(np.linalg.det(M_reg))
#         except np.linalg.LinAlgError:
#             information_content = -np.inf  # Penalize ill-conditioned matrices
#
#         # Combine with sigmoid smoothing on information content
#         beta = 10  # steepness for sigmoid
#         gamma = 0.1  # midpoint
#         smooth_info = 1 / (1 + np.exp(-beta * (information_content - gamma)))
#
#         # Weighted combination (can tune alpha)
#         alpha = 0.5
#         pp_obj = alpha * reverse_ranking_score + (1 - alpha) * smooth_info
#
#         print(
#             f"ODR: reverse_score={reverse_ranking_score:.3f}, info={information_content:.3e}, smooth_info={smooth_info:.3f}, obj={pp_obj:.3e}")
#
#     else:
#         pp_obj = None
#
#     return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
#
#
#
# #
# #
# # from middoe.des_utils import _slicer, _reporter, configure_logger
# # from middoe.krnl_simula import simula
# # from collections import defaultdict
# # from functools import partial
# # import numpy as np
# # from pymoo.algorithms.soo.nonconvex.de import DE
# # from pymoo.operators.sampling.lhs import LHS
# # from pymoo.core.problem import ElementwiseProblem
# # from pymoo.optimize import minimize as pymoo_minimize
# # from multiprocessing import Pool
# # import traceback
# # from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# # from typing import Dict, Any, Union
# # from middoe.sc_estima import parameter_ranking
# #
# # # ========== MAIN ENTRY POINT ==========
# # def mbdoe_pp(
# #         des_opt: Dict[str, Any],
# #         system: Dict[str, Any],
# #         models: Dict[str, Any],
# #         round: int,
# #         num_parallel_runs: int = 1
# # ) -> Dict[str, Union[Dict[str, Any], float]]:
# #     """
# #     Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).
# #     """
# #     if num_parallel_runs > 1:
# #         with Pool(num_parallel_runs) as pool:
# #             results_list = pool.map(
# #                 _safe_run,
# #                 [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
# #             )
# #         successful = [res for res in results_list if res is not None]
# #         if not successful:
# #             raise RuntimeError(
# #                 "All MBDOE-PP optimisation runs failed. "
# #                 "Try increasing 'maxpp', adjusting bounds, or relaxing constraints."
# #             )
# #         best_design_decisions, best_pp_obj, best_swps = max(successful, key=lambda x: x[1])
# #     else:
# #         try:
# #             best_design_decisions, best_pp_obj, best_swps = _run_single_pp(
# #                 des_opt, system, models, core_number=0, round=round
# #             )
# #         except Exception as e:
# #             raise RuntimeError(f"Single-core optimisation failed: {e}")
# #         print("Design your experiment based on:")
# #         print("  tii   :", best_design_decisions['tii'])
# #         print("  tvi   :", best_design_decisions['tvi'])
# #         print("  swps  :", best_design_decisions['swps'])
# #         print("  St    :", best_design_decisions['St'])
# #         print("  pp_obj:", best_design_decisions['pp_obj'])
# #     return best_design_decisions
# #
# # # ========== UTILITY FUNCTIONS ==========
# # def _safe_run(args):
# #     des_opt, system, models, core_num, round = args
# #     configure_logger()
# #     try:
# #         return _run_single_pp(des_opt, system, models, core_num, round)
# #     except Exception as e:
# #         print(f"[Core {core_num}] FAILED: {e}")
# #         traceback.print_exc()
# #         return None
# #
# # def _create_index_dict(ti_iphi_vars, tv_iphi_vars, tv_iphi_seg, tv_ophi_vars, tv_ophi_seg,
# #                        tv_ophi_sampling, tv_ophi_forcedsamples, ti, tf):
# #     index_dict = {'ti': {}, 'swps': {}, 'st': {}}
# #     idx_counter = 0
# #     for i, name in enumerate(ti_iphi_vars):
# #         index_dict['ti'][name] = [idx_counter]
# #         idx_counter += 1
# #     for i, name in enumerate(tv_iphi_vars):
# #         seg = tv_iphi_seg[i]
# #         level_idxs = list(range(idx_counter, idx_counter + seg - 1))
# #         index_dict['swps'][name + 'l'] = level_idxs
# #         idx_counter += seg - 1
# #     for i, name in enumerate(tv_iphi_vars):
# #         seg = tv_iphi_seg[i]
# #         time_idxs = list(range(idx_counter, idx_counter + seg - 2))
# #         index_dict['swps'][name + 't'] = time_idxs
# #         idx_counter += seg - 2
# #     sampling_groups = defaultdict(list)
# #     for var in tv_ophi_vars:
# #         group_id = tv_ophi_sampling[var]
# #         sampling_groups[group_id].append(var)
# #     for group_id, group_vars in sampling_groups.items():
# #         var = group_vars[0]
# #         i = tv_ophi_vars.index(var)
# #         seg = tv_ophi_seg[i]
# #         num_forced = len(tv_ophi_forcedsamples[var])
# #         num_free = seg - num_forced
# #         idxs = list(range(idx_counter, idx_counter + num_free))
# #         for var_in_group in group_vars:
# #             index_dict['st'][var_in_group] = idxs
# #         idx_counter += num_free
# #     return index_dict
# #
# # def _get_initial_odr_ranking(models, active_solvers, mutation):
# #     primary_solver = active_solvers[0]
# #     if 'LSA' not in models or primary_solver not in models['LSA']:
# #         raise ValueError(
# #             "No pre-computed LSA information found in models['LSA'][solver_name']. "
# #             "Cannot perform ODR-based design. "
# #             "Please use a different method (e.g., 'D', 'A') for the first experiment design."
# #         )
# #     Z = models['LSA'][primary_solver]
# #     free_params_indices = [i for i, is_free in enumerate(mutation[primary_solver]) if is_free]
# #     Z_free = Z[:, free_params_indices]
# #     if Z_free is None or Z_free.size == 0 or Z_free.shape[1] == 0:
# #         n_free_params = len(free_params_indices)
# #         initial_ranking = list(range(n_free_params))
# #         print(f"Warning: LSA matrix for {primary_solver} is empty for free parameters, using natural order: {initial_ranking}")
# #         return initial_ranking
# #     initial_ranking_free_indices = parameter_ranking(Z_free)
# #     print(f"Initial ODR ranking determined from pre-computed LSA: {initial_ranking_free_indices}")
# #     return initial_ranking_free_indices
# #
# # def orthogonal_deflation_scores(Z, max_iter=None):
# #     Z = np.array(Z)
# #     n_params = Z.shape[1]
# #     if max_iter is None:
# #         max_iter = n_params
# #     scores = np.zeros(n_params)
# #     selected = []
# #     residual = Z.copy()
# #     for k in range(max_iter):
# #         col_norms = np.linalg.norm(residual, axis=0)
# #         for idx in selected:
# #             col_norms[idx] = -np.inf
# #         next_idx = np.argmax(col_norms)
# #         if col_norms[next_idx] == -np.inf:
# #             break
# #         selected.append(next_idx)
# #         scores[next_idx] += col_norms[next_idx] / (k + 1)
# #         P_k = residual[:, selected]
# #         try:
# #             P_k_inv = np.linalg.inv(P_k.T @ P_k)
# #             proj_matrix = P_k @ P_k_inv @ P_k.T
# #         except np.linalg.LinAlgError:
# #             break
# #         residual = Z - proj_matrix @ Z
# #     return scores
# #
# # def _run_single_pp(des_opt, system, models, core_number=0, round=round):
# #     tf = system['t_s'][1]
# #     ti = system['t_d']
# #     tv_iphi_vars = list(system['tvi'].keys())
# #     tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
# #     tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
# #     tv_iphi_seg = [system['tvi'][var]['stps'] + 1 for var in tv_iphi_vars]
# #     tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
# #     tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
# #     tv_iphi_offsetl = [system['tvi'][var]['offl'] / system['tvi'][var]['max'] for var in tv_iphi_vars]
# #     tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}
# #     ti_iphi_vars = list(system['tii'].keys())
# #     ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
# #     ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]
# #     tv_ophi_vars = [var for var in system['tvo'].keys() if system['tvo'][var].get('meas', True)]
# #     tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
# #     tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
# #     tv_ophi_sampling = {var: system['tvo'][var]['samp_s'] for var in tv_ophi_vars}
# #     tv_ophi_forcedsamples = {var: [v / tf for v in system['tvo'][var]['samp_f']] for var in tv_ophi_vars}
# #     ti_ophi_vars = [var for var in system['tio'].keys() if system['tio'][var].get('meas', True)]
# #     active_solvers = models['can_m']
# #     theta_parameters = models['theta']
# #     design_criteria = des_opt['pp_ob']
# #     maxpp = des_opt['itr']['maxpp']
# #     tolpp = des_opt['itr']['tolpp']
# #     population_size = des_opt['itr']['pps']
# #     eps = des_opt['eps']
# #     if 'V_matrix' in models:
# #         V_matrix = {solver: np.array(models['V_matrix'][solver]) for solver in models['can_m']}
# #     else:
# #         V_matrix = {solver: np.eye(len(models['theta'][solver])) * 1e5 for solver in models['can_m']}
# #     mutation = models.get('mutation', {solver: [True] * len(models['theta'][solver]) for solver in models['can_m']})
# #     pltshow = des_opt['plt']
# #     optmethod = des_opt.get('meth', 'L')
# #     index_dict = _create_index_dict(
# #         ti_iphi_vars, tv_iphi_vars, tv_iphi_seg,
# #         tv_ophi_vars, tv_ophi_seg, tv_ophi_sampling,
# #         tv_ophi_forcedsamples, ti, tf
# #     )
# #     odr_target_ranking = None
# #     if design_criteria == 'odr':
# #         odr_target_ranking = _get_initial_odr_ranking(models, active_solvers, mutation)
# #         des_opt['odr_target_ranking'] = odr_target_ranking
# #         print(f"ODR target ranking set: {odr_target_ranking}")
# #     result, _ = _optimiser(
# #         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
# #         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
# #         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
# #         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
# #         ti_ophi_vars,
# #         tf, ti,
# #         active_solvers, theta_parameters,
# #         eps, maxpp, tolpp, population_size,
# #         mutation, V_matrix, design_criteria,
# #         system, models, optmethod, des_opt, index_dict
# #     )
# #     x_final = getattr(result, 'X', getattr(result, 'x', None))
# #     if x_final is None:
# #         raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
# #     try:
# #         phi, swps, St, pp_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _pp_runner(
# #             x_final,
# #             tv_iphi_vars, tv_iphi_max,
# #             ti_iphi_vars, ti_iphi_max,
# #             tv_ophi_vars, ti_ophi_vars,
# #             active_solvers, theta_parameters,
# #             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
# #             tf, eps, mutation, V_matrix, design_criteria,
# #             index_dict,
# #             des_opt,
# #             system,
# #             models
# #         )
# #     except ValueError as e:
# #         if "MBDoE optimiser kernel was unsuccessful" in str(e):
# #             print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
# #             return None
# #         else:
# #             raise
# #     phi, phit, swps, St = _reporter(
# #         phi, phit, swps, St,
# #         pp_obj,
# #         t_values,
# #         tv_ophi_dict, ti_ophi_dict,
# #         tv_iphi_vars, tv_iphi_max,
# #         ti_iphi_vars, ti_iphi_max,
# #         tf,
# #         design_criteria,
# #         round, pltshow,
# #         core_number
# #     )
# #     design_decisions = {
# #         'tii': phi,
# #         'tvi': phit,
# #         'swps': swps,
# #         'St': St,
# #         'pp_obj': pp_obj,
# #         't_values': t_values
# #     }
# #     return design_decisions, pp_obj, swps
# #
# #
# # def _optimiser(
# #         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
# #         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
# #         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
# #         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
# #         ti_ophi_vars,
# #         tf, ti,
# #         active_models, theta_parameters,
# #         eps, maxpp, tolpp, population_size,
# #         mutation, V_matrix, design_criteria,
# #         system, models, optmethod, des_opt, index_dict
# # ):
# #     bounds = []
# #     x0 = []
# #
# #     # Use the pre-created index_dict
# #     for i, (name, mn, mx) in enumerate(zip(ti_iphi_vars, ti_iphi_min, ti_iphi_max)):
# #         lo, hi = mn / mx, 1.0
# #         bounds.append((lo, hi))
# #         x0.append((lo + hi) / 2)
# #
# #     start = len(index_dict['ti'])
# #     for i, name in enumerate(tv_iphi_vars):
# #         seg = tv_iphi_seg[i]
# #         mn, mx = tv_iphi_min[i], tv_iphi_max[i]
# #         lo = mn / mx
# #         level_idxs = index_dict['swps'][name + 'l']
# #         for _ in range(seg - 1):
# #             bounds.append((lo, 1.0))
# #             x0.append((lo + 1.0) / 2)
# #         start += seg - 1
# #
# #     for i, name in enumerate(tv_iphi_vars):
# #         seg = tv_iphi_seg[i]
# #         lo, hi = ti / tf, 1 - ti / tf
# #         time_idxs = index_dict['swps'][name + 't']
# #         for _ in range(seg - 2):
# #             bounds.append((lo, hi))
# #             x0.append((lo + hi) / 2)
# #         start += seg - 2
# #
# #     sampling_groups = defaultdict(list)
# #     for var in tv_ophi_vars:
# #         group_id = tv_ophi_sampling[var]
# #         sampling_groups[group_id].append(var)
# #
# #     for group_id, group_vars in sampling_groups.items():
# #         var = group_vars[0]
# #         i = tv_ophi_vars.index(var)
# #         seg = tv_ophi_seg[i]
# #         num_forced = len(tv_ophi_forcedsamples[var])
# #         num_free = seg - num_forced
# #         lo, hi = ti / tf, 1 - ti / tf
# #
# #         idxs = index_dict['st'][var]
# #         for _ in range(num_free):
# #             bounds.append((lo, hi))
# #             x0.append((lo + hi) / 2)
# #         start += num_free
# #
# #     lower = np.array([b[0] for b in bounds])
# #     upper = np.array([b[1] for b in bounds])
# #     x0 = np.array(x0)
# #
# #     constraint_index_list = []
# #     for i, name in enumerate(tv_iphi_vars):
# #         const = tv_iphi_const[i]
# #         if const != 'rel':
# #             idxs = index_dict['swps'][name + 'l']
# #             for j in range(len(idxs) - 1):
# #                 constraint_index_list.append(('lvl', i, idxs[j], idxs[j + 1]))
# #
# #     for i, name in enumerate(tv_iphi_vars):
# #         idxs = index_dict['swps'][name + 't']
# #         for j in range(len(idxs) - 1):
# #             constraint_index_list.append(('t', i, idxs[j], idxs[j + 1]))
# #
# #     for i, name in enumerate(tv_ophi_vars):
# #         idxs = index_dict['st'][name]
# #         for j in range(len(idxs) - 1):
# #             constraint_index_list.append(('st', i, idxs[j], idxs[j + 1]))
# #
# #     total_constraints = len(constraint_index_list)
# #
# #     local_obj = partial(
# #         _pp_of,
# #         tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
# #         ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
# #         tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
# #         tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
# #         tv_ophi_sampling=tv_ophi_sampling,
# #         active_solvers=active_models, theta_parameters=theta_parameters,
# #         tf=tf, eps=eps, mutation=mutation,
# #         V_matrix=V_matrix, design_criteria=design_criteria,
# #         index_dict=index_dict, des_opt=des_opt, system=system, models=models
# #     )
# #
# #     class DEProblem(ElementwiseProblem):
# #         def __init__(self):
# #             super().__init__(
# #                 n_var=len(bounds),
# #                 n_obj=1,
# #                 n_constr=total_constraints,
# #                 xl=lower,
# #                 xu=upper
# #             )
# #
# #         def _evaluate(self, x, out, *args, **kwargs):
# #             f_val = local_obj(x)
# #             g = []
# #             for kind, i, i1, i2 in constraint_index_list:
# #                 if kind == 'lvl':
# #                     offs = tv_iphi_offsetl[i]
# #                     const = tv_iphi_const[i]
# #                     diff = x[i2] - x[i1] if const == 'inc' else x[i1] - x[i2]
# #                     g.append(offs - diff)
# #                 elif kind == 't':
# #                     offs = tv_iphi_offsett[i]
# #                     g.append(offs - (x[i2] - x[i1]))
# #                 elif kind == 'st':
# #                     offs = tv_ophi_offsett_ophi[i]
# #                     g.append(offs - (x[i2] - x[i1]))
# #             out['F'] = f_val
# #             out['G'] = np.array(g, dtype=np.float64)
# #
# #     problem = DEProblem()
# #
# #     if optmethod == 'PS':
# #         algorithm = PatternSearch()
# #         res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxpp), seed=None, verbose=True)
# #
# #     elif optmethod == 'DE':
# #         algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
# #         res_refine = pymoo_minimize(
# #             problem,
# #             algorithm,
# #             termination=('n_gen', maxpp),
# #             seed=None,
# #             verbose=True,
# #             constraint_tolerance=tolpp,
# #             save_history=True
# #         )
# #
# #     elif optmethod == 'DEPS':
# #         algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
# #         res_de = pymoo_minimize(
# #             problem,
# #             algorithm_de,
# #             termination=('n_gen', maxpp / 2),
# #             seed=None,
# #             verbose=True,
# #             constraint_tolerance=tolpp,
# #             save_history=True
# #         )
# #         algorithm_refine = PatternSearch()
# #         res_refine = pymoo_minimize(
# #             problem,
# #             algorithm_refine,
# #             termination=('n_gen', int(maxpp / 2)),
# #             seed=None,
# #             verbose=True,
# #             constraint_tolerance=tolpp,
# #             x0=res_de.X
# #         )
# #
# #     else:
# #         raise ValueError("optmethod must be 'PS', 'DE', or 'DEPS'")
# #
# #     return res_refine, index_dict
# #
# #
# # def _pp_of(
# #         x,
# #         tv_iphi_vars, tv_iphi_max,
# #         ti_iphi_vars, ti_iphi_max,
# #         tv_ophi_vars, ti_ophi_vars,
# #         active_solvers, theta_parameters,
# #         tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
# #         tf, eps, mutation, V_matrix, design_criteria,
# #         index_dict, des_opt, system, models
# # ):
# #     """
# #     Objective function wrapper for MBDOE-PP.
# #     """
# #     try:
# #         _, _, _, pp_obj, _, _, _, _ = _pp_runner(
# #             x,
# #             tv_iphi_vars, tv_iphi_max,
# #             ti_iphi_vars, ti_iphi_max,
# #             tv_ophi_vars, ti_ophi_vars,
# #             active_solvers, theta_parameters,
# #             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
# #             tf, eps, mutation, V_matrix, design_criteria,
# #             index_dict,
# #             des_opt,
# #             system,
# #             models
# #         )
# #         if np.isnan(pp_obj):
# #             return 1e6
# #         return -pp_obj  # negative => maximize pp_obj
# #     except Exception as e:
# #         print(f"Exception in pp_objective_function: {e}")
# #         return 1e6
# #
# #
# #
# # def _pp_runner(
# #         x,
# #         tv_iphi_vars,
# #         tv_iphi_max,
# #         ti_iphi_vars,
# #         ti_iphi_max,
# #         tv_ophi_vars,
# #         ti_ophi_vars,
# #         active_solvers,
# #         theta_parameters,
# #         tv_iphi_cvp,
# #         tv_ophi_forcedsamples,
# #         tv_ophi_sampling,
# #         tf,
# #         eps,
# #         mutation,
# #         V_matrix,
# #         MBDOE_PP_criterion,
# #         index_dict,
# #         des_opt,
# #         system,
# #         models
# # ):
# #     """
# #     Simulate models and evaluate the PP objective.
# #     Returns the sliced inputs, objective value, time vector, and outputs.
# #     """
# #     if x is None:
# #         raise ValueError("MBDoE optimiser kernel was unsuccessful, increase iteration to avoid constraint violations.")
# #     x = x.tolist() if not isinstance(x, list) else x
# #
# #     dt_real = system['t_r']
# #     nodes = int(round(tf / dt_real)) + 1
# #     tlin = np.linspace(0, 1, nodes)
# #     ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
# #     St = {var: np.array(sorted(St[var])) for var in St}
# #     t_values_flat = [tp for times in St.values() for tp in times]
# #     t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()
# #
# #     LSA = defaultdict(lambda: defaultdict(dict))
# #     J_dot_matrix = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
# #                                                  len(theta_parameters[active_solvers[0]]))))
# #     indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars if var in St}
# #     M0 = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
# #                                        len(theta_parameters[active_solvers[0]]))))
# #     M = {}
# #     tv_ophi, ti_ophi, phit_interp = {}, {}, {}
# #
# #     for solver_name in active_solvers:
# #         thetac = theta_parameters[solver_name]
# #         theta = np.array([1.0] * len(thetac))
# #         ti_iphi_data = ti
# #         swps_data = swps
# #         phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
# #         phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
# #         tsc = tf
# #
# #         tv_out, ti_out, interp_data = simula(t_values, swps_data, ti_iphi_data,
# #                                              phisc, phitsc, tsc,
# #                                              theta, thetac,
# #                                              tv_iphi_cvp, {}, solver_name,
# #                                              system, models)
# #         tv_ophi[solver_name] = tv_out
# #         ti_ophi[solver_name] = ti_out
# #         phit_interp = interp_data
# #
# #         y_values_dict = {var: np.array(tv_out[var]) for var in tv_ophi_vars}
# #         y_values_dict.update({var: np.array([ti_out[var]]) for var in ti_ophi_vars})
# #         y_combined = dict(y_values_dict)
# #
# #         free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
# #         y_i_values_dict = {}
# #
# #         for para_idx in free_params_indices:
# #             modified_theta = theta.copy()
# #             modified_theta[para_idx] += eps
# #
# #             tv_out_mod, ti_out_mod, _ = simula(t_values, swps_data, ti_iphi_data,
# #                                                phisc, phitsc, tsc,
# #                                                modified_theta, thetac,
# #                                                tv_iphi_cvp, {}, solver_name,
# #                                                system, models)
# #
# #             y_i_values_dict = {var: np.array(tv_out_mod[var]) for var in tv_ophi_vars}
# #             y_i_values_dict.update({var: np.array([ti_out_mod[var]]) for var in ti_ophi_vars})
# #             y_modified_combined = dict(y_i_values_dict)
# #
# #             for var in indices:
# #                 LSA[var][solver_name][para_idx] = (
# #                                                           y_modified_combined[var] - y_combined[var]
# #                                                   ) / eps
# #
# #         solver_LSA_list = []
# #         for var in indices:
# #             row = [LSA[var][solver_name][i].mean() for i in free_params_indices]
# #             solver_LSA_list.append(np.array(row))
# #         solver_LSA = np.array(solver_LSA_list)
# #
# #         # Build variance-covariance matrix Sigma_y
# #         std_dev = {}
# #         for grp, keys in [('tvo', tv_ophi_vars), ('tio', ti_ophi_vars)]:
# #             for var in keys:
# #                 unc = system.get(grp, {}).get(var, {}).get('unc', 1.0)
# #                 if unc is None or np.isnan(unc):
# #                     unc = 1.0
# #                 std_dev[var] = unc
# #
# #         vars_order = list(indices.keys())
# #         N_r = len(vars_order)
# #         Sigma_y = np.zeros((N_r, N_r))
# #         for i, var in enumerate(vars_order):
# #             Sigma_y[i, i] = std_dev.get(var, 1.0) ** 2
# #         Sigma_y_inv = np.linalg.inv(Sigma_y)
# #
# #         # Fisher Information Matrix
# #         J = solver_LSA
# #         M_fisher = J.T @ Sigma_y_inv @ J
# #
# #         if J_dot_matrix[solver_name].shape != M_fisher.shape:
# #             J_dot_matrix[solver_name] = np.zeros_like(M_fisher)
# #         J_dot_matrix[solver_name] += M_fisher
# #         M0[solver_name] = J_dot_matrix[solver_name]
# #
# #         reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
# #         M[solver_name] = M0[solver_name] + np.linalg.inv(reduced_V_matrix)
# #
# #     # Evaluate criterion
# #     solver_name = active_solvers[0]
# #     if MBDOE_PP_criterion == 'D':
# #         pp_obj = np.linalg.det(M[solver_name])
# #     elif MBDOE_PP_criterion == 'A':
# #         pp_obj = np.trace(M[solver_name])
# #     elif MBDOE_PP_criterion == 'E':
# #         pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))
# #     elif MBDOE_PP_criterion == 'ME':
# #         pp_obj = -np.linalg.cond(M[solver_name])
# #     elif MBDOE_PP_criterion == 'odr':
# #         target_ranking = des_opt.get('odr_target_ranking', list(range(solver_LSA.shape[1])))
# #         Z = solver_LSA
# #         current_ranking = parameter_ranking(Z)
# #         reverse_target_ranking = list(reversed(target_ranking))
# #         print(f"ODR - Current: {current_ranking}")
# #         print(f"ODR - Target (reversed): {reverse_target_ranking}")
# #         reverse_rank_dict = {param: rank for rank, param in enumerate(reverse_target_ranking)}
# #         n_params = len(current_ranking)
# #         squared_diffs = [(i - reverse_rank_dict.get(param, n_params)) ** 2 for i, param in enumerate(current_ranking)]
# #         penalty = sum(squared_diffs)
# #         max_penalty = n_params * (n_params ** 2 - 1) / 6
# #         normalized_penalty = penalty / max_penalty if max_penalty > 0 else 0
# #         reverse_ranking_score = 1 - normalized_penalty
# #         reg_lambda = 1e-6
# #         M_reg = M[solver_name] + reg_lambda * np.eye(n_params)
# #         try:
# #             information_content = np.log(np.linalg.det(M_reg))
# #         except np.linalg.LinAlgError:
# #             information_content = -np.inf
# #         beta = 10
# #         gamma = 0.1
# #         smooth_info = 1 / (1 + np.exp(-beta * (information_content - gamma)))
# #         alpha = 0.5
# #         pp_obj = alpha * reverse_ranking_score + (1 - alpha) * smooth_info
# #         print(
# #             f"ODR: reverse_score={reverse_ranking_score:.3f}, info={information_content:.3e}, smooth_info={smooth_info:.3f}, obj={pp_obj:.3e}")
# #     else:
# #         pp_obj = None
# #
# #     return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
# #
#
#




from middoe.des_utils import _slicer, _reporter, configure_logger
from middoe.krnl_simula import simula
from collections import defaultdict
from functools import partial
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from multiprocessing import Pool
import traceback
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from typing import Dict, Any, Union
from middoe.sc_estima import parameter_ranking
import logging

logger = logging.getLogger(__name__)


def mbdoe_pp(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    round: int,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    """
    Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).

    This function searches for the most informative experimental design to
    improve parameter estimation accuracy by maximising a precision criterion
    based on the Fisher Information Matrix (FIM). Optimisation can be run
    in single-core or parallel mode, and the best valid design is returned.

    Parameters
    ----------
    des_opt : dict
        Optimisation settings, including:
            - 'criteria' : dict[str, str]
                Precision criterion (e.g., {'MBDOE_PP_criterion': 'D'}).
            - 'iteration_settings' : dict
                Iteration controls:
                    * 'maxpp': int — maximum iterations
                    * 'tolpp': float — convergence tolerance
            - 'eps' : float
                Perturbation step size for numerical derivatives.

    system : dict
        System model and constraints:
            - 'tf', 'ti' : float
                Final and initial times for simulation.
            - 'tv_iphi', 'ti_iphi' : dict
                Time-variant and time-invariant input definitions.
            - 'tv_ophi', 'ti_ophi' : dict
                Output definitions (time-variant/invariant).
            - 'tv_iphi_seg' : dict[str, int]
                Control variable segmentation (CVP).
            - 'tv_iphi_offsett', 'tv_iphi_offsetl' : dict
                Time/level offsets for switching logic.
            - 'tv_ophi_matching' : dict[str, bool]
                Synchronised sampling across outputs.
            - 'sampling_constraints' : dict
                Sampling restrictions:
                    * 'min_interval': float — minimum time between samples
                    * 'max_samples_per_response': int
                    * 'total_samples': int
                    * 'forbidden_times': dict[str, list[float]]
                    * 'forced_times': dict[str, list[float]]
                    * 'sync_sampling': bool

    models : dict
        Model definitions and settings:
            - 'active_models' : list[str]
                Models to evaluate.
            - 'theta_parameters' : dict[str, list[float]]
                Nominal parameter values.
            - 'mutation' : dict[str, list[bool]]
                Parameter masks for optimisation.
            - 'V_matrix' : dict[str, ndarray]
                Prior covariance matrices.
            - 'ext_models' : dict[str, Callable]
                Mapping of model names to simulator functions.

    round : int
        Current experimental design round (used for logging/conditional logic).

    num_parallel_runs : int, optional
        Number of optimisation runs in parallel (default: 1, single-core mode).

    Returns
    -------
    results : dict
        Best design found:
            - 'x' : dict[str, Any]
                Optimal design variables (e.g., sampling times, control profiles).
            - 'fun' : float
                Objective value (e.g., log det(FIM)).
            - 'swps' : dict[str, list[float]]
                Switching times for time-variant controls.

    Raises
    ------
    RuntimeError
        If all parallel runs fail or if the single-core run encounters an exception.

    Notes
    -----
    MBDoE-PP designs experiments that maximise information for parameter estimation,
    using precision-based criteria such as:
        - D-optimality (maximise log det(FIM))
        - A-optimality (minimise trace(FIM⁻¹))
        - E-optimality (maximise smallest eigenvalue of FIM)

    Supported constraints include:
        - Control level bounds and switching rules
        - Minimum sampling intervals
        - Total/maximum samples per output
        - Forced or forbidden sampling times
        - Synchronised sampling across outputs

    References
    ----------
    Franceschini, G., & Macchietto, S. (2008).
    Model-based design of experiments for parameter precision: State of the art.
    Chemical Engineering Science, 63(19), 4846–4872.
    https://doi.org/10.1016/j.ces.2008.07.006

    See Also
    --------
    _safe_run : Parallel optimisation wrapper.
    _run_single_pp : Single-core optimisation routine.

    Examples
    --------
    >>> result = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=4)
    >>> print("Best objective value:", result['fun'])
    >>> print("Optimal design:", result['x'])
    """
    # Assuming active solvers and mutation info available
    active_solvers = models['can_m']
    solver_name = active_solvers[0] if active_solvers else None

    if solver_name and 'LSA' in models and solver_name in models['LSA']:
        Z_init = models['LSA'][solver_name]
        initial_ranking = parameter_ranking(Z_init)
        logger.debug(f"Initial parameter ranking (most to least estimable) for solver '{solver_name}': {initial_ranking}")
        des_opt['initial_ranking'] = initial_ranking
    else:
        logger.debug(f"No initial LSA matrix found for solver '{solver_name}' to compute initial ranking.")
    if num_parallel_runs > 1:
        with Pool(num_parallel_runs) as pool:
            results_list = pool.map(
                _safe_run,
                [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
            )

        successful = [res for res in results_list if res is not None]
        if not successful:
            raise RuntimeError(
                "All MBDOE-PP optimisation runs failed. "
                "Try increasing 'maxpp', adjusting bounds, or relaxing constraints."
            )

        best_design_decisions, best_pp_obj, best_swps = max(successful, key=lambda x: x[1])

    else:
        try:
            best_design_decisions, best_pp_obj, best_swps = _run_single_pp(
                des_opt, system, models, core_number=0, round=round
            )
        except Exception as e:
            raise RuntimeError(f"Single-core optimisation failed: {e}")
        print("Design your experiment based on:")
        print("  tii   :", best_design_decisions['tii'])
        print("  tvi   :", best_design_decisions['tvi'])
        print("  swps  :", best_design_decisions['swps'])
        print("  St    :", best_design_decisions['St'])
        print("  pp_obj:", best_design_decisions['pp_obj'])
    return best_design_decisions



def _safe_run(args):
    des_opt, system, models, core_num, round = args
    configure_logger()
    try:
        return _run_single_pp(des_opt, system, models, core_num, round)
    except Exception as e:
        print(f"[Core {core_num}] FAILED: {e}")
        traceback.print_exc()
        return None


def _run_single_pp(des_opt, system, models, core_number=0, round=round):
    # --------------------------- EXTRACT DATA --------------------------- #
    tf = system['t_s'][1]
    ti = system['t_d']

    # Time-variant inputs
    tv_iphi_vars = list(system['tvi'].keys())
    tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
    tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
    tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
    tv_iphi_offsetl = [
        system['tvi'][var]['offl'] / system['tvi'][var]['max']
        for var in tv_iphi_vars
    ]
    tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}

    # Time-invariant inputs
    ti_iphi_vars = list(system['tii'].keys())
    ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]

    # Time-variant outputs
    tv_ophi_vars = [
        var for var in system['tvo'].keys()
        if system['tvo'][var].get('meas', True)
    ]
    tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
    tv_ophi_sampling = {var: system['tvo'][var]['samp_s'] for var in tv_ophi_vars}
    tv_ophi_forcedsamples = {var: [v / tf for v in system['tvo'][var]['samp_f']] for var in tv_ophi_vars}

    # Time-invariant outputs
    ti_ophi_vars = [
        var for var in system['tio'].keys()
        if system['tio'][var].get('meas', True)
    ]

    # Solver and optimization settings
    active_solvers = models['can_m']

    theta_parameters = models['theta']
    # Criterion, iteration, and penalty settings
    design_criteria = des_opt['pp_ob']
    maxpp = des_opt['itr']['maxpp']
    tolpp = des_opt['itr']['tolpp']
    population_size = des_opt['itr']['pps']
    eps = des_opt['eps']
    # Ensure V_matrix is a dict of solver -> NumPy array (for indexing safety)
    if 'V_matrix' in models:
        V_matrix = {
            solver: np.array(models['V_matrix'][solver])
            for solver in models['can_m']
        }
    else:
        V_matrix = {
            solver: np.array([
                [1e5 if i == j else 0 for j in range(len(models['theta'][solver]))]
                for i in range(len(models['theta'][solver]))
            ])
            for solver in models['can_m']
        }

    # Ensure mutation is a dict of solver -> list[bool]
    if 'mutation' in models:
        mutation = models['mutation']
    else:
        mutation = {
            solver: [True] * len(models['theta'][solver])
            for solver in models['can_m']
        }
    pltshow= des_opt['plt']
    optmethod = des_opt.get('meth', 'L')

    # ------------------------ CHOOSE METHOD (Local vs Global) ------------------------ #

    result, index_dict = _optimiser(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
        ti_ophi_vars,
        tf, ti,
        active_solvers, theta_parameters,
        eps, maxpp, tolpp,population_size,
        mutation, V_matrix, design_criteria,
        system, models, optmethod
    )

    x_final = getattr(result, 'X', getattr(result, 'x', None))
    if x_final is None:
        raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
    # ------------------- USE FINAL SOLUTION IN _runnerpp ---------------- #
    try:
        phi, swps, St, pp_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _pp_runner(
            x_final,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            system,
            models
        )
    except ValueError as e:
        if "MBDoE optimiser kernel was unsuccessful" in str(e):
            print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
            return None
        else:
            raise  # propagate other ValueErrors if unrelated

    # --------------------------- REPORT & SAVE --------------------------- #

    # You may have a specialized reporter or reuse the same `_reporter`
    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        pp_obj,                  # pass the objective for logging
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf,
        design_criteria,
        round, pltshow,
        core_number
    )

    # ------------------------- RETURN FINAL DATA ------------------------- #
    design_decisions = {
        'tii': phi,
        'tvi': phit,
        'swps': swps,
        'St': St,
        'pp_obj': pp_obj,
        't_values': t_values
    }

    return design_decisions, pp_obj, swps


def _optimiser(
    tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
    tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
    ti_iphi_vars, ti_iphi_max, ti_iphi_min,
    tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
    ti_ophi_vars,
    tf, ti,
    active_models, theta_parameters,
    eps, maxpp, tolpp,population_size,
    mutation, V_matrix, design_criteria,
    system, models, optmethod
):

    bounds = []
    x0 = []
    index_dict = {
        'ti': {},
        'swps': {},
        'st': {}
    }

    for i, (name, mn, mx) in enumerate(zip(ti_iphi_vars, ti_iphi_min, ti_iphi_max)):
        lo, hi = mn / mx, 1.0
        bounds.append((lo, hi))
        x0.append((lo + hi) / 2)
        index_dict['ti'][name] = [i]

    start = len(x0)
    for i, name in enumerate(tv_iphi_vars):
        seg = tv_iphi_seg[i]
        mn, mx = tv_iphi_min[i], tv_iphi_max[i]
        lo = mn / mx
        level_idxs = list(range(start, start + seg - 1))
        index_dict['swps'][name + 'l'] = level_idxs
        for _ in range(seg - 1):
            bounds.append((lo, 1.0))
            x0.append((lo + 1.0) / 2)
        start += seg - 1

    for i, name in enumerate(tv_iphi_vars):
        seg = tv_iphi_seg[i]
        lo, hi = ti / tf, 1 - ti / tf
        time_idxs = list(range(start, start + seg - 2))
        index_dict['swps'][name + 't'] = time_idxs
        for _ in range(seg - 2):
            bounds.append((lo, hi))
            x0.append((lo + hi) / 2)
        start += seg - 2

    sampling_groups = defaultdict(list)
    for var in tv_ophi_vars:
        group_id = tv_ophi_sampling[var]
        sampling_groups[group_id].append(var)

    for group_id, group_vars in sampling_groups.items():
        var = group_vars[0]
        i = tv_ophi_vars.index(var)
        seg = tv_ophi_seg[i]
        num_forced = len(tv_ophi_forcedsamples[var])
        num_free = seg - num_forced
        lo, hi = ti / tf, 1 - ti / tf

        idxs = list(range(start, start + num_free))
        for var_in_group in group_vars:
            index_dict['st'][var_in_group] = idxs

        for _ in range(num_free):
            bounds.append((lo, hi))
            x0.append((lo + hi) / 2)
        start += num_free

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    x0 = np.array(x0)

    constraint_index_list = []
    for i, name in enumerate(tv_iphi_vars):
        const = tv_iphi_const[i]
        if const != 'rel':
            idxs = index_dict['swps'][name + 'l']
            for j in range(len(idxs) - 1):
                constraint_index_list.append(('lvl', i, idxs[j], idxs[j + 1]))

    for i, name in enumerate(tv_iphi_vars):
        idxs = index_dict['swps'][name + 't']
        for j in range(len(idxs) - 1):
            constraint_index_list.append(('t', i, idxs[j], idxs[j + 1]))

    for i, name in enumerate(tv_ophi_vars):
        idxs = index_dict['st'][name]
        for j in range(len(idxs) - 1):
            constraint_index_list.append(('st', i, idxs[j], idxs[j + 1]))

    total_constraints = len(constraint_index_list)

    local_obj = partial(
        _pp_of,
        tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
        ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
        tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
        tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
        tv_ophi_sampling=tv_ophi_sampling,
        active_solvers=active_models, theta_parameters=theta_parameters,
        tf=tf, eps=eps, mutation=mutation,
        V_matrix=V_matrix, design_criteria=design_criteria,
        index_dict=index_dict, system=system, models=models
    )

    class DEProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(bounds),
                n_obj=1,
                n_constr=total_constraints,
                xl=lower,
                xu=upper
            )

        def _evaluate(self, x, out, *args, **kwargs):
            f_val = local_obj(x)
            g = []
            for kind, i, i1, i2 in constraint_index_list:
                if kind == 'lvl':
                    offs = tv_iphi_offsetl[i]
                    const = tv_iphi_const[i]
                    diff = x[i2] - x[i1] if const == 'inc' else x[i1] - x[i2]
                    g.append(offs - diff)
                elif kind == 't':
                    offs = tv_iphi_offsett[i]
                    g.append(offs - (x[i2] - x[i1]))
                elif kind == 'st':
                    offs = tv_ophi_offsett_ophi[i]
                    g.append(offs - (x[i2] - x[i1]))
            out['F'] = f_val
            out['G'] = np.array(g, dtype=np.float64)

    problem = DEProblem()

    if optmethod == 'PS':
        algorithm = PatternSearch()
        res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxpp), seed=None, verbose=True)

    elif optmethod == 'DE':
        algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
        res_refine = pymoo_minimize(
            problem,
            algorithm,
            termination=('n_gen', maxpp),
            seed=None,
            verbose=True,
            constraint_tolerance=tolpp,
            save_history=True
        )

    elif optmethod == 'DEPS':
        algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
        res_de = pymoo_minimize(
            problem,
            algorithm_de,
            termination=('n_gen', maxpp/2),
            seed=None,
            verbose=True,
            constraint_tolerance=tolpp,
            save_history=True
        )
        algorithm_refine = PatternSearch()
        res_refine = pymoo_minimize(
            problem,
            algorithm_refine,
            termination=('n_gen', int(maxpp)),
            seed=None,
            verbose=True,
            constraint_tolerance=tolpp,
            x0=res_de.X
        )

    else:
        raise ValueError("optmethod must be 'L', 'G', or 'GL'")

    return res_refine, index_dict


def _pp_of(
    x,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
    tf, eps, mutation, V_matrix, design_criteria,
    index_dict,
    system,
    models
):

    """
    Objective function wrapper for MBDOE-PP.
    """
    try:
        _, _, _, pp_obj, _, _, _, _ = _pp_runner(
            x,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            system,
            models
        )
        if np.isnan(pp_obj):
            return 1e6
        return -pp_obj  # negative => maximize pp_obj
    except Exception as e:
        print(f"Exception in pp_objective_function: {e}")
        return 1e6



def _pp_runner(
    x,
    tv_iphi_vars,
    tv_iphi_max,
    ti_iphi_vars,
    ti_iphi_max,
    tv_ophi_vars,
    ti_ophi_vars,
    active_solvers,
    theta_parameters,
    tv_iphi_cvp,
    tv_ophi_forcedsamples,
    tv_ophi_sampling,
    tf,
    eps,
    mutation,
    V_matrix,
    MBDOE_PP_criterion,
    index_dict,
    system,
    models
):
    """
    Simulate models and evaluate the PP objective.
    Returns the sliced inputs, objective value, time vector, and outputs.
    """
    if x is None:
        raise ValueError("MBDoE optimiser kernel was unsuccessful, increase iteration to avoid constraint violations.")
    x = x.tolist() if not isinstance(x, list) else x

    dt_real = system['t_r']
    nodes = int(round(tf / dt_real)) + 1
    tlin = np.linspace(0, 1, nodes)
    ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
    St = {var: np.array(sorted(St[var])) for var in St}
    t_values_flat = [tp for times in St.values() for tp in times]
    t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

    LSA = defaultdict(lambda: defaultdict(dict))
    J_dot_matrix = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                                                 len(theta_parameters[active_solvers[0]]))))
    indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars if var in St}
    M0 = defaultdict(lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                                       len(theta_parameters[active_solvers[0]]))))
    M = {}
    tv_ophi, ti_ophi, phit_interp = {}, {}, {}

    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        theta = np.array([1.0] * len(thetac))
        ti_iphi_data = ti
        swps_data = swps
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf

        tv_out, ti_out, interp_data = simula(t_values, swps_data, ti_iphi_data,
                                             phisc, phitsc, tsc,
                                             theta, thetac,
                                             tv_iphi_cvp, {}, solver_name,
                                             system, models)
        tv_ophi[solver_name] = tv_out
        ti_ophi[solver_name] = ti_out
        phit_interp = interp_data

        y_values_dict = {var: np.array(tv_out[var]) for var in tv_ophi_vars}
        y_values_dict.update({var: np.array([ti_out[var]]) for var in ti_ophi_vars})
        y_combined = dict(y_values_dict)

        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
        y_i_values_dict = {}

        for para_idx in free_params_indices:
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            tv_out_mod, ti_out_mod, _ = simula(t_values, swps_data, ti_iphi_data,
                                               phisc, phitsc, tsc,
                                               modified_theta, thetac,
                                               tv_iphi_cvp, {}, solver_name,
                                               system, models)

            y_i_values_dict = {var: np.array(tv_out_mod[var]) for var in tv_ophi_vars}
            y_i_values_dict.update({var: np.array([ti_out_mod[var]]) for var in ti_ophi_vars})
            y_modified_combined = dict(y_i_values_dict)

            for var in indices:
                LSA[var][solver_name][para_idx] = (
                    y_modified_combined[var] - y_combined[var]
                ) / eps

        solver_LSA_list = []
        for var in indices:
            row = [LSA[var][solver_name][i].mean() for i in free_params_indices]
            solver_LSA_list.append(np.array(row))
        solver_LSA = np.array(solver_LSA_list)

        # Build variance-covariance matrix Sigma_y
        std_dev = {}
        for grp, keys in [('tvo', tv_ophi_vars), ('tio', ti_ophi_vars)]:
            for var in keys:
                unc = system.get(grp, {}).get(var, {}).get('unc', 1.0)
                if unc is None or np.isnan(unc):
                    unc = 1.0
                std_dev[var] = unc

        vars_order = list(indices.keys())
        N_r = len(vars_order)
        Sigma_y = np.zeros((N_r, N_r))
        for i, var in enumerate(vars_order):
            Sigma_y[i, i] = std_dev.get(var, 1.0) ** 2
        Sigma_y_inv = np.linalg.inv(Sigma_y)

        # Fisher Information Matrix
        J = solver_LSA
        M_fisher = J.T @ Sigma_y_inv @ J

        if J_dot_matrix[solver_name].shape != M_fisher.shape:
            J_dot_matrix[solver_name] = np.zeros_like(M_fisher)
        J_dot_matrix[solver_name] += M_fisher
        M0[solver_name] = J_dot_matrix[solver_name]

        reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
        M[solver_name] = M0[solver_name] + np.linalg.inv(reduced_V_matrix)

    # Evaluate criterion
    solver_name = active_solvers[0]
    if MBDOE_PP_criterion == 'D':
        pp_obj = np.linalg.det(M[solver_name])
    elif MBDOE_PP_criterion == 'A':
        pp_obj = np.trace(M[solver_name])
    elif MBDOE_PP_criterion == 'E':
        pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))
    elif MBDOE_PP_criterion == 'ME':
        pp_obj = -np.linalg.cond(M[solver_name])
    elif MBDOE_PP_criterion == 'orb':
        # Residual norm–based ODR objective
        Z = solver_LSA
        m, n = Z.shape
        remaining = list(range(n))
        residual_norms_all = np.zeros(n)
        Q = np.zeros((m, 0))

        for _ in range(n):
            norms = []
            for j in remaining:
                v = Z[:, j]
                if Q.shape[1] > 0:
                    proj = Q @ (Q.T @ v)
                    r = v - proj
                else:
                    r = v
                norms.append(np.linalg.norm(r))

            best_idx = int(np.argmax(norms))
            j_sel = remaining.pop(best_idx)
            residual_norms_all[j_sel] = norms[best_idx]

            v = Z[:, j_sel]
            if Q.shape[1] > 0:
                v = v - Q @ (Q.T @ v)
            norm = np.linalg.norm(v)
            if norm > 0:
                q = v / norm
                Q = np.hstack((Q, q[:, None]))

        max_norm = np.max(residual_norms_all)
        normalized_norms = residual_norms_all / max_norm if max_norm > 0 else residual_norms_all

        tau = 0.1
        alpha = 1.5
        epsilon = 1e-6
        weights = np.ones(n)  # customize as needed
        target_params = [idx for idx, val in enumerate(normalized_norms) if val <= tau]
        if not target_params:
            target_params = [np.argmin(normalized_norms)]

        J_u = 0.0
        for w, j in zip(weights[target_params], target_params):
            J_u += w / (normalized_norms[j] ** 2 + epsilon) ** alpha

        pp_obj = J_u

        print(f"ORB residual norms: {normalized_norms}")
        print(f"ORB target params: {target_params}")
        print(f"ORB Objective: {pp_obj}")
    else:
        pp_obj = None

    return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
