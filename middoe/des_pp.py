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
# import logging
# import scipy.linalg
# import pandas as pd
# from pathlib import Path
#
# logger = logging.getLogger(__name__)
#
#
# def mbdoe_pp(
#     des_opt: Dict[str, Any],
#     system: Dict[str, Any],
#     models: Dict[str, Any],
#     round: int,
#     num_parallel_runs: int = 1
# ) -> Dict[str, Union[Dict[str, Any], float]]:
#     """
#     Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).
#
#     This function searches for the most informative experimental design to
#     improve parameter estimation accuracy by maximising a precision criterion
#     based on the Fisher Information Matrix (FIM). Optimisation can be run
#     in single-core or parallel mode, and the best valid design is returned.
#
#     Parameters
#     ----------
#     des_opt : dict
#         Optimisation settings, including:
#             - 'criteria' : dict[str, str]
#                 Precision criterion (e.g., {'MBDOE_PP_criterion': 'D'}).
#             - 'iteration_settings' : dict
#                 Iteration controls:
#                     * 'maxpp': int — maximum iterations
#                     * 'tolpp': float — convergence tolerance
#             - 'eps' : float
#                 Perturbation step size for numerical derivatives.
#
#     system : dict
#         System model and constraints:
#             - 'tf', 'ti' : float
#                 Final and initial times for simulation.
#             - 'tv_iphi', 'ti_iphi' : dict
#                 Time-variant and time-invariant input definitions.
#             - 'tv_ophi', 'ti_ophi' : dict
#                 Output definitions (time-variant/invariant).
#             - 'tv_iphi_seg' : dict[str, int]
#                 Control variable segmentation (CVP).
#             - 'tv_iphi_offsett', 'tv_iphi_offsetl' : dict
#                 Time/level offsets for switching logic.
#             - 'tv_ophi_matching' : dict[str, bool]
#                 Synchronised sampling across outputs.
#             - 'sampling_constraints' : dict
#                 Sampling restrictions:
#                     * 'min_interval': float — minimum time between samples
#                     * 'max_samples_per_response': int
#                     * 'total_samples': int
#                     * 'forbidden_times': dict[str, list[float]]
#                     * 'forced_times': dict[str, list[float]]
#                     * 'sync_sampling': bool
#
#     models : dict
#         Model definitions and settings:
#             - 'active_models' : list[str]
#                 Models to evaluate.
#             - 'theta_parameters' : dict[str, list[float]]
#                 Nominal parameter values.
#             - 'mutation' : dict[str, list[bool]]
#                 Parameter masks for optimisation.
#             - 'V_matrix' : dict[str, ndarray]
#                 Prior covariance matrices.
#             - 'ext_models' : dict[str, Callable]
#                 Mapping of model names to simulator functions.
#
#     round : int
#         Current experimental design round (used for logging/conditional logic).
#
#     num_parallel_runs : int, optional
#         Number of optimisation runs in parallel (default: 1, single-core mode).
#
#     Returns
#     -------
#     results : dict
#         Best design found:
#             - 'x' : dict[str, Any]
#                 Optimal design variables (e.g., sampling times, control profiles).
#             - 'fun' : float
#                 Objective value (e.g., log det(FIM)).
#             - 'swps' : dict[str, list[float]]
#                 Switching times for time-variant controls.
#
#     Raises
#     ------
#     RuntimeError
#         If all parallel runs fail or if the single-core run encounters an exception.
#
#     Notes
#     -----
#     MBDoE-PP designs experiments that maximise information for parameter estimation,
#     using precision-based criteria such as:
#         - D-optimality (maximise log det(FIM))
#         - A-optimality (minimise trace(FIM⁻¹))
#         - E-optimality (maximise smallest eigenvalue of FIM)
#
#     Supported constraints include:
#         - Control level bounds and switching rules
#         - Minimum sampling intervals
#         - Total/maximum samples per output
#         - Forced or forbidden sampling times
#         - Synchronised sampling across outputs
#
#     References
#     ----------
#     Franceschini, G., & Macchietto, S. (2008).
#     Model-based design of experiments for parameter precision: State of the art.
#     Chemical Engineering Science, 63(19), 4846–4872.
#     https://doi.org/10.1016/j.ces.2008.07.006
#
#     See Also
#     --------
#     _safe_run : Parallel optimisation wrapper.
#     _run_single_pp : Single-core optimisation routine.
#
#     Examples
#     --------
#     >>> result = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=4)
#     >>> print("Best objective value:", result['fun'])
#     >>> print("Optimal design:", result['x'])
#     """
#     # Assuming active solvers and mutation info available
#     active_solvers = models['can_m']
#     solver_name = active_solvers[0] if active_solvers else None
#
#     if solver_name and 'LSA' in models and solver_name in models['LSA']:
#         Z_init = models['LSA'][solver_name]
#         initial_ranking = parameter_ranking(Z_init)
#         logger.debug(f"Initial parameter ranking (most to least estimable) for solver '{solver_name}': {initial_ranking}")
#         des_opt['initial_ranking'] = initial_ranking
#     else:
#         logger.debug(f"No initial LSA matrix found for solver '{solver_name}' to compute initial ranking.")
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
#
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
# def _run_single_pp(des_opt, system, models, core_number=0, round=round):
#     # --------------------------- EXTRACT DATA --------------------------- #
#     tf = system['t_s'][1]
#     ti = system['t_d']
#
#     # Time-variant inputs
#     tv_iphi_vars = list(system['tvi'].keys())
#     tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
#     tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
#     tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
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
#
#     theta_parameters = models['theta']
#     # Criterion, iteration, and penalty settings
#     design_criteria = des_opt['pp_ob']
#     maxpp = des_opt['itr']['maxpp']
#     tolpp = des_opt['itr']['tolpp']
#     population_size = des_opt['itr']['pps']
#     eps = des_opt['eps']
#     # Ensure V_matrix is a dict of solver -> NumPy array (for indexing safety)
#     if 'V_matrix' in models:
#         V_matrix = {
#             solver: np.array(models['V_matrix'][solver])
#             for solver in models['can_m']
#         }
#     else:
#         V_matrix = {
#             solver: np.array([
#                 [1e5 if i == j else 0 for j in range(len(models['theta'][solver]))]
#                 for i in range(len(models['theta'][solver]))
#             ])
#             for solver in models['can_m']
#         }
#
#     # Ensure mutation is a dict of solver -> list[bool]
#     if 'mutation' in models:
#         mutation = models['mutation']
#     else:
#         mutation = {
#             solver: [True] * len(models['theta'][solver])
#             for solver in models['can_m']
#         }
#     pltshow= des_opt['plt']
#     optmethod = des_opt.get('meth', 'L')
#
#     # ------------------------ CHOOSE METHOD (Local vs Global) ------------------------ #
#
#     result, index_dict = _optimiser(
#         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#         ti_ophi_vars,
#         tf, ti,
#         active_solvers, theta_parameters,
#         eps, maxpp, tolpp,population_size,
#         mutation, V_matrix, design_criteria,
#         system, models, optmethod
#     )
#
#     x_final = getattr(result, 'X', getattr(result, 'x', None))
#     if x_final is None:
#         raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
#     # ------------------- USE FINAL SOLUTION IN _runnerpp ---------------- #
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
#             system,
#             models
#         )
#     except ValueError as e:
#         if "MBDoE optimiser kernel was unsuccessful" in str(e):
#             print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
#             return None
#         else:
#             raise  # propagate other ValueErrors if unrelated
#
#     # --------------------------- REPORT & SAVE --------------------------- #
#
#     # You may have a specialized reporter or reuse the same `_reporter`
#     phi, phit, swps, St = _reporter(
#         phi, phit, swps, St,
#         pp_obj,                  # pass the objective for logging
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
#     tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#     tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#     ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#     tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#     ti_ophi_vars,
#     tf, ti,
#     active_models, theta_parameters,
#     eps, maxpp, tolpp,population_size,
#     mutation, V_matrix, design_criteria,
#     system, models, optmethod
# ):
#
#     bounds = []
#     x0 = []
#     index_dict = {
#         'ti': {},
#         'swps': {},
#         'st': {}
#     }
#
#     for i, (name, mn, mx) in enumerate(zip(ti_iphi_vars, ti_iphi_min, ti_iphi_max)):
#         lo, hi = mn / mx, 1.0
#         bounds.append((lo, hi))
#         x0.append((lo + hi) / 2)
#         index_dict['ti'][name] = [i]
#
#     start = len(x0)
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         mn, mx = tv_iphi_min[i], tv_iphi_max[i]
#         lo = mn / mx
#         level_idxs = list(range(start, start + seg - 1))
#         index_dict['swps'][name + 'l'] = level_idxs
#         for _ in range(seg - 1):
#             bounds.append((lo, 1.0))
#             x0.append((lo + 1.0) / 2)
#         start += seg - 1
#
#     for i, name in enumerate(tv_iphi_vars):
#         seg = tv_iphi_seg[i]
#         lo, hi = ti / tf, 1 - ti / tf
#         time_idxs = list(range(start, start + seg - 2))
#         index_dict['swps'][name + 't'] = time_idxs
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
#         idxs = list(range(start, start + num_free))
#         for var_in_group in group_vars:
#             index_dict['st'][var_in_group] = idxs
#
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
#         index_dict=index_dict, system=system, models=models
#     )
#     from pymoo.core.problem import ElementwiseProblem, LoopedElementwiseEvaluation
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
#             termination=('n_gen', maxpp/2),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolpp,
#             save_history=True
#         )
#         algorithm_refine = PatternSearch()
#         res_refine = pymoo_minimize(
#             problem,
#             algorithm_refine,
#             termination=('n_gen', int(maxpp)),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolpp,
#             x0=res_de.X
#         )
#
#     else:
#         raise ValueError("optmethod must be 'L', 'G', or 'GL'")
#
#     return res_refine, index_dict
#
#
# def _pp_of(
#     x,
#     tv_iphi_vars, tv_iphi_max,
#     ti_iphi_vars, ti_iphi_max,
#     tv_ophi_vars, ti_ophi_vars,
#     active_solvers, theta_parameters,
#     tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#     tf, eps, mutation, V_matrix, design_criteria,
#     index_dict,
#     system,
#     models
# ):
#
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
#             system,
#             models
#         )
#         if np.isnan(pp_obj):
#             return 1e6
#         return -pp_obj  # negative => maximize pp_obj
#     except Exception as e:
#         print(f"\n{'=' * 60}")
#         print(f"Exception in pp_objective_function: {e}")
#         print(f"Full traceback:")
#         traceback.print_exc()  # ← THIS PRINTS THE FULL STACK TRACE
#         print(f"{'=' * 60}\n")
#         return 1e6
#
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
#
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
#         # Check if V_matrix is already reduced or full
#         if V_matrix[solver_name].shape[0] == len(free_params_indices):
#             reduced_V_matrix = V_matrix[solver_name]
#         elif V_matrix[solver_name].shape[0] == len(theta_parameters[solver_name]):
#             reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
#         else:
#             raise ValueError(
#                 f"V_matrix shape {V_matrix[solver_name].shape} doesn't match either "
#                 f"free params ({len(free_params_indices)}) or total params ({len(theta_parameters[solver_name])})"
#             )
#
#         # Mp = np.linalg.pinv(reduced_V_matrix)
#         # M[solver_name] = M0[solver_name] + Mp
#
#         # Regularise prior covariance before inverting
#         alpha = 1e-8
#         V_reg = reduced_V_matrix + alpha * np.eye(reduced_V_matrix.shape[0])
#         Mp = np.linalg.inv(V_reg)
#         M[solver_name] = M0[solver_name] + Mp
#
#     # ================================================================
#     # EVALUATE CRITERION
#     # ================================================================
#     solver_name = active_solvers[0]
#
#     if MBDOE_PP_criterion == 'D':
#         pp_obj = np.linalg.det(M[solver_name])
#
#     elif MBDOE_PP_criterion == 'A':
#         pp_obj = np.trace(M[solver_name])
#
#     elif MBDOE_PP_criterion == 'E':
#         pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))
#
#     elif MBDOE_PP_criterion == 'ME':
#         pp_obj = -np.linalg.cond(M[solver_name])
#     else:
#         pp_obj = None
#
#     return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
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
import scipy.linalg
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def mbdoe_pp(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    round: int,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    r"""
    Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).

    This function searches for the most informative experimental design to
    improve parameter estimation accuracy by maximising a precision criterion
    based on the Fisher Information Matrix (FIM). Optimisation can be run
    in single-core or parallel mode, and the best valid design is returned.

    Parameters
    ----------
    des_opt : dict
        Optimisation settings, including:
            - 'pp_ob' : str
                Precision criterion ('D', 'A', 'E', 'ME'):
                    * 'D': D-optimality (maximise det(FIM))
                    * 'A': A-optimality (maximise trace(FIM))
                    * 'E': E-optimality (maximise min eigenvalue)
                    * 'ME': Modified E-optimality (minimise condition number)
            - 'itr' : dict
                Iteration controls:
                    * 'maxpp': int — maximum iterations
                    * 'tolpp': float — convergence tolerance
                    * 'pps': int — population size
            - 'eps' : float
                Perturbation step size for numerical derivatives.
            - 'plt' : bool
                Enable plotting of results.
            - 'meth' : str, optional
                Optimisation method ('DE', 'PS', 'DEPS').
            - 'initial_ranking' : list[int], optional
                Pre-computed parameter ranking (set automatically if LSA available).

    system : dict
        System model and constraints:
            - 't_s' : tuple[float, float]
                Start and end times (ti, tf).
            - 't_d' : float
                Dead time / initial delay.
            - 't_r' : float
                Time resolution for simulation.
            - 'tvi' : dict
                Time-variant input definitions with keys:
                    * 'max', 'min': float — bounds
                    * 'stps': int — number of switching segments
                    * 'const': str — constraint type ('inc', 'dec', 'rel')
                    * 'offt': float — minimum time offset between switches
                    * 'offl': float — minimum level offset
                    * 'cvp': bool — control variable parameterisation flag
            - 'tii' : dict
                Time-invariant input definitions ('max', 'min').
            - 'tvo' : dict
                Time-variant output definitions:
                    * 'meas': bool — include in measurement
                    * 'sp': int — number of sampling points
                    * 'offt': float — minimum time between samples
                    * 'samp_s': str — sampling synchronisation group
                    * 'samp_f': list[float] — forced sampling times
                    * 'unc': float — measurement uncertainty (standard deviation)
            - 'tio' : dict
                Time-invariant output definitions.

    models : dict
        Model definitions and settings:
            - 'can_m' : list[str]
                Active model names (typically single model for PP).
            - 'theta' : dict[str, list[float]]
                Nominal parameter values for each model.
            - 'mutation' : dict[str, list[bool]]
                Parameter masks indicating which parameters are free (True) vs fixed (False).
            - 'V_matrix' : dict[str, np.ndarray]
                Prior parameter covariance matrices (typically diagonal).
            - 'LSA' : dict[str, np.ndarray], optional
                Pre-computed local sensitivity analysis matrices for initial ranking.

    round : int
        Current experimental design round (used for logging/reporting).

    num_parallel_runs : int, optional
        Number of optimisation runs in parallel (default: 1, single-core mode).

    Returns
    -------
    results : dict
        Best design found:
            - 'tii' : dict[str, float]
                Optimal time-invariant input values.
            - 'tvi' : dict[str, Any]
                Optimal time-variant input profiles.
            - 'swps' : dict[str, list[float]]
                Switching times for time-variant controls.
            - 'St' : dict[str, np.ndarray]
                Sampling times for each output variable.
            - 'pp_obj' : float
                Parameter precision objective value (maximised).
            - 't_values' : list[float]
                Full simulation time vector.

    Raises
    ------
    RuntimeError
        If all parallel runs fail or if the single-core run encounters an exception.

    Notes
    -----
    MBDoE-PP designs experiments that maximise information for parameter estimation,
    using precision-based criteria:

    - **D-optimality**: Maximises \( \det(\mathbf{M}) \), where \( \mathbf{M} \) is the
      Fisher Information Matrix. Minimises the volume of confidence ellipsoid.

    - **A-optimality**: Maximises \( \text{trace}(\mathbf{M}) \). Minimises average
      parameter variance.

    - **E-optimality**: Maximises \( \lambda_{\min}(\mathbf{M}) \). Minimises
      worst-case parameter variance (most conservative).

    - **ME-optimality**: Minimises \( \text{cond}(\mathbf{M}) \). Improves numerical
      conditioning and reduces correlation.

    The Fisher Information Matrix is computed as:

    \[
    \mathbf{M} = \mathbf{M}_0 + \mathbf{M}_p = \mathbf{Q}^T \mathbf{\Sigma}_y^{-1} \mathbf{Q} + \mathbf{V}^{-1}
    \]

    where \( \mathbf{Q} \) is the sensitivity matrix, \( \mathbf{\Sigma}_y \) is the
    measurement covariance, and \( \mathbf{V} \) is the prior parameter covariance.

    **Parameter Ranking**: If pre-computed LSA matrices are available, initial parameter
    estimability ranking is computed and logged.

    Supported constraints include:
        - Control level bounds and switching rules
        - Minimum sampling intervals
        - Total/maximum samples per output
        - Forced or forbidden sampling times
        - Synchronised sampling across outputs

    References
    ----------
    .. [1] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.
       https://doi.org/10.1016/j.ces.2008.07.006

    .. [2] Galvanin, F., Macchietto, S., & Bezzo, F. (2007).
       Model-based design of parallel experiments.
       *Industrial & Engineering Chemistry Research*, 46(3), 871–882.
       https://doi.org/10.1021/ie0611406

    See Also
    --------
    _safe_run : Parallel optimisation wrapper with exception handling.
    _run_single_pp : Single-core optimisation routine.
    _optimiser : Core optimisation problem setup and solver.
    parameter_ranking : Compute parameter estimability ranking from LSA.

    Examples
    --------
    >>> result = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=4)
    >>> print("Best objective value:", result['pp_obj'])
    >>> print("Optimal sampling times:", result['St'])
    >>> print("Parameter precision achieved")
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
    """
    Safely execute a single MBDoE-PP optimisation run with exception handling.

    This wrapper function is used for parallel execution to prevent individual
    core failures from crashing the entire optimisation. Exceptions are caught,
    logged, and returned as None. Logger is configured at the start of each run.

    Parameters
    ----------
    args : tuple
        Packed arguments for _run_single_pp:
            - des_opt : dict
                Design optimisation settings.
            - system : dict
                System configuration.
            - models : dict
                Model definitions.
            - core_num : int
                Core/process identifier for logging.
            - round : int
                Experimental design round number.

    Returns
    -------
    result : tuple or None
        If successful: (design_decisions, pp_obj, swps).
        If failed: None.

    Notes
    -----
    This function is intended for use with multiprocessing.Pool.map().
    Exception tracebacks are printed to stderr for debugging.

    See Also
    --------
    _run_single_pp : Core single-run optimisation logic.
    mbdoe_pp : Main entry point that manages parallel execution.
    """
    des_opt, system, models, core_num, round = args
    configure_logger()
    try:
        return _run_single_pp(des_opt, system, models, core_num, round)
    except Exception as e:
        print(f"[Core {core_num}] FAILED: {e}")
        traceback.print_exc()
        return None


def _run_single_pp(des_opt, system, models, core_number=0, round=round):
    """
    Execute a single MBDoE-PP optimisation run on one core.

    This function unpacks system and model configurations, constructs the
    optimisation problem for parameter precision, invokes the solver, and
    evaluates the final design.

    Parameters
    ----------
    des_opt : dict
        Design optimisation settings (see mbdoe_pp documentation).
    system : dict
        System configuration including inputs, outputs, and constraints.
    models : dict
        Model definitions including parameters, mutations, and prior covariances.
    core_number : int, optional
        Identifier for the current core/process (default: 0).
    round : int
        Current experimental design round.

    Returns
    -------
    design_decisions : dict
        Optimal design variables and objective value.
    pp_obj : float
        Parameter precision objective value (to be maximised).
    swps : dict
        Switching times for time-variant control variables.

    Raises
    ------
    ValueError
        If optimisation result lacks required attributes ('X' or 'x').
        If kernel simulation is infeasible for the final design.

    Notes
    -----
    This function performs the following steps:
        1. Extract and normalise system bounds and constraints.
        2. Build index mapping for decision variables.
        3. Construct default V_matrix (prior covariance) if not provided.
        4. Construct default mutation masks if not provided.
        5. Invoke _optimiser to solve the optimisation problem.
        6. Evaluate the final design using _pp_runner.
        7. Report results via _reporter.

    **Default Prior Covariance**: If V_matrix is not provided, a diagonal matrix
    with large variances (1e5) is constructed, representing uninformative priors.

    **Default Mutation Masks**: If not provided, all parameters are assumed free.

    See Also
    --------
    _optimiser : Optimisation problem construction and solver.
    _pp_runner : Simulation kernel for design evaluation.
    _reporter : Result formatting and visualisation.
    """
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
    eps, maxpp, tolpp, population_size,
    mutation, V_matrix, design_criteria,
    system, models, optmethod
):
    """
    Construct and solve the MBDoE-PP optimisation problem.

    This function builds the decision variable vector, bounds, constraints,
    and index mapping, then invokes a pymoo-based optimiser to maximise
    the parameter precision objective.

    Parameters
    ----------
    tv_iphi_vars : list[str]
        Names of time-variant input variables.
    tv_iphi_seg : list[int]
        Number of segments (switching points + 1) for each time-variant input.
    tv_iphi_max : list[float]
        Upper bounds for time-variant inputs.
    tv_iphi_min : list[float]
        Lower bounds for time-variant inputs.
    tv_iphi_const : list[str]
        Constraint types: 'inc' (increasing), 'dec' (decreasing), 'rel' (relaxed).
    tv_iphi_offsett : list[float]
        Normalised minimum time offset between switches (relative to tf).
    tv_iphi_offsetl : list[float]
        Normalised minimum level offset between switches (relative to max).
    tv_iphi_cvp : dict[str, bool]
        Control variable parameterisation flags.
    ti_iphi_vars : list[str]
        Names of time-invariant input variables.
    ti_iphi_max : list[float]
        Upper bounds for time-invariant inputs.
    ti_iphi_min : list[float]
        Lower bounds for time-invariant inputs.
    tv_ophi_vars : list[str]
        Names of measurable time-variant output variables.
    tv_ophi_seg : list[int]
        Number of sampling points for each output.
    tv_ophi_offsett_ophi : list[float]
        Normalised minimum time offset between output samples.
    tv_ophi_sampling : dict[str, str]
        Sampling group identifiers for synchronisation.
    tv_ophi_forcedsamples : dict[str, list[float]]
        Forced sampling times (normalised) for each output.
    ti_ophi_vars : list[str]
        Names of time-invariant output variables.
    tf : float
        Final simulation time.
    ti : float
        Initial simulation time / dead time.
    active_models : list[str]
        Names of active models (typically single model for PP).
    theta_parameters : dict[str, list[float]]
        Nominal parameter values for each model.
    eps : float
        Perturbation step for numerical derivatives.
    maxpp : int
        Maximum number of optimisation iterations/generations.
    tolpp : float
        Constraint tolerance for pymoo solvers.
    population_size : int
        Population size for evolutionary algorithms.
    mutation : dict[str, list[bool]]
        Free/fixed parameter masks for each model.
    V_matrix : dict[str, np.ndarray]
        Prior parameter covariance matrices.
    design_criteria : str
        Precision criterion ('D', 'A', 'E', 'ME').
    system : dict
        Full system configuration (passed to objective function).
    models : dict
        Full model definitions (passed to objective function).
    optmethod : str
        Optimisation method: 'PS' (Pattern Search), 'DE' (Differential Evolution),
        'DEPS' (DE followed by PS refinement).

    Returns
    -------
    result : pymoo.core.result.Result
        Optimisation result object containing optimal decision vector (X or x).
    index_dict : dict
        Mapping of variable names to decision vector indices:
            - 'ti' : dict[str, list[int]] — time-invariant inputs
            - 'swps' : dict[str, list[int]] — switching times/levels
            - 'st' : dict[str, list[int]] — sampling times

    Raises
    ------
    ValueError
        If optmethod is not one of 'PS', 'DE', 'DEPS'.

    Notes
    -----
    Decision variable ordering:
        1. Time-invariant input levels (normalised)
        2. Time-variant input levels for each segment
        3. Switching times between segments
        4. Sampling times for outputs (grouped by synchronisation)

    Constraints enforce:
        - Monotonic increase/decrease for control profiles
        - Minimum time gaps between switches and samples
        - Sampling time ordering within groups

    The objective function is negated for maximisation (pymoo minimises by default).

    See Also
    --------
    _pp_of : Objective function wrapper for evaluation.
    _pp_runner : Simulation kernel invoked within objective.
    """
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
    from pymoo.core.problem import ElementwiseProblem, LoopedElementwiseEvaluation
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
    Objective function wrapper for MBDoE-PP optimisation.

    This function evaluates the parameter precision objective for a given
    design vector by invoking the simulation kernel (_pp_runner). Exceptions
    and NaN values are handled by returning a large penalty value.

    Parameters
    ----------
    x : np.ndarray or list
        Decision variable vector (normalised).
    tv_iphi_vars : list[str]
        Time-variant input variable names.
    tv_iphi_max : list[float]
        Upper bounds for time-variant inputs.
    ti_iphi_vars : list[str]
        Time-invariant input variable names.
    ti_iphi_max : list[float]
        Upper bounds for time-invariant inputs.
    tv_ophi_vars : list[str]
        Time-variant output variable names.
    ti_ophi_vars : list[str]
        Time-invariant output variable names.
    active_solvers : list[str]
        Active model names (typically single model for PP).
    theta_parameters : dict[str, list[float]]
        Nominal parameter vectors for each model.
    tv_iphi_cvp : dict[str, bool]
        Control variable parameterisation flags.
    tv_ophi_forcedsamples : dict[str, list[float]]
        Forced sampling times (normalised).
    tv_ophi_sampling : dict[str, str]
        Sampling group identifiers.
    tf : float
        Final simulation time.
    eps : float
        Perturbation step for sensitivity computation.
    mutation : dict[str, list[bool]]
        Free/fixed parameter masks.
    V_matrix : dict[str, np.ndarray]
        Prior parameter covariance matrices.
    design_criteria : str
        Precision criterion ('D', 'A', 'E', 'ME').
    index_dict : dict
        Mapping of variable names to decision vector indices.
    system : dict
        Full system configuration.
    models : dict
        Full model definitions.

    Returns
    -------
    obj_value : float
        Negative of the precision objective (for minimisation).
        Returns 1e6 if evaluation fails or results in NaN.

    Notes
    -----
    The objective is negated because pymoo minimises by default, while
    parameter precision seeks to maximise information content (det, trace, eigenvalue).

    Full exception tracebacks are printed to stderr for debugging purposes.

    See Also
    --------
    _pp_runner : Simulation kernel for model evaluation.
    _optimiser : Optimisation problem setup.
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
        print(f"\n{'=' * 60}")
        print(f"Exception in pp_objective_function: {e}")
        print(f"Full traceback:")
        traceback.print_exc()  # ← THIS PRINTS THE FULL STACK TRACE
        print(f"{'=' * 60}\n")
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
    r"""
    Simulate models and evaluate the parameter precision objective.

    This function decodes the decision vector, constructs time grids, invokes
    the simulator for nominal and perturbed parameters, computes local sensitivity
    matrices, builds the Fisher Information Matrix (FIM), and calculates the
    precision metric (D, A, E, or ME-optimality).

    Parameters
    ----------
    x : np.ndarray or list
        Decision variable vector (normalised).
    tv_iphi_vars : list[str]
        Time-variant input variable names.
    tv_iphi_max : list[float]
        Upper bounds for time-variant inputs.
    ti_iphi_vars : list[str]
        Time-invariant input variable names.
    ti_iphi_max : list[float]
        Upper bounds for time-invariant inputs.
    tv_ophi_vars : list[str]
        Measurable time-variant output variable names.
    ti_ophi_vars : list[str]
        Measurable time-invariant output variable names.
    active_solvers : list[str]
        Active model names (typically single model for PP).
    theta_parameters : dict[str, list[float]]
        Nominal parameter values for each model.
    tv_iphi_cvp : dict[str, bool]
        Control variable parameterisation flags.
    tv_ophi_forcedsamples : dict[str, list[float]]
        Forced sampling times (normalised).
    tv_ophi_sampling : dict[str, str]
        Sampling group identifiers.
    tf : float
        Final simulation time.
    eps : float
        Perturbation step for numerical sensitivity (forward differences).
    mutation : dict[str, list[bool]]
        Free/fixed parameter masks for each model.
    V_matrix : dict[str, np.ndarray]
        Prior parameter covariance matrices.
    MBDOE_PP_criterion : str
        Precision criterion:
            - 'D': D-optimality (maximise det(FIM))
            - 'A': A-optimality (maximise trace(FIM))
            - 'E': E-optimality (maximise min eigenvalue)
            - 'ME': Modified E-optimality (minimise condition number)
    index_dict : dict
        Mapping of variable names to decision vector indices.
    system : dict
        Full system configuration (passed to simula).
    models : dict
        Full model definitions (passed to simula).

    Returns
    -------
    ti : dict[str, float]
        Time-invariant input values (denormalised).
    swps : dict[str, list[float]]
        Switching times and levels for time-variant inputs (denormalised).
    St : dict[str, np.ndarray]
        Sampling times for each output (denormalised).
    pp_obj : float
        Parameter precision objective value (positive, to be maximised).
    t_values : list[float]
        Full simulation time vector (denormalised).
    tv_ophi : dict[str, dict[str, np.ndarray]]
        Time-variant outputs for each model and variable.
    ti_ophi : dict[str, dict[str, float]]
        Time-invariant outputs for each model and variable.
    phit_interp : dict
        Interpolation data for time-variant inputs.

    Raises
    ------
    ValueError
        If decision vector is None (optimiser failure).
        If V_matrix shape does not match parameter dimensions.

    Notes
    -----
    **Fisher Information Matrix (FIM):**

    The FIM is computed as:

    \[
    \mathbf{M} = \mathbf{M}_0 + \mathbf{M}_p = \mathbf{Q}^T \mathbf{\Sigma}_y^{-1} \mathbf{Q} + \mathbf{V}^{-1}
    \]

    where:
        - \( \mathbf{Q} \): Sensitivity matrix (\( \partial y / \partial \theta \))
        - \( \mathbf{\Sigma}_y \): Measurement covariance (diagonal)
        - \( \mathbf{V} \): Prior parameter covariance

    **Precision Criteria:**

    - **D-optimality**: \( \det(\mathbf{M}) \) — volume of confidence ellipsoid
    - **A-optimality**: \( \text{trace}(\mathbf{M}) \) — average parameter variance
    - **E-optimality**: \( \lambda_{\min}(\mathbf{M}) \) — worst-case variance
    - **ME-optimality**: \( -\text{cond}(\mathbf{M}) \) — numerical conditioning

    **Regularisation**: Prior covariance is regularised with \( \alpha = 10^{-8} \)
    before inversion to avoid numerical instability.

    The function automatically handles:
        - Reduced vs full-size V_matrix (extracts free parameters)
        - Missing uncertainty specifications (defaults to 1.0)
        - Mixed time-variant and time-invariant outputs

    See Also
    --------
    simula : Core simulation kernel for model evaluation.
    _slicer : Decodes decision vector into design variables.
    _pp_of : Objective function wrapper.
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

        # Check if V_matrix is already reduced or full
        if V_matrix[solver_name].shape[0] == len(free_params_indices):
            reduced_V_matrix = V_matrix[solver_name]
        elif V_matrix[solver_name].shape[0] == len(theta_parameters[solver_name]):
            reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
        else:
            raise ValueError(
                f"V_matrix shape {V_matrix[solver_name].shape} doesn't match either "
                f"free params ({len(free_params_indices)}) or total params ({len(theta_parameters[solver_name])})"
            )

        # Mp = np.linalg.pinv(reduced_V_matrix)
        # M[solver_name] = M0[solver_name] + Mp

        # Regularise prior covariance before inverting
        alpha = 1e-8
        V_reg = reduced_V_matrix + alpha * np.eye(reduced_V_matrix.shape[0])
        Mp = np.linalg.inv(V_reg)
        M[solver_name] = M0[solver_name] + Mp

    # ================================================================
    # EVALUATE CRITERION
    # ================================================================
    solver_name = active_solvers[0]

    if MBDOE_PP_criterion == 'D':
        pp_obj = np.linalg.det(M[solver_name])

    elif MBDOE_PP_criterion == 'A':
        pp_obj = np.trace(M[solver_name])

    elif MBDOE_PP_criterion == 'E':
        pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))

    elif MBDOE_PP_criterion == 'ME':
        pp_obj = -np.linalg.cond(M[solver_name])
    else:
        pp_obj = None

    return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp