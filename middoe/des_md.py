# from middoe.des_utils import _slicer, _reporter,  configure_logger
# from functools import partial
# from pymoo.algorithms.soo.nonconvex.de import DE
# from pymoo.operators.sampling.lhs import LHS
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize as pymoo_minimize
# from multiprocessing import Pool
# import traceback
# from middoe.krnl_simula import simula
# from collections import defaultdict
# import numpy as np
# from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# from typing import Dict, Any, Union
#
#
#
# def mbdoe_md(
#     des_opt: Dict[str, Any],
#     system: Dict[str, Any],
#     models: Dict[str, Any],
#     round: int,
#     num_parallel_runs: int = 1
# ) -> Dict[str, Union[Dict[str, Any], float]]:
#     """
#     Perform Model-Based Design of Experiments for Model Discrimination (MBDoE-MD).
#
#     This function identifies experimental designs that maximise the ability to
#     discriminate between competing models. Optimisation can run in parallel or
#     single-core mode. Failed runs are discarded, and the best valid design is returned.
#
#     Parameters
#     ----------
#     des_opt : dict
#         Optimisation settings, including:
#             - 'criteria' : dict[str, str]
#                 Discrimination criterion (e.g., {'MBDOE_MD_criterion': 'T'}).
#             - 'iteration_settings' : dict
#                 Iteration controls:
#                     * 'maxmd': int — maximum iterations
#                     * 'tolmd': float — convergence tolerance
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
#                 Competing models to discriminate.
#             - 'theta_parameters' : dict[str, list[float]]
#                 Nominal parameter values for each model.
#             - 'mutation' : dict[str, list[bool]]
#                 Parameter masks for optimisation.
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
#             - 'tii' : dict[str, Any]
#                 Time-invariant input variables.
#             - 'tvi' : dict[str, Any]
#                 Time-variant input variables.
#             - 'swps' : dict[str, list[float]]
#                 Switching times for time-variant controls.
#             - 'St' : list[float]
#                 Sampling times.
#             - 'md_obj' : float
#                 Model discrimination objective value.
#
#     Raises
#     ------
#     RuntimeError
#         If all parallel runs fail or if the single-core run encounters an exception.
#
#     Notes
#     -----
#     MBDoE-MD aims to design experiments that enhance the ability to reject
#     incorrect models by maximising discrimination criteria such as:
#         - T-optimality (maximise pairwise model separation)
#         - HR (maximise weighted Kullback–Leibler divergence)
#         - BFF (Bayesian/Frequentist hybrid measures)
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
#     Chen, C., & Asprey, S. P. (2003).
#     On the design of optimally informative dynamic experiments for model discrimination
#     in multiresponse nonlinear situations.
#     Industrial & Engineering Chemistry Research, 42(6), 1379–1390.
#     https://doi.org/10.1021/ie020468o
#
#     See Also
#     --------
#     _safe_run_md : Parallel optimisation wrapper.
#     _run_single_md : Single-core optimisation routine.
#
#     Examples
#     --------
#     >>> result = mbdoe_md(des_opt, system, models, round=1, num_parallel_runs=4)
#     >>> print("Best objective value:", result['md_obj'])
#     >>> print("Optimal design:", result['tii'], result['tvi'])
#     """
#     if num_parallel_runs > 1:
#         with Pool(num_parallel_runs) as pool:
#             results_list = pool.map(
#                 _safe_run_md,
#                 [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
#             )
#
#         successful = [res for res in results_list if res is not None]
#         if not successful:
#             raise RuntimeError(
#                 "All MBDoE-MD optimisation runs failed. "
#                 "Try increasing 'maxmd', adjusting bounds, or relaxing constraints."
#             )
#
#         best_design_decisions, best_md_obj, best_swps = max(successful, key=lambda x: x[1])
#
#     else:
#         try:
#             best_design_decisions, best_md_obj, best_swps = _run_single_md(
#                 des_opt, system, models, core_number=0, round=round
#             )
#         except Exception as e:
#             raise RuntimeError(f"Single-core optimisation failed: {e}")
#
#         print("Design your experiment based on:")
#         print("  tii   :", best_design_decisions['tii'])
#         print("  tvi   :", best_design_decisions['tvi'])
#         print("  swps  :", best_design_decisions['swps'])
#         print("  St    :", best_design_decisions['St'])
#         print("  md_obj:", best_design_decisions['md_obj'])
#     return best_design_decisions
#
# def _safe_run_md(args):
#     des_opt, system, models, core_num, round = args
#
#     try:
#         return _run_single_md(des_opt, system, models, core_num, round)
#     except Exception as e:
#         print(f"[Core {core_num}] FAILED: {e}")
#         traceback.print_exc()
#         return None
#
# def _run_single_md(des_opt, system, models, core_number=0, round=round):
#     tf = system['t_s'][1]
#     ti = system['t_s'][0]
#
#     tv_iphi_vars = list(system['tvi'].keys())
#     tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
#     tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
#     tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
#     tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
#     tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
#     tv_iphi_offsetl = [system['tvi'][var]['offl'] / system['tvi'][var]['max'] for var in tv_iphi_vars]
#     tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}
#
#     ti_iphi_vars = list(system['tii'].keys())
#     ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
#     ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]
#
#     tv_ophi_vars = [var for var in system['tvo'].keys() if system['tvo'][var].get('meas', True)]
#     tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
#     tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
#     tv_ophi_sampling = {var: system['tvo'][var].get('samp_s', 'default_group') for var in tv_ophi_vars}
#     tv_ophi_forcedsamples = {
#         var: [v / tf for v in system['tvo'][var].get('samp_f', [])]
#         for var in tv_ophi_vars
#     }
#
#     ti_ophi_vars = [var for var in system['tio'].keys() if system['tio'][var].get('meas', True)]
#
#     active_solvers = models['can_m']
#     theta_parameters = models['theta']
#
#     design_criteria = des_opt['md_ob']
#     maxmd = des_opt['itr']['maxmd']
#     tolmd = des_opt['itr']['tolmd']
#     eps = des_opt['eps']
#
#     # Ensure V_matrix is a dict of solver -> NumPy array (for indexing safety)
#     if 'V_matrix' in models:
#         V_matrix = {
#             solver: np.array(models['V_matrix'][solver])
#             for solver in models['can_m']
#         }
#     else:
#         V_matrix = {
#             solver: np.array([
#                 [1e-50 if i == j else 0 for j in range(len(models['theta'][solver]))]
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
#     population_size = des_opt['itr']['pps']
#     pltshow = des_opt['plt']
#     optmethod = des_opt.get('meth', 'L')
#
#     result, index_dict = _optimiser_md(
#         tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#         tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#         ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#         tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#         ti_ophi_vars,
#         tf, ti,
#         active_solvers, theta_parameters,
#         eps, maxmd, tolmd,population_size,
#         mutation, design_criteria,
#         system, models, optmethod
#     )
#
#     x_final = getattr(result, 'X', getattr(result, 'x', None))
#     if x_final is None:
#         raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
#
#     try:
#         phi, swps, St, md_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _runner_md(
#             x_final,
#             tv_iphi_vars, tv_iphi_max,
#             ti_iphi_vars, ti_iphi_max,
#             tv_ophi_vars, ti_ophi_vars,
#             active_solvers, theta_parameters,
#             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#             design_criteria, tf, eps, mutation,
#             index_dict,
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
#     phi, phit, swps, St = _reporter(
#         phi, phit, swps, St,
#         md_obj,
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
#     design_decisions = {
#         'tii': phi,
#         'tvi': phit,
#         'swps': swps,
#         'St': St,
#         'md_obj': md_obj,
#         't_values': t_values
#     }
#
#
#     return design_decisions, md_obj, swps
#
#
# def _optimiser_md(
#     tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
#     tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
#     ti_iphi_vars, ti_iphi_max, ti_iphi_min,
#     tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
#     ti_ophi_vars,
#     tf, ti,
#     active_solvers, theta_parameters,
#     eps, maxmd, tolmd, population_size,
#     mutation, design_criteria,
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
#         _md_of,
#         tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
#         ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
#         tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
#         tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
#         tv_ophi_sampling=tv_ophi_sampling,
#         active_solvers=active_solvers, theta_parameters=theta_parameters,
#         tf=tf, eps=eps, mutation=mutation,
#         design_criteria=design_criteria,
#         index_dict=index_dict, system=system, models=models
#     )
#
#     class Problem(ElementwiseProblem):
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
#     problem = Problem()
#
#     if optmethod == 'PS':
#         algorithm = PatternSearch()
#         res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxmd), seed=None, verbose=True)
#
#     elif optmethod == 'DE':
#         algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
#         res_refine = pymoo_minimize(
#             problem,
#             algorithm,
#             termination=('n_gen', maxmd),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolmd,
#             save_history=True
#         )
#
#     elif optmethod == 'DEPS':
#         algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
#         res_de = pymoo_minimize(
#             problem,
#             algorithm_de,
#             termination=('n_gen', maxmd/2),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolmd,
#             save_history=True
#         )
#         algorithm_refine = PatternSearch()
#         res_refine = pymoo_minimize(
#             problem,
#             algorithm_refine,
#             termination=('n_gen', int(maxmd)),
#             seed=None,
#             verbose=True,
#             constraint_tolerance=tolmd,
#             x0=res_de.X
#         )
#
#     else:
#         raise ValueError("optmethod must be 'L', 'G', or 'GL'")
#
#     return res_refine, index_dict
#
#
#
#
#
# def _md_of(
#     x,
#     tv_iphi_vars, tv_iphi_max,
#     ti_iphi_vars, ti_iphi_max,
#     tv_ophi_vars, ti_ophi_vars,
#     active_solvers, theta_parameters,
#     tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#     tf, eps, mutation, design_criteria,
#     index_dict, system, models
# ):
#     """
#     Objective function wrapper for MBDOE-MD.
#     """
#     try:
#         _, _, _, md_obj, _, _, _, _ = _runner_md(
#             x,
#             tv_iphi_vars, tv_iphi_max,
#             ti_iphi_vars, ti_iphi_max,
#             tv_ophi_vars, ti_ophi_vars,
#             active_solvers, theta_parameters,
#             tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
#             design_criteria,
#             tf, eps, mutation,
#             index_dict,
#             system, models
#         )
#         if np.isnan(md_obj):
#             return 1e6
#         return -md_obj  # negative for maximization
#     except Exception as e:
#         print(f"Exception in md_objective_function: {e}")
#         return 1e6
#
#
#
# def _runner_md(
#     x,
#     tv_iphi_vars,
#     tv_iphi_max,
#     ti_iphi_vars,
#     ti_iphi_max,
#     tv_ophi_vars,
#     ti_ophi_vars,
#     active_solvers,
#     theta_parameters,
#     tv_iphi_cvp,
#     tv_ophi_forcedsamples,
#     tv_ophi_sampling,
#     design_criteria,
#     tf,
#     eps,
#     mutation,
#     index_dict,
#     system,
#     models
# ):
#     """
#     Simulate models and evaluate the MD objective.
#     Returns the sliced inputs, objective value, time vector, and outputs.
#     """
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
#     y_values_dict = defaultdict(dict)
#     y_i_values_dict = defaultdict(dict)
#     indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars}
#
#     J_dot = defaultdict(
#         lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
#                           len(theta_parameters[active_solvers[0]])))
#     )
#     tv_ophi = {}
#     ti_ophi = {}
#     phit_interp = {}
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
#         tv_out, ti_out, interp_data = simula(
#             t_values, swps_data, ti_iphi_data,
#             phisc, phitsc, tsc,
#             theta, thetac,
#             tv_iphi_cvp, {},
#             solver_name,
#             system,
#             models
#         )
#
#         tv_ophi[solver_name] = tv_out
#         ti_ophi[solver_name] = ti_out
#         phit_interp = interp_data
#
#         for var in tv_ophi_vars:
#             y_values_dict[solver_name][var] = np.array(tv_out[var])
#         for var in ti_ophi_vars:
#             y_values_dict[solver_name][var] = np.array([ti_out[var]])
#
#         free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
#
#         for para_idx in free_params_indices:
#             modified_theta = theta.copy()
#             modified_theta[para_idx] += eps
#
#             tv_out_mod, ti_out_mod, _ = simula(
#                 t_values, swps_data, ti_iphi_data,
#                 phisc, phitsc, tsc,
#                 modified_theta, thetac,
#                 tv_iphi_cvp, {},
#                 solver_name,
#                 system,
#                 models
#             )
#
#             for var in tv_ophi_vars:
#                 y_i_values_dict[solver_name][var] = np.array(tv_out_mod[var])
#             for var in ti_ophi_vars:
#                 y_i_values_dict[solver_name][var] = np.array([ti_out_mod[var]])
#
#             for var in indices:
#                 LSA[var][solver_name][para_idx] = (
#                     y_i_values_dict[solver_name][var] - y_values_dict[solver_name][var]
#                 ) / eps
#
#     md_obj = 0.0
#
#     if design_criteria == 'HR':
#         for i, s1 in enumerate(active_solvers):
#             for s2 in active_solvers[i+1:]:
#                 for var, mask in indices.items():
#                     y1 = y_values_dict[s1][var]
#                     y2 = y_values_dict[s2][var]
#                     md_obj += np.sum((y1[mask] - y2[mask]) ** 2)
#
#     elif design_criteria == 'BFF':
#         std_dev = {}
#         for var in tv_ophi_vars:
#             unc = system.get('tvo', {}).get(var, {}).get('unc', 1.0)
#             if unc is None or np.isnan(unc):
#                 unc = 1.0
#             std_dev[var] = unc
#
#         Sigma_y = np.diag([std_dev[var] ** 2 for var in tv_ophi_vars])
#
#         for i, s1 in enumerate(active_solvers):
#             for s2 in active_solvers[i+1:]:
#                 for t_idx, t in enumerate(t_values):
#                     y1 = np.array([y_values_dict[s1][var][t_idx] for var in tv_ophi_vars])
#                     y2 = np.array([y_values_dict[s2][var][t_idx] for var in tv_ophi_vars])
#                     diff = y1 - y2
#
#                     # Sensitivity matrices at time t
#                     free_s1 = [i for i, v in enumerate(mutation[s1]) if v]
#                     thetac_s1 = np.array(theta_parameters[s1])[free_s1]
#                     V1 = np.array([[LSA[var][s1][p][t_idx] for p in free_s1] for var in tv_ophi_vars])
#                     Sigma_theta_s1_inv = np.diag(1.0 / (thetac_s1 ** 2 + 1e-50))
#                     W1 = V1 @ Sigma_theta_s1_inv @ V1.T
#
#                     free_s2 = [i for i, v in enumerate(mutation[s2]) if v]
#                     thetac_s2 = np.array(theta_parameters[s2])[free_s2]
#                     V2 = np.array([[LSA[var][s2][p][t_idx] for p in free_s2] for var in tv_ophi_vars])
#                     Sigma_theta_s2_inv = np.diag(1.0 / (thetac_s2 ** 2 + 1e-50))
#                     W2 = V2 @ Sigma_theta_s2_inv @ V2.T
#
#                     try:
#                         S = Sigma_y + W1 + W2
#                         md_obj += diff.T @ np.linalg.inv(S) @ diff
#                     except np.linalg.LinAlgError:
#                         md_obj += 1e6  # Penalise ill-conditioned cases
#
#     logger = configure_logger()
#     logger.info(f"mbdoe-MD:{design_criteria} is running with {md_obj:.4f}")
#
#     return ti, swps, St, md_obj, t_values, tv_ophi, ti_ophi, phit_interp


from middoe.des_utils import _slicer, _reporter,  configure_logger
from functools import partial
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from multiprocessing import Pool
import traceback
from middoe.krnl_simula import simula
from collections import defaultdict
import numpy as np
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from typing import Dict, Any, Union



def mbdoe_md(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    round: int,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    r"""
    Perform Model-Based Design of Experiments for Model Discrimination (MBDoE-MD).

    This function identifies experimental designs that maximise the ability to
    discriminate between competing models. Optimisation can run in parallel or
    single-core mode. Failed runs are discarded, and the best valid design is returned.

    Parameters
    ----------
    des_opt : dict
        Optimisation settings, including:
            - 'mdob' : str
                Discrimination criterion:
                    * 'HR': Hunter-Reiner T-optimality
                        Maximises sum of squared differences between model predictions
                        across all time points and measured outputs. Suitable for
                        discriminating models with significant structural differences.

                    * 'BFF': Buzzi-Ferraris-Forzatti T-optimality
                        Alternative T-optimality criterion with different weighting
                        scheme for model divergence computation. Accounts for signal
                        magnitude in discrimination objective.

            - 'itr' : dict
                Iteration controls:
                    * 'maxmd': int — maximum iterations
                    * 'toldmd': float — convergence tolerance
                    * 'pps': int — population size
            - 'eps' : float
                Perturbation step size for numerical derivatives.
            - 'plt' : bool
                Enable plotting of results.
            - 'meth' : str, optional
                Optimisation method ('DE', 'PS', 'DEPS').

    system : dict
        System model and constraints:
            - 't_s' : tuple[float, float]
                Start and end times (ti, tf).
            - 't_d' : tuple[float, float]
                Restricted initial/final intervals (dead time).
            - 't_r' : float
                Time resolution for simulation.
            - 'tvi' : dict
                Time-variant input definitions with keys:
                    * 'max', 'min': float — bounds
                    * 'stps': int — number of switching segments
                    * 'const': str — constraint type ('inc', 'dec', 'rel')
                    * 'offt': float — minimum time offset between switches
                    * 'offl': float — minimum level offset
                    * 'cvp': str — control variable profile ('CPF' piecewise-constant,
                                   'LPF' piecewise-linear)
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
                Competing models to discriminate (active solver names).
            - 'theta' : dict[str, list[float]]
                Nominal parameter values for each model.
            - 'mutation' : dict[str, list[bool]]
                Parameter masks indicating which parameters are free (True) vs fixed (False).
            - 'V_matrix' : dict[str, np.ndarray], optional
                Parameter covariance matrices (default: diagonal with small values).

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
            - 'md_obj' : float
                Model discrimination objective value (maximised).
            - 't_values' : list[float]
                Full simulation time vector.

    Raises
    ------
    RuntimeError
        If all parallel runs fail or if the single-core run encounters an exception.

    Notes
    -----
    MBDoE-MD aims to design experiments that enhance the ability to reject
    incorrect models by maximising discrimination criteria based on T-optimality.

    The discrimination objective maximises the divergence between competing model
    predictions, enabling statistical tests (P-test, t-test) to identify the most
    representative model.

    **Workflow for Model Discrimination:**
    1. Calibrate all candidate models on available data
    2. Design discriminative experiment using MBDoE-MD (HR or BFF criterion)
    3. Conduct designed experiment
    4. Re-calibrate models with new data
    5. Compute model probabilities via P-test
    6. Select model with highest posterior probability
    7. Validate selected model

    Supported constraints include:
        - Control level bounds and switching rules (increasing, decreasing, relaxed)
        - Minimum sampling intervals and total sample limits
        - Forced or forbidden sampling times
        - Synchronised sampling across outputs
        - Piecewise-constant (CPF) or piecewise-linear (LPF) control profiles

    References
    ----------
    .. [1] Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
       MIDDoE: An MBDoE Python package for model identification, discrimination,
       and calibration. *Computers & Chemical Engineering*.

    .. [2] Hunter, W.G., & Reiner, A.M. (1965).
       Designs for discriminating between two rival models.
       *Technometrics*, 7(3), 307–323.

    .. [3] Buzzi-Ferraris, G., & Forzatti, P. (1983).
       Sequential experimental design for model discrimination in the case of
       multiple responses. *Chemical Engineering Science*, 38(2), 225–232.

    See Also
    --------
    _safe_run_md : Parallel optimisation wrapper with exception handling.
    _run_single_md : Single-core optimisation routine.
    _optimiser_md : Core optimisation problem setup and solver.

    Examples
    --------
    >>> import numpy as np
    >>> from middoe import des_md
    >>>
    >>> # Define system with 4 competing models
    >>> system = {
    ...     'tvo': {
    ...         'X': {'init': 0.5, 'meas': True, 'unc': 0.05, 'sp': 17},
    ...         'S': {'init': 15.0, 'meas': True, 'unc': 1.0, 'sp': 17},
    ...         'P': {'init': 0.0, 'meas': True, 'unc': 0.05, 'sp': 17}
    ...     },
    ...     'tvi': {
    ...         'T': {'min': 296.15, 'max': 306.15, 'stps': 5, 'cvp': 'CPF',
    ...               'const': 'rel', 'offt': 2.0, 'offl': 0.5}
    ...     },
    ...     'tii': {
    ...         'S0': {'min': 0.366, 'max': 0.65},
    ...         'X0': {'min': 0.19, 'max': 0.595}
    ...     },
    ...     't_s': (0.0, 20.0),
    ...     't_d': (1.0, 1.0)
    ... }
    >>>
    >>> # Define 4 competing models (after calibration)
    >>> models = {
    ...     'can_m': ['M1', 'M2', 'M3', 'M4'],
    ...     'krt': {'M1': 'fermentation_model_1', 'M2': 'fermentation_model_2',
    ...             'M3': 'fermentation_model_3', 'M4': 'fermentation_model_4'},
    ...     'theta': {
    ...         'M1': [0.408, 0.22, 71.5, 0.28, 0.607, 0.1],
    ...         'M2': [0.450, 0.20, 65.0, 0.30, 0.550, 0.12],
    ...         'M3': [0.380, 0.24, 75.0, 0.26, 0.630, 0.09],
    ...         'M4': [0.420, 0.21, 70.0, 0.29, 0.580, 0.11]
    ...     },
    ...     'mutation': {
    ...         'M1': [True, True, True, False, False, False],
    ...         'M2': [True, True, True, False, False, False],
    ...         'M3': [True, True, True, False, False, False],
    ...         'M4': [True, True, True, False, False, False]
    ...     }
    ... }
    >>>
    >>> # Configure MBDoE-MD with Hunter-Reiner criterion
    >>> des_opt = {
    ...     'mdob': 'HR',
    ...     'meth': 'DEPS',
    ...     'itr': {'maxmd': 5000, 'toldmd': 1e-8, 'pps': 50},
    ...     'eps': 0.01,
    ...     'plt': True
    ... }
    >>>
    >>> # Run MBDoE-MD (parallel mode with 4 cores)
    >>> design = des_md.mbdoe_md(des_opt, system, models, round=2, num_parallel_runs=4)
    >>>
    >>> # Access results
    >>> print(f"Optimal temperature profile: {design['tvi']['T']}")
    >>> print(f"Optimal initial substrate: {design['tii']['S0']:.4f} mol/L")
    >>> print(f"Optimal initial biomass: {design['tii']['X0']:.4f} mol/L")
    >>> print(f"Sampling times: {design['St']['X']}")
    >>> print(f"HR discrimination value: {design['md_obj']:.4e}")
    """
    if num_parallel_runs > 1:
        with Pool(num_parallel_runs) as pool:
            results_list = pool.map(
                _safe_run_md,
                [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
            )

        successful = [res for res in results_list if res is not None]
        if not successful:
            raise RuntimeError(
                "All MBDoE-MD optimisation runs failed. "
                "Try increasing 'maxmd', adjusting bounds, or relaxing constraints."
            )

        best_design_decisions, best_md_obj, best_swps = max(successful, key=lambda x: x[1])

    else:
        try:
            best_design_decisions, best_md_obj, best_swps = _run_single_md(
                des_opt, system, models, core_number=0, round=round
            )
        except Exception as e:
            raise RuntimeError(f"Single-core optimisation failed: {e}")

        print("Design your experiment based on:")
        print("  tii   :", best_design_decisions['tii'])
        print("  tvi   :", best_design_decisions['tvi'])
        print("  swps  :", best_design_decisions['swps'])
        print("  St    :", best_design_decisions['St'])
        print("  md_obj:", best_design_decisions['md_obj'])
    return best_design_decisions


def _safe_run_md(args):
    """
    Safely execute a single MBDoE-MD optimisation run with exception handling.

    This wrapper function is used for parallel execution to prevent individual
    core failures from crashing the entire optimisation. Exceptions are caught,
    logged, and returned as None.

    Parameters
    ----------
    args : tuple
        Packed arguments for _run_single_md:
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
        If successful: (design_decisions, md_obj, swps).
        If failed: None.

    Notes
    -----
    This function is intended for use with multiprocessing.Pool.map().
    Exception tracebacks are printed to stderr for debugging.

    See Also
    --------
    _run_single_md : Core single-run optimisation logic.
    mbdoe_md : Main entry point that manages parallel execution.
    """
    des_opt, system, models, core_num, round = args

    try:
        return _run_single_md(des_opt, system, models, core_num, round)
    except Exception as e:
        print(f"[Core {core_num}] FAILED: {e}")
        traceback.print_exc()
        return None


def _run_single_md(des_opt, system, models, core_number=0, round=round):
    """
    Execute a single MBDoE-MD optimisation run on one core.

    This function unpacks system and model configurations, constructs the
    optimisation problem, invokes the solver, and evaluates the final design.

    Parameters
    ----------
    des_opt : dict
        Design optimisation settings (see mbdoe_md documentation).
    system : dict
        System configuration including inputs, outputs, and constraints.
    models : dict
        Model definitions including competing models, parameters, and mutations.
    core_number : int, optional
        Identifier for the current core/process (default: 0).
    round : int
        Current experimental design round.

    Returns
    -------
    design_decisions : dict
        Optimal design variables and objective value.
    md_obj : float
        Model discrimination objective value (to be maximised).
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
        3. Invoke _optimiser_md to solve the optimisation problem.
        4. Evaluate the final design using _runner_md.
        5. Report results via _reporter.

    Default covariance matrices (V_matrix) and mutation masks are constructed
    if not provided in the models dictionary.

    See Also
    --------
    _optimiser_md : Optimisation problem construction and solver.
    _runner_md : Simulation kernel for design evaluation.
    _reporter : Result formatting and visualisation.
    """
    tf = system['t_s'][1]
    ti = system['t_s'][0]

    tv_iphi_vars = list(system['tvi'].keys())
    tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
    tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
    tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
    tv_iphi_offsetl = [system['tvi'][var]['offl'] / system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}

    ti_iphi_vars = list(system['tii'].keys())
    ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]

    tv_ophi_vars = [var for var in system['tvo'].keys() if system['tvo'][var].get('meas', True)]
    tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
    tv_ophi_sampling = {var: system['tvo'][var].get('samp_s', 'default_group') for var in tv_ophi_vars}
    tv_ophi_forcedsamples = {
        var: [v / tf for v in system['tvo'][var].get('samp_f', [])]
        for var in tv_ophi_vars
    }

    ti_ophi_vars = [var for var in system['tio'].keys() if system['tio'][var].get('meas', True)]

    active_solvers = models['can_m']
    theta_parameters = models['theta']

    design_criteria = des_opt['md_ob']
    maxmd = des_opt['itr']['maxmd']
    tolmd = des_opt['itr']['tolmd']
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
                [1e-50 if i == j else 0 for j in range(len(models['theta'][solver]))]
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
    population_size = des_opt['itr']['pps']
    pltshow = des_opt['plt']
    optmethod = des_opt.get('meth', 'L')

    result, index_dict = _optimiser_md(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
        ti_ophi_vars,
        tf, ti,
        active_solvers, theta_parameters,
        eps, maxmd, tolmd,population_size,
        mutation, design_criteria,
        system, models, optmethod
    )

    x_final = getattr(result, 'X', getattr(result, 'x', None))
    if x_final is None:
        raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")

    try:
        phi, swps, St, md_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _runner_md(
            x_final,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            design_criteria, tf, eps, mutation,
            index_dict,
            system,
            models
        )
    except ValueError as e:
        if "MBDoE optimiser kernel was unsuccessful" in str(e):
            print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
            return None
        else:
            raise

    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        md_obj,
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf,
        design_criteria,
        round, pltshow,
        core_number
    )

    design_decisions = {
        'tii': phi,
        'tvi': phit,
        'swps': swps,
        'St': St,
        'md_obj': md_obj,
        't_values': t_values
    }


    return design_decisions, md_obj, swps


def _optimiser_md(
    tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
    tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
    ti_iphi_vars, ti_iphi_max, ti_iphi_min,
    tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
    ti_ophi_vars,
    tf, ti,
    active_solvers, theta_parameters,
    eps, maxmd, tolmd, population_size,
    mutation, design_criteria,
    system, models, optmethod
):
    """
    Construct and solve the MBDoE-MD optimisation problem.

    This function builds the decision variable vector, bounds, constraints,
    and index mapping, then invokes a pymoo-based optimiser to maximise
    the model discrimination objective.

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
        Initial simulation time.
    active_solvers : list[str]
        Names of competing models.
    theta_parameters : dict[str, list[float]]
        Nominal parameter values for each model.
    eps : float
        Perturbation step for numerical derivatives.
    maxmd : int
        Maximum number of optimisation iterations/generations.
    tolmd : float
        Constraint tolerance for pymoo solvers.
    population_size : int
        Population size for evolutionary algorithms.
    mutation : dict[str, list[bool]]
        Free/fixed parameter masks for each model.
    design_criteria : str
        Discrimination criterion ('HR', 'BFF').
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
    _md_of : Objective function wrapper for evaluation.
    _runner_md : Simulation kernel invoked within objective.
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
        _md_of,
        tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
        ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
        tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
        tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
        tv_ophi_sampling=tv_ophi_sampling,
        active_solvers=active_solvers, theta_parameters=theta_parameters,
        tf=tf, eps=eps, mutation=mutation,
        design_criteria=design_criteria,
        index_dict=index_dict, system=system, models=models
    )

    class Problem(ElementwiseProblem):
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

    problem = Problem()

    if optmethod == 'PS':
        algorithm = PatternSearch()
        res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxmd), seed=None, verbose=True)

    elif optmethod == 'DE':
        algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
        res_refine = pymoo_minimize(
            problem,
            algorithm,
            termination=('n_gen', maxmd),
            seed=None,
            verbose=True,
            constraint_tolerance=tolmd,
            save_history=True
        )

    elif optmethod == 'DEPS':
        algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
        res_de = pymoo_minimize(
            problem,
            algorithm_de,
            termination=('n_gen', maxmd/2),
            seed=None,
            verbose=True,
            constraint_tolerance=tolmd,
            save_history=True
        )
        algorithm_refine = PatternSearch()
        res_refine = pymoo_minimize(
            problem,
            algorithm_refine,
            termination=('n_gen', int(maxmd)),
            seed=None,
            verbose=True,
            constraint_tolerance=tolmd,
            x0=res_de.X
        )

    else:
        raise ValueError("optmethod must be 'L', 'G', or 'GL'")

    return res_refine, index_dict


def _md_of(
    x,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
    tf, eps, mutation, design_criteria,
    index_dict, system, models
):
    """
    Objective function wrapper for MBDoE-MD optimisation.

    This function evaluates the model discrimination objective for a given
    design vector by invoking the simulation kernel (_runner_md). Exceptions
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
        Competing model names.
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
    design_criteria : str
        Discrimination criterion ('HR', 'BFF').
    index_dict : dict
        Mapping of variable names to decision vector indices.
    system : dict
        Full system configuration.
    models : dict
        Full model definitions.

    Returns
    -------
    obj_value : float
        Negative of the discrimination objective (for minimisation).
        Returns 1e6 if evaluation fails or results in NaN.

    Notes
    -----
    The objective is negated because pymoo minimises by default, while
    model discrimination seeks to maximise information content.

    See Also
    --------
    _runner_md : Simulation kernel for model evaluation.
    _optimiser_md : Optimisation problem setup.
    """
    try:
        _, _, _, md_obj, _, _, _, _ = _runner_md(
            x,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            design_criteria,
            tf, eps, mutation,
            index_dict,
            system, models
        )
        if np.isnan(md_obj):
            return 1e6
        return -md_obj  # negative for maximization
    except Exception as e:
        print(f"Exception in md_objective_function: {e}")
        return 1e6


def _runner_md(
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
    design_criteria,
    tf,
    eps,
    mutation,
    index_dict,
    system,
    models
):
    r"""
    Simulate competing models and evaluate the model discrimination objective.

    This function decodes the decision vector, constructs time grids, invokes
    the simulator for each model (nominal and perturbed parameters), computes
    local sensitivity approximations, and calculates the discrimination metric
    (HR or BFF).

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
        Competing model names.
    theta_parameters : dict[str, list[float]]
        Nominal parameter values for each model.
    tv_iphi_cvp : dict[str, bool]
        Control variable parameterisation flags.
    tv_ophi_forcedsamples : dict[str, list[float]]
        Forced sampling times (normalised).
    tv_ophi_sampling : dict[str, str]
        Sampling group identifiers.
    design_criteria : str
        Discrimination criterion:
            - 'HR': Hunter–Reiner (sum of squared prediction differences)
            - 'BFF': Bayesian-Frequentist Fusion (weighted by parameter uncertainty)
    tf : float
        Final simulation time.
    eps : float
        Perturbation step for numerical sensitivity (forward differences).
    mutation : dict[str, list[bool]]
        Free/fixed parameter masks for each model.
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
    md_obj : float
        Model discrimination objective value (positive, to be maximised).
    t_values : list[float]
        Full simulation time vector (denormalised).
    tv_ophi : dict[str, dict[str, np.ndarray]]
        Time-variant outputs for each model and variable.
    ti_ophi : dict[str, dict[str, float]]
        Time-invariant outputs for each model and variable.
    phit_interp : dict
        Interpolation data for time-variant inputs.

    Notes
    -----
    **HR (Hunter–Reiner) Criterion:**
        Maximises the sum of squared differences between model predictions:

        \[
        J_{\text{HR}} = \sum_{i<j} \sum_{t \in S_t} \sum_{y} (y_i(t) - y_j(t))^2
        \]

    **BFF (Bayesian-Frequentist Fusion) Criterion:**
        Accounts for parameter uncertainty via local sensitivity analysis:

        \[
        J_{\text{BFF}} = \sum_{i<j} \sum_{t} (y_i - y_j)^T S^{-1} (y_i - y_j)
        \]

        where \( S = \Sigma_y + W_i + W_j \) combines measurement noise and
        parameter-induced covariance (\( W = V \Sigma_{\theta}^{-1} V^T \)).

    The function logs the objective value via the configured logger.

    See Also
    --------
    simula : Core simulation kernel for model evaluation.
    _slicer : Decodes decision vector into design variables.
    _md_of : Objective function wrapper.
    """
    x = x.tolist() if not isinstance(x, list) else x

    dt_real = system['t_r']
    nodes = int(round(tf / dt_real)) + 1
    tlin = np.linspace(0, 1, nodes)
    ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
    St = {var: np.array(sorted(St[var])) for var in St}
    t_values_flat = [tp for times in St.values() for tp in times]
    t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

    LSA = defaultdict(lambda: defaultdict(dict))
    y_values_dict = defaultdict(dict)
    y_i_values_dict = defaultdict(dict)
    indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars}

    J_dot = defaultdict(
        lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                          len(theta_parameters[active_solvers[0]])))
    )
    tv_ophi = {}
    ti_ophi = {}
    phit_interp = {}

    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        theta = np.array([1.0] * len(thetac))
        ti_iphi_data = ti
        swps_data = swps
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf

        tv_out, ti_out, interp_data = simula(
            t_values, swps_data, ti_iphi_data,
            phisc, phitsc, tsc,
            theta, thetac,
            tv_iphi_cvp, {},
            solver_name,
            system,
            models
        )

        tv_ophi[solver_name] = tv_out
        ti_ophi[solver_name] = ti_out
        phit_interp = interp_data

        for var in tv_ophi_vars:
            y_values_dict[solver_name][var] = np.array(tv_out[var])
        for var in ti_ophi_vars:
            y_values_dict[solver_name][var] = np.array([ti_out[var]])

        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]

        for para_idx in free_params_indices:
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            tv_out_mod, ti_out_mod, _ = simula(
                t_values, swps_data, ti_iphi_data,
                phisc, phitsc, tsc,
                modified_theta, thetac,
                tv_iphi_cvp, {},
                solver_name,
                system,
                models
            )

            for var in tv_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array(tv_out_mod[var])
            for var in ti_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array([ti_out_mod[var]])

            for var in indices:
                LSA[var][solver_name][para_idx] = (
                    y_i_values_dict[solver_name][var] - y_values_dict[solver_name][var]
                ) / eps

    md_obj = 0.0

    if design_criteria == 'HR':
        for i, s1 in enumerate(active_solvers):
            for s2 in active_solvers[i+1:]:
                for var, mask in indices.items():
                    y1 = y_values_dict[s1][var]
                    y2 = y_values_dict[s2][var]
                    md_obj += np.sum((y1[mask] - y2[mask]) ** 2)

    elif design_criteria == 'BFF':
        std_dev = {}
        for var in tv_ophi_vars:
            unc = system.get('tvo', {}).get(var, {}).get('unc', 1.0)
            if unc is None or np.isnan(unc):
                unc = 1.0
            std_dev[var] = unc

        Sigma_y = np.diag([std_dev[var] ** 2 for var in tv_ophi_vars])

        for i, s1 in enumerate(active_solvers):
            for s2 in active_solvers[i+1:]:
                for t_idx, t in enumerate(t_values):
                    y1 = np.array([y_values_dict[s1][var][t_idx] for var in tv_ophi_vars])
                    y2 = np.array([y_values_dict[s2][var][t_idx] for var in tv_ophi_vars])
                    diff = y1 - y2

                    # Sensitivity matrices at time t
                    free_s1 = [i for i, v in enumerate(mutation[s1]) if v]
                    thetac_s1 = np.array(theta_parameters[s1])[free_s1]
                    V1 = np.array([[LSA[var][s1][p][t_idx] for p in free_s1] for var in tv_ophi_vars])
                    Sigma_theta_s1_inv = np.diag(1.0 / (thetac_s1 ** 2 + 1e-50))
                    W1 = V1 @ Sigma_theta_s1_inv @ V1.T

                    free_s2 = [i for i, v in enumerate(mutation[s2]) if v]
                    thetac_s2 = np.array(theta_parameters[s2])[free_s2]
                    V2 = np.array([[LSA[var][s2][p][t_idx] for p in free_s2] for var in tv_ophi_vars])
                    Sigma_theta_s2_inv = np.diag(1.0 / (thetac_s2 ** 2 + 1e-50))
                    W2 = V2 @ Sigma_theta_s2_inv @ V2.T

                    try:
                        S = Sigma_y + W1 + W2
                        md_obj += diff.T @ np.linalg.inv(S) @ diff
                    except np.linalg.LinAlgError:
                        md_obj += 1e6  # Penalise ill-conditioned cases

    logger = configure_logger()
    logger.info(f"mbdoe-MD:{design_criteria} is running with {md_obj:.4f}")

    return ti, swps, St, md_obj, t_values, tv_ophi, ti_ophi, phit_interp