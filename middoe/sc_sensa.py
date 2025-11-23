# # sc_sensa.py
#
# import numpy as np
# from scipy.stats import norm, qmc
# import pandas as pd
# import scipy as sp
# from concurrent.futures import ProcessPoolExecutor
# import os
# from middoe.krnl_simula import simula
# from middoe.iden_utils import plot_sobol_results
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
#
# def sensa(gsa, models, system):
#     """
#     Perform Sobol sensitivity analysis for the given settings and model structure.
#
#     Parameters
#     ----------
#     gsa : dict
#         Dictionary containing the settings for the global sensitivity analysis.
#     models : dict
#         Dictionary containing the settings for the modelling process.
#     system : dict
#         Dictionary containing the structure of the model.
#
#     Returns
#     -------
#     tuple
#         A tuple containing the Sobol analysis results and the Sobol problem definition.
#     """
#     phi_nom = gsa['tii_n']  # Nominal values for time-invariant variables
#     phit_nom = gsa['tvi_n']  # Nominal values for time-variant variables
#     sampling = gsa['samp']
#     var_sensitivity = gsa['var_s']
#     par_sensitivity = gsa['par_s']
#     plt_show = gsa['plt']
#     active_solvers = models['can_m']
#     theta_parameters = models['theta']
#     thetamaxd= models['t_u']
#     thetamind= models['t_l']
#     cvp_initial = {
#         var: 'no_CVP'
#         for var in system['tvi'].keys()
#     }
#     # piecewise_func = system['tv_iphi']['initial_piecewise_func'][0]
#     parallel_value = gsa.get('multi', False)
#
#     # Retrieve time information from model structure and ensure it's an integer
#     time_field = system['t_s'][1]
#     t = list(np.linspace(0, 1, 305))  # Ensure `t` is a list of floats
#     # t = {key: list(np.linspace(0, 1, 305)) for key in system['tv_iphi'].keys()}
#     tsc = int(time_field)  # Ensure `tsc` is an integer
#     sobol_problem = {}
#     sa = {}
#
#     if isinstance(parallel_value, float):
#         # MULTI-CORE EXECUTION
#         power = parallel_value  # e.g., 0.7 means use 70% of CPU cores
#         total_cpus = os.cpu_count()
#         num_workers = int(np.floor(total_cpus * power))  # Use 70% of available CPUs
#
#         # Use ProcessPoolExecutor for multiprocessing
#         with ProcessPoolExecutor(max_workers=num_workers) as executor:
#             for solver in active_solvers:
#                 print(f'Running GSA-Sobol for model: {solver} in parallel')
#                 theta = [float(val) for val in theta_parameters[solver]]
#                 thetamax = [float(m) for m in thetamaxd[solver]]
#                 thetamin = [float(m) for m in thetamind[solver]]
#
#                 # Construct scaling factors
#                 phisc = {key: 1.0 for key in system['tii'].keys()}  # Time-invariant input scaling factors
#                 phitsc = {key: 1.0 for key in system['tvi'].keys()}  # Time-variant input scaling factors
#                 thetac = [1.0 for _ in theta]  # Parameter scaling factors
#
#                 # Prepare variable bounds
#                 if isinstance(gsa.get('var_d'), float):
#                     phi_bounds = [
#                         (gsa['tii_n'][i] / gsa['var_d'],
#                          gsa['tvi_n'][i] * gsa['var_d'])
#                         for i in range(len(gsa['tii_n']))
#                     ]
#                     phit_bounds = [
#                         (gsa['tii_n'][i] / gsa['var_d'],
#                          gsa['tvi_n'][i] * gsa['var_d'])
#                         for i in range(len(gsa['tvi_n']))
#                     ]
#                 else:
#                     phi_bounds = [
#                         (var_attrs['min'], var_attrs['max']) for var_attrs in system['tii'].values()
#                     ]
#                     phit_bounds = [
#                         (var_attrs['min'], var_attrs['max']) for var_attrs in system['tvi'].values()
#                     ]
#
#                 if isinstance(gsa.get('par_d'), float):
#                     param_bounds = [(theta[i] / gsa['par_d'], theta[i] * gsa['par_d']) for i in range(len(theta))]
#                 else:
#                     param_bounds = list(zip(thetamin, thetamax))
#
#
#                 sobol_problem[solver] = {'num_vars': 0, 'names': [], 'bounds': []}
#
#                 if par_sensitivity:
#                     sobol_problem[solver]['num_vars'] += len(theta)
#                     sobol_problem[solver]['names'].extend([f'Param_{i + 1}' for i in range(len(theta))])
#                     sobol_problem[solver]['bounds'].extend(param_bounds)
#
#                 if var_sensitivity:
#                     sobol_problem[solver]['num_vars'] += len(phi_nom) + len(phit_nom)
#                     # sobol_problem[model]['names'].extend(system['ti_iphi']['var'] + system['tv_iphi']['var'])
#                     sobol_problem[solver]['names'].extend(
#                         list(system['tii'].keys()) + list(system['tvi'].keys()))
#
#                     sobol_problem[solver]['bounds'].extend(phi_bounds)
#                     sobol_problem[solver]['bounds'].extend(phit_bounds)
#
#                 sobol_samples = sample(sobol_problem[solver], N=sampling)
#
#                 # Split sobol_samples into chunks for parallel processing
#                 chunk_size = len(sobol_samples) // num_workers
#                 sobol_sample_chunks = [sobol_samples[i:i + chunk_size] for i in range(0, len(sobol_samples), chunk_size)]
#
#                 # Submit tasks for each chunk of sobol_samples for parallel processing
#                 futures = [executor.submit(_process_sample_chunk, chunk, solver, t, phisc, phitsc, theta, thetac, cvp_initial, tsc, sobol_problem[solver], phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models) for chunk in sobol_sample_chunks]
#
#                 # Collect results from all chunks
#                 results = []
#                 for future in futures:
#                     results.extend(future.result())
#
#                 sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)
#
#     else:
#         # SINGLE-CORE EXECUTION
#         for solver in active_solvers:
#             print(f'Running GSA-Sobol for model: {solver} in single-core')
#             theta = [float(val) for val in theta_parameters[solver]]
#             thetamax = [float(m) for m in thetamaxd[solver]]
#             thetamin = [float(m) for m in thetamind[solver]]
#
#             phisc = {key: 1.0 for key in system['tii'].keys()}  # Time-invariant input scaling factors
#             phitsc = {key: 1.0 for key in system['tvi'].keys()}  # Time-variant input scaling factors
#             thetac = [1.0 for _ in theta]  # Parameter scaling factors
#
#             if isinstance(gsa.get('var_d'), float):
#                 phi_bounds = [
#                     (phi_nom[i] / gsa['var_d'], phi_nom[i] * gsa['var_d'])
#                     for i, _ in enumerate(system['tii'].keys())
#                 ]
#                 phit_bounds = [
#                     (phit_nom[i] / gsa['var_d'], phit_nom[i] * gsa['var_d'])
#                     for i, _ in enumerate(system['tvi'].keys())
#                 ]
#             else:
#                 phi_bounds = [
#                     (attrs['min'], attrs['max']) for attrs in system['tii'].values()
#                 ]
#                 phit_bounds = [
#                     (attrs['min'], attrs['max']) for attrs in system['tvi'].values()
#                 ]
#
#             if isinstance(gsa.get('par_d'), float):
#                 param_bounds = [(theta[i] / gsa['par_d'], theta[i] * gsa['par_d']) for i in range(len(theta))]
#             else:
#                 param_bounds = list(zip(thetamin, thetamax))
#
#             sobol_problem[solver] = {'num_vars': 0, 'names': [], 'bounds': []}
#
#             if par_sensitivity:
#                 sobol_problem[solver]['num_vars'] += len(theta)
#                 sobol_problem[solver]['names'].extend([f'Param_{i + 1}' for i in range(len(theta))])
#                 sobol_problem[solver]['bounds'].extend(param_bounds)
#
#             if var_sensitivity:
#                 sobol_problem[solver]['num_vars'] += len(phi_nom) + len(phit_nom)
#                 # sobol_problem[model]['names'].extend(system['ti_iphi']['var'] + system['tv_iphi']['var'])
#                 sobol_problem[solver]['names'].extend(
#                     list(system['tii'].keys()) + list(system['tvi'].keys()))
#
#                 sobol_problem[solver]['bounds'].extend(phi_bounds)
#                 sobol_problem[solver]['bounds'].extend(phit_bounds)
#
#             sobol_samples = sample(sobol_problem[solver], N=sampling)
#
#             # Process samples in single-core mode (no parallel processing)
#             results = _process_sample_chunk(sobol_samples, solver, t, phisc, phitsc, theta, thetac, cvp_initial, tsc, sobol_problem[solver], phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models)
#
#             sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)
#
#     # Construct return dictionary
#     sobol_output = {
#         'analysis': sa,
#         'problem': sobol_problem
#     }
#
#     return sobol_output
#
# def _sobol_analysis(
#         settings, model_outputs, calc_2nd_order=True, resamples=100, confidence=0.95,
#         use_parallel=False, n_cores=None, keep_resamples=False, rng_seed=None):
#     """
#     Perform Sobol sensitivity analysis on the given model outputs.
#
#     Parameters:
#     settings (dict): Dictionary containing the settings for the analysis.
#     model_outputs (numpy.ndarray): Array of model outputs to analyze.
#     calc_2nd_order (bool, optional): Whether to calculate second-order indices. Default is True.
#     resamples (int, optional): Number of resamples for confidence interval estimation. Default is 100.
#     confidence (float, optional): Confidence level for the intervals. Default is 0.95.
#     use_parallel (bool, optional): Whether to use parallel processing. Default is False.
#     n_cores (int, optional): Number of cores to use for parallel processing. Default is None.
#     keep_resamples (bool, optional): Whether to keep resamples for confidence interval calculation. Default is False.
#     rng_seed (int, optional): Seed for the random number generator. Default is None.
#
#     Returns:
#     dict: Dictionary containing the Sobol sensitivity indices and confidence intervals.
#     """
#     if rng_seed:
#         rng = np.random.default_rng(rng_seed).integers
#     else:
#         rng = np.random.randint
#
#     _, num_params = _extract_unique_groups(settings)
#
#     if calc_2nd_order and model_outputs.size % (2 * num_params + 2) == 0:
#         num_samples = int(model_outputs.size / (2 * num_params + 2))
#     elif not calc_2nd_order and model_outputs.size % (num_params + 2) == 0:
#         num_samples = int(model_outputs.size / (num_params + 2))
#     else:
#         raise RuntimeError("Sample mismatch between the model outputs and settings")
#
#     model_outputs = (model_outputs - model_outputs.mean()) / model_outputs.std()
#
#     A, B, AB, BA = _split_output_values(model_outputs, num_params, num_samples, calc_2nd_order)
#
#     resample_indices = rng(num_samples, size=(num_samples, resamples))
#     z_score = norm.ppf(0.5 + confidence / 2)
#
#     results = _initialize_sensitivity_storage(num_params, resamples, keep_resamples, calc_2nd_order)
#
#     for j in range(num_params):
#         # First-order Sobol index
#         results["S1"][j] = _compute_first_order_sensitivity(A, AB[:, j], B)
#
#         # Total-order Sobol index
#         results["ST"][j] = _compute_total_order_sensitivity(A, AB[:, j], B)
#
#         if keep_resamples:
#             # Only calculate confidence intervals if keep_resamples is True
#             results["S1_conf_all"][:, j] = _compute_first_order_sensitivity(
#                 A[resample_indices], AB[resample_indices, j], B[resample_indices])
#             results["ST_conf_all"][:, j] = _compute_total_order_sensitivity(
#                 A[resample_indices], AB[resample_indices, j], B[resample_indices])
#
#             # Confidence intervals
#             results["S1_conf"][j] = z_score * np.std(results["S1_conf_all"][:, j], ddof=1)
#             results["ST_conf"][j] = z_score * np.std(results["ST_conf_all"][:, j], ddof=1)
#         else:
#             # Set confidence intervals to zero or NaN if keep_resamples is False
#             results["S1_conf"][j] = np.nan
#             results["ST_conf"][j] = np.nan
#
#     return results
#
# def _split_output_values(output_values, num_params, num_samples, calc_2nd_order):
#     """
#     Split the output values into matrices A, B, AB, and BA for Sobol sensitivity analysis.
#
#     Parameters:
#     output_values (numpy.ndarray): Array of model outputs to be split.
#     num_params (int): Number of parameters in the model.
#     num_samples (int): Number of samples in the output values.
#     calc_2nd_order (bool): Whether to calculate second-order indices.
#
#     Returns:
#     tuple: A tuple containing matrices A, B, AB, and BA.
#            - A (numpy.ndarray): Matrix of output values for the first set of samples.
#            - B (numpy.ndarray): Matrix of output values for the second set of samples.
#            - AB (numpy.ndarray): Matrix of output values for the first-order indices.
#            - BA (numpy.ndarray or None): Matrix of output values for the second-order indices (if calc_2nd_order is True).
#     """
#     AB_matrix = np.zeros((num_samples, num_params))
#     BA_matrix = np.zeros((num_samples, num_params)) if calc_2nd_order else None
#     step_size = 2 * num_params + 2 if calc_2nd_order else num_params + 2
#
#     A = output_values[::step_size]
#     B = output_values[(step_size - 1)::step_size]
#
#     for j in range(num_params):
#         AB_matrix[:, j] = output_values[(j + 1)::step_size]
#         if calc_2nd_order:
#             BA_matrix[:, j] = output_values[(j + 1 + num_params)::step_size]
#
#     return A, B, AB_matrix, BA_matrix
#
# def _initialize_sensitivity_storage(num_params, resample_count, keep_resamples, calc_2nd_order):
#     """
#     Initialize storage for sensitivity analysis results.
#
#     Parameters:
#     num_params (int): Number of parameters in the model.
#     resample_count (int): Number of resamples for confidence interval estimation.
#     keep_resamples (bool): Whether to keep resamples for confidence interval calculation.
#     calc_2nd_order (bool): Whether to calculate second-order indices.
#
#     Returns:
#     dict: Dictionary containing initialized storage for sensitivity indices and confidence intervals.
#     """
#     results = {
#         "S1": np.zeros(num_params),
#         "S1_conf": np.zeros(num_params),
#         "ST": np.zeros(num_params),
#         "ST_conf": np.zeros(num_params)
#     }
#
#     if keep_resamples:
#         results["S1_conf_all"] = np.zeros((resample_count, num_params))
#         results["ST_conf_all"] = np.zeros((resample_count, num_params))
#
#     if calc_2nd_order:
#         results["S2"] = np.full((num_params, num_params), np.nan)
#         results["S2_conf"] = np.full((num_params, num_params), np.nan)
#
#     return results
#
# def _compute_first_order_sensitivity(A, AB, B):
#     """
#     Compute the first-order Sobol sensitivity index.
#
#     Parameters:
#     A (numpy.ndarray): Matrix of output values for the first set of samples.
#     AB (numpy.ndarray): Matrix of output values for the first-order indices.
#     B (numpy.ndarray): Matrix of output values for the second set of samples.
#
#     Returns:
#     numpy.ndarray: First-order Sobol sensitivity index.
#     """
#     return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
#
# def _compute_total_order_sensitivity(A, AB, B):
#     """
#     Compute the total-order Sobol sensitivity index.
#
#     Parameters:
#     A (numpy.ndarray): Matrix of output values for the first set of samples.
#     AB (numpy.ndarray): Matrix of output values for the first-order indices.
#     B (numpy.ndarray): Matrix of output values for the second set of samples.
#
#     Returns:
#     numpy.ndarray: Total-order Sobol sensitivity index.
#     """
#     return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)
#
# def _validate_group_membership(settings):
#     """
#     Validate the group membership definitions in the settings.
#
#     Parameters:
#     settings (dict): Dictionary containing the settings for the analysis.
#
#     Returns:
#     bool: True if the group definitions are valid, False otherwise.
#     """
#     group_definitions = settings.get("groups")
#     if not group_definitions or group_definitions == settings["names"]:
#         return False
#     return len(set(group_definitions)) > 1
#
# def _rescale_parameters(sample_set, problem_def):
#     """
#     Rescale the sample set based on the problem definition.
#
#     Parameters:
#     sample_set (numpy.ndarray): The set of samples to be rescaled.
#     problem_def (dict): The problem definition containing bounds and optional distributions.
#
#     Returns:
#     numpy.ndarray: The rescaled sample set.
#     """
#     boundaries = problem_def["bounds"]
#     dist_definitions = problem_def.get("dists")
#
#     if dist_definitions is None:
#         _apply_bounds(sample_set, boundaries)
#     else:
#         sample_set = _apply_distribution(sample_set, boundaries, dist_definitions)
#
#     problem_def["sample_scaled"] = True
#     return sample_set
#
# def _apply_bounds(sample_set, bounds):
#     """
#     Rescale the sample set based on the provided bounds.
#
#     Parameters:
#     sample_set (numpy.ndarray): The set of samples to be rescaled.
#     bounds (list of tuple): List of tuples containing the lower and upper bounds for each parameter.
#
#     Raises:
#     ValueError: If any lower bound is greater than or equal to the corresponding upper bound.
#     """
#     boundary_matrix = np.array(bounds)
#     lower_bounds = boundary_matrix[:, 0]
#     upper_bounds = boundary_matrix[:, 1]
#
#     if np.any(lower_bounds >= upper_bounds):
#         raise ValueError("Bounds are invalid: upper bound must be greater than lower bound")
#
#     np.add(
#         np.multiply(sample_set, (upper_bounds - lower_bounds), out=sample_set),
#         lower_bounds,
#         out=sample_set,
#     )
#
# def _apply_distribution(sample_set, bounds, distributions):
#     """
#     Rescale the sample set based on the provided bounds and distributions.
#
#     Parameters:
#     sample_set (numpy.ndarray): The set of samples to be rescaled.
#     bounds (list of tuple): List of tuples containing the lower and upper bounds for each parameter.
#     distributions (list of str): List of distribution types for each parameter.
#
#     Returns:
#     numpy.ndarray: The rescaled sample set.
#     """
#     boundary_matrix = np.array(bounds, dtype=object)
#     scaled_params = np.empty_like(sample_set)
#
#     for i in range(scaled_params.shape[1]):
#         lower, upper = boundary_matrix[i][0], boundary_matrix[i][1]
#
#         if distributions[i] == "triang":
#             scaled_params[:, i] = sp.stats.triang.ppf(
#                 sample_set[:, i], c=upper, scale=lower
#             )
#         elif distributions[i] == "unif":
#             scaled_params[:, i] = sample_set[:, i] * (upper - lower) + lower
#         elif distributions[i] == "logunif":
#             scaled_params[:, i] = sp.stats.loguniform.ppf(sample_set[:, i], a=lower, b=upper)
#         elif distributions[i] == "norm":
#             scaled_params[:, i] = sp.stats.norm.ppf(sample_set[:, i], loc=lower, scale=upper)
#         elif distributions[i] == "lognorm":
#             scaled_params[:, i] = np.exp(sp.stats.norm.ppf(sample_set[:, i], loc=lower, scale=upper))
#         else:
#             raise ValueError("Unsupported distribution type. Choose from ['unif', 'triang', 'norm', 'lognorm']")
#
#     return scaled_params
#
# def _extract_unique_groups(settings):
#     """
#     Extract unique group names from the settings.
#
#     Parameters:
#     settings (dict): Dictionary containing the settings for the analysis.
#
#     Returns:
#     tuple: A tuple containing the list of unique group names and their count.
#     """
#     group_members = settings.get("groups", settings["names"])
#     unique_group_set = list(pd.unique(np.asarray(group_members)))
#
#     return unique_group_set, len(unique_group_set)
#
# def _generate_group_matrix(groups):
#     """
#     Generate a membership matrix for the given groups.
#
#     Parameters:
#     groups (list): List of group names for each parameter.
#
#     Returns:
#     tuple: A tuple containing the membership matrix and the list of unique group names.
#     """
#     group_array = np.asarray(groups)
#     unique_group_list = pd.unique(group_array)
#     group_index_map = {group_name: idx for idx, group_name in enumerate(unique_group_list)}
#     membership_matrix = np.zeros((len(groups), len(unique_group_list)), dtype=int)
#
#     for param_idx, group_name in enumerate(groups):
#         membership_matrix[param_idx, group_index_map[group_name]] = 1
#
#     return membership_matrix, unique_group_list
#
# def sample(settings, N, calc_second_order=True, scramble=True, skip_values=0, seed=None):
#     """
#     Generate a sample matrix for Sobol sensitivity analysis.
#
#     Parameters:
#     settings (dict): Dictionary containing the settings for the analysis.
#     N (int): Number of samples to generate.
#     calc_second_order (bool, optional): Whether to calculate second-order indices. Default is True.
#     scramble (bool, optional): Whether to scramble the Sobol sequence. Default is True.
#     skip_values (int, optional): Number of values to skip in the Sobol sequence. Default is 0.
#     seed (int, optional): Seed for the random number generator. Default is None.
#
#     Returns:
#     numpy.ndarray: The generated sample matrix.
#     """
#     num_vars = settings["num_vars"]
#     groups = _validate_group_membership(settings)
#     sobol_sequence = qmc.Sobol(d=2 * num_vars, scramble=scramble, seed=seed)
#
#     if skip_values > 0 and isinstance(skip_values, int):
#         sobol_sequence.fast_forward(skip_values)
#     elif skip_values < 0 or not isinstance(skip_values, int):
#         raise ValueError("skip_values must be a non-negative integer.")
#
#     base_samples = sobol_sequence.random(N)
#     if not groups:
#         dim_groups = settings["num_vars"]
#     else:
#         group_matrix, group_names = _generate_group_matrix(groups)
#         dim_groups = len(group_names)
#
#     total_samples = 2 * dim_groups + 2 if calc_second_order else dim_groups + 2
#     sample_matrix = np.zeros((total_samples * N, num_vars))
#
#     index = 0
#     for i in range(N):
#         sample_matrix[index] = base_samples[i, :num_vars]  # Copy A matrix
#         index += 1
#
#         for k in range(dim_groups):
#             for j in range(num_vars):
#                 sample_matrix[index, j] = base_samples[i, j + num_vars] if (not groups and j == k) else base_samples[
#                     i, j]
#             index += 1
#
#         if calc_second_order:
#             for k in range(dim_groups):
#                 for j in range(num_vars):
#                     sample_matrix[index, j] = base_samples[i, j] if (not groups and j == k) else base_samples[
#                         i, j + num_vars]
#                 index += 1
#
#         sample_matrix[index] = base_samples[i, num_vars:]  # Copy B matrix
#         index += 1
#
#     return _rescale_parameters(sample_matrix, settings)
#
# def _process_sample_chunk(sobol_sample_chunk, solver, t, phisc, phitsc, theta, thetac, piecewise_func, tsc, sobol_problem, phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models):
#     """
#     Process a chunk of Sobol samples for sensitivity analysis.
#
#     Parameters:
#     sobol_sample_chunk (numpy.ndarray): Chunk of Sobol samples to process.
#     model (function): Solver function to use for simulation.
#     simulate (function): Simulation function to use.
#     t (dictionary): Dictionary of time points for the simulation for each variable.
#     phisc (dict): Dictionary of scaling factors for time-invariant variables.
#     phitsc (dict): Dictionary of scaling factors for time-variant variables.
#     theta (list): List of parameter values.
#     thetac (list): List of scaling factors for parameters.
#     piecewise_func (dictionary): Piecewise function to use in the simulation for each time-variant variable.
#     tsc (int): Time scaling factor.
#     sobol_problem (dict): Dictionary containing the Sobol problem definition.
#     phi_nom (list): List of nominal values for time-invariant variables.
#     phit_nom (list): List of nominal values for time-variant variables.
#     system (dict): Dictionary containing the model structure.
#     var_sensitivity (bool): Whether to perform variable sensitivity analysis.
#     par_sensitivity (bool): Whether to perform parameter sensitivity analysis.
#
#     Returns:
#     list: List of simulation results for each set of parameters in the chunk.
#     """
#     results = []
#
#     for params in sobol_sample_chunk:
#         if par_sensitivity and not var_sensitivity:
#             # For parameter sensitivity only
#             phi = {key: value for key, value in zip(system['tii'].keys(), phi_nom)}
#             phit = {key: value for key, value in zip(system['tvi'].keys(), phit_nom)}
#             params_split = params[:len(theta)]
#         elif var_sensitivity and not par_sensitivity:
#             # For variable sensitivity only
#             params_split = theta  # Use nominal theta
#             var_split = params
#             phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
#             phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}
#         else:
#             # For both parameter and variable sensitivity
#             params_split = params[:len(theta)]
#             var_split = params[len(theta):]
#             phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
#             phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}
#
#         # Ensure tvi values are floats
#         phit = {key: float(value) for key, value in phit.items()}
#
#         tv_ophi, ti_ophi, phit_interp = simula(t, {}, phisc, phi, phit, tsc, params_split, thetac, piecewise_func, phitsc, solver, system, models)
#
#         results.append(tv_ophi)
#
#     return results
#
# def _process_results(solver, results, t, sobol_problem_solver, plt_show):
#     """
#     Process the results of Sobol sensitivity analysis.
#
#     Parameters:
#     model (str): The model used for the analysis.
#     results (list): List of simulation results for each set of parameters.
#     t (list): List of time points for the simulation.
#     sobol_problem_solver (dict): Dictionary containing the Sobol problem definition for the model.
#     framework_settings (dict): Dictionary containing the settings for the framework.
#
#     Returns:
#     tuple: A tuple containing the Sobol analysis results and the Sobol analysis results excluding the first time point.
#     """
#     sobol_analysis_results = {}
#     SA = {}
#     sobol_analysis_results_excluded = {}
#
#     for response_key in results[0].keys():
#         response_values = np.array([result[response_key] for result in results])
#
#         sobol_analysis_results[solver] = [_sobol_analysis(sobol_problem_solver, response_values[:, j]) for j in range(len(t))]
#         t_excluded = t[1:]
#         sobol_analysis_results_excluded[solver] = sobol_analysis_results[solver][1:]
#         if plt_show:
#             plot_sobol_results(t_excluded, sobol_analysis_results_excluded[solver], sobol_problem_solver, solver, response_key)
#         SA[response_key] = sobol_analysis_results_excluded[solver]
#     return SA

# sc_sensa.py

import numpy as np
from scipy.stats import norm, qmc
import pandas as pd
import scipy as sp
from concurrent.futures import ProcessPoolExecutor
import os
from middoe.krnl_simula import simula
from middoe.iden_utils import plot_sobol_results
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def sensa(gsa, models, system):
    """
    Perform global Sobol sensitivity analysis to quantify parameter and variable importance.

    This function implements variance-based global sensitivity analysis using Sobol indices,
    identifying which parameters and input variables most influence model predictions.
    It supports both single-core and multi-core parallel execution, making it suitable
    for computationally expensive models. The analysis generates time-varying sensitivity
    indices showing how parameter importance evolves during dynamic experiments.

    Parameters
    ----------
    gsa : dict
        Global sensitivity analysis configuration:
            - 'tii_n' : list[float]
                Nominal values for time-invariant inputs.
            - 'tvi_n' : list[float]
                Nominal values for time-variant inputs.
            - 'samp' : int
                Number of Sobol samples (N). Total simulations = N*(2D+2) where D = num inputs.
            - 'var_s' : bool
                Enable variable sensitivity analysis.
            - 'par_s' : bool
                Enable parameter sensitivity analysis.
            - 'plt' : bool
                Generate Sobol plots automatically.
            - 'var_d' : float, optional
                Variable variation factor (e.g., 1.2 = ±20%). If not provided, uses bounds.
            - 'par_d' : float, optional
                Parameter variation factor. If not provided, uses t_l and t_u bounds.
            - 'multi' : False or float, optional
                Parallel execution: False=single-core, 0.0-1.0=fraction of CPUs to use.
    models : dict
        Model definitions:
            - 'can_m' : list[str]
                Active model/solver names.
            - 'theta' : dict[str, list[float]]
                Nominal parameter values.
            - 't_u' : dict[str, list[float]]
                Parameter upper bounds.
            - 't_l' : dict[str, list[float]]
                Parameter lower bounds.
    system : dict
        System configuration:
            - 't_s' : tuple[float, float]
                Time span.
            - 'tvi' : dict
                Time-variant input definitions.
            - 'tii' : dict
                Time-invariant input definitions with 'min' and 'max' bounds.
            - 'tvo' : dict
                Time-variant output definitions.

    Returns
    -------
    sobol_output : dict
        Complete Sobol analysis results:
            - 'analysis' : dict[str, dict[str, list[dict]]]
                Nested structure: {solver: {response: [time_step_results]}}
                Each time_step_results contains:
                    * 'S1' : np.ndarray — First-order indices (main effects)
                    * 'ST' : np.ndarray — Total indices (main + interactions)
                    * 'S1_conf' : np.ndarray — S1 confidence intervals
                    * 'ST_conf' : np.ndarray — ST confidence intervals
            - 'problem' : dict[str, dict]
                Sobol problem definitions: {solver: {'num_vars', 'names', 'bounds'}}

    Notes
    -----
    **Sobol Sensitivity Indices**:
    Variance-based global sensitivity analysis decomposes output variance into contributions
    from individual inputs and their interactions.

    **First-Order Index (S1)**:
    Main effect of input i, excluding interactions:
        \[
        S_i = \frac{V_{X_i}[E_{X_{\sim i}}(Y|X_i)]}{V(Y)}
        \]
    Interpretation: Fraction of output variance due to varying input i alone.

    **Total Index (ST)**:
    Total effect including all interactions:
        \[
        ST_i = \frac{E_{X_{\sim i}}[V_{X_i}(Y|X_{\sim i})]}{V(Y)}
        \]
    Interpretation: Fraction of variance that would remain if all inputs except i were fixed.

    **Computational Cost**:
    For N samples, D inputs: N × (2D + 2) simulations required.

    References
    ----------
    .. [1] Sobol', I. M. (2001).
       Global sensitivity indices for nonlinear mathematical models.
       *Mathematics and Computers in Simulation*, 55(1-3), 271-280.

    .. [2] Saltelli, A., et al. (2010).
       Variance based sensitivity analysis of model output.
       *Computer Physics Communications*, 181(2), 259-270.

    See Also
    --------
    sample : Generates Sobol quasi-random samples.
    _sobol_analysis : Computes Sobol indices from model outputs.

    Examples
    --------
    >>> gsa = {
    ...     'tii_n': [1.5, 300],
    ...     'tvi_n': [320],
    ...     'samp': 1000,
    ...     'var_s': True,
    ...     'par_s': True,
    ...     'plt': True,
    ...     'multi': 0.8
    ... }
    >>> results = sensa(gsa, models, system)
    >>> ST = results['analysis']['M1']['CA'][50]['ST']
    """
    phi_nom = gsa['tii_n']
    phit_nom = gsa['tvi_n']
    sampling = gsa['samp']
    var_sensitivity = gsa['var_s']
    par_sensitivity = gsa['par_s']
    plt_show = gsa['plt']
    active_solvers = models['can_m']
    theta_parameters = models['theta']
    thetamaxd = models['t_u']
    thetamind = models['t_l']

    cvp_initial = {var: 'no_CVP' for var in system['tvi'].keys()}
    parallel_value = gsa.get('multi', False)

    time_field = system['t_s'][1]
    t = list(np.linspace(0, 1, 305))
    tsc = int(time_field)

    sobol_problem = {}
    sa = {}

    if isinstance(parallel_value, float):
        # MULTI-CORE EXECUTION
        power = parallel_value
        total_cpus = os.cpu_count()
        num_workers = int(np.floor(total_cpus * power))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for solver in active_solvers:
                print(f'Running GSA-Sobol for model: {solver} in parallel')
                theta = [float(val) for val in theta_parameters[solver]]
                thetamax = [float(m) for m in thetamaxd[solver]]
                thetamin = [float(m) for m in thetamind[solver]]

                phisc = {key: 1.0 for key in system['tii'].keys()}
                phitsc = {key: 1.0 for key in system['tvi'].keys()}
                thetac = [1.0 for _ in theta]

                if isinstance(gsa.get('var_d'), float):
                    phi_bounds = [
                        (gsa['tii_n'][i] / gsa['var_d'], gsa['tvi_n'][i] * gsa['var_d'])
                        for i in range(len(gsa['tii_n']))
                    ]
                    phit_bounds = [
                        (gsa['tii_n'][i] / gsa['var_d'], gsa['tvi_n'][i] * gsa['var_d'])
                        for i in range(len(gsa['tvi_n']))
                    ]
                else:
                    phi_bounds = [(var_attrs['min'], var_attrs['max']) for var_attrs in system['tii'].values()]
                    phit_bounds = [(var_attrs['min'], var_attrs['max']) for var_attrs in system['tvi'].values()]

                if isinstance(gsa.get('par_d'), float):
                    param_bounds = [(theta[i] / gsa['par_d'], theta[i] * gsa['par_d']) for i in range(len(theta))]
                else:
                    param_bounds = list(zip(thetamin, thetamax))

                sobol_problem[solver] = {'num_vars': 0, 'names': [], 'bounds': []}

                if par_sensitivity:
                    sobol_problem[solver]['num_vars'] += len(theta)
                    sobol_problem[solver]['names'].extend([f'Param_{i + 1}' for i in range(len(theta))])
                    sobol_problem[solver]['bounds'].extend(param_bounds)

                if var_sensitivity:
                    sobol_problem[solver]['num_vars'] += len(phi_nom) + len(phit_nom)
                    sobol_problem[solver]['names'].extend(list(system['tii'].keys()) + list(system['tvi'].keys()))
                    sobol_problem[solver]['bounds'].extend(phi_bounds)
                    sobol_problem[solver]['bounds'].extend(phit_bounds)

                sobol_samples = sample(sobol_problem[solver], N=sampling)

                chunk_size = len(sobol_samples) // num_workers
                sobol_sample_chunks = [sobol_samples[i:i + chunk_size] for i in range(0, len(sobol_samples), chunk_size)]

                futures = [
                    executor.submit(_process_sample_chunk, chunk, solver, t, phisc, phitsc, theta, thetac,
                                  cvp_initial, tsc, sobol_problem[solver], phi_nom, phit_nom, system,
                                  var_sensitivity, par_sensitivity, models)
                    for chunk in sobol_sample_chunks
                ]

                results = []
                for future in futures:
                    results.extend(future.result())

                sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)

    else:
        # SINGLE-CORE EXECUTION
        for solver in active_solvers:
            print(f'Running GSA-Sobol for model: {solver} in single-core')
            theta = [float(val) for val in theta_parameters[solver]]
            thetamax = [float(m) for m in thetamaxd[solver]]
            thetamin = [float(m) for m in thetamind[solver]]

            phisc = {key: 1.0 for key in system['tii'].keys()}
            phitsc = {key: 1.0 for key in system['tvi'].keys()}
            thetac = [1.0 for _ in theta]

            if isinstance(gsa.get('var_d'), float):
                phi_bounds = [(phi_nom[i] / gsa['var_d'], phi_nom[i] * gsa['var_d'])
                            for i, _ in enumerate(system['tii'].keys())]
                phit_bounds = [(phit_nom[i] / gsa['var_d'], phit_nom[i] * gsa['var_d'])
                             for i, _ in enumerate(system['tvi'].keys())]
            else:
                phi_bounds = [(attrs['min'], attrs['max']) for attrs in system['tii'].values()]
                phit_bounds = [(attrs['min'], attrs['max']) for attrs in system['tvi'].values()]

            if isinstance(gsa.get('par_d'), float):
                param_bounds = [(theta[i] / gsa['par_d'], theta[i] * gsa['par_d']) for i in range(len(theta))]
            else:
                param_bounds = list(zip(thetamin, thetamax))

            sobol_problem[solver] = {'num_vars': 0, 'names': [], 'bounds': []}

            if par_sensitivity:
                sobol_problem[solver]['num_vars'] += len(theta)
                sobol_problem[solver]['names'].extend([f'Param_{i + 1}' for i in range(len(theta))])
                sobol_problem[solver]['bounds'].extend(param_bounds)

            if var_sensitivity:
                sobol_problem[solver]['num_vars'] += len(phi_nom) + len(phit_nom)
                sobol_problem[solver]['names'].extend(list(system['tii'].keys()) + list(system['tvi'].keys()))
                sobol_problem[solver]['bounds'].extend(phi_bounds)
                sobol_problem[solver]['bounds'].extend(phit_bounds)

            sobol_samples = sample(sobol_problem[solver], N=sampling)

            results = _process_sample_chunk(
                sobol_samples, solver, t, phisc, phitsc, theta, thetac, cvp_initial, tsc,
                sobol_problem[solver], phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models
            )

            sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)

    sobol_output = {'analysis': sa, 'problem': sobol_problem}
    return sobol_output


def _sobol_analysis(settings, model_outputs, calc_2nd_order=True, resamples=100, confidence=0.95,
                   use_parallel=False, n_cores=None, keep_resamples=False, rng_seed=None):
    """
    Compute Sobol sensitivity indices from model outputs using variance decomposition.

    This is the core computational routine that estimates first-order (S1) and total-order (ST)
    Sobol indices from the structured model output matrix. It implements the efficient
    estimators from Saltelli et al. with optional bootstrap confidence intervals.

    Parameters
    ----------
    settings : dict
        Sobol problem definition with 'num_vars' and optional 'groups'.
    model_outputs : np.ndarray, shape (N*(2D+2),)
        Model evaluations at Sobol sample points (flattened).
    calc_2nd_order : bool, optional
        Whether outputs include second-order samples (default: True).
    resamples : int, optional
        Number of bootstrap resamples for confidence intervals (default: 100).
    confidence : float, optional
        Confidence level for intervals (default: 0.95).
    use_parallel : bool, optional
        Use parallel processing for bootstrap (default: False).
    n_cores : int, optional
        Number of cores for parallel bootstrap (default: None).
    keep_resamples : bool, optional
        Store all bootstrap samples (default: False).
    rng_seed : int, optional
        Random seed for bootstrap reproducibility (default: None).

    Returns
    -------
    results : dict
        Sobol indices and confidence intervals:
            - 'S1' : np.ndarray — First-order indices
            - 'ST' : np.ndarray — Total indices
            - 'S1_conf' : np.ndarray — S1 confidence intervals
            - 'ST_conf' : np.ndarray — ST confidence intervals

    Notes
    -----
    **First-Order Index Estimator**:
        \[
        \hat{S}_j = \frac{\frac{1}{N}\sum_{i=1}^N Y_B^{(i)} (Y_{AB_j}^{(i)} - Y_A^{(i)})}{\text{Var}(Y)}
        \]

    **Total Index Estimator**:
        \[
        \hat{ST}_j = \frac{\frac{1}{2N}\sum_{i=1}^N (Y_A^{(i)} - Y_{AB_j}^{(i)})^2}{\text{Var}(Y)}
        \]

    See Also
    --------
    sensa : Main function that calls this for each time point.
    _compute_first_order_sensitivity : S1 estimator.
    _compute_total_order_sensitivity : ST estimator.

    Examples
    --------
    >>> problem = {'num_vars': 3}
    >>> outputs = np.random.randn(800)  # 100 * (2*3 + 2)
    >>> results = _sobol_analysis(problem, outputs)
    >>> print(f"S1: {results['S1']}")
    >>> print(f"ST: {results['ST']}")
    """
    if rng_seed:
        rng = np.random.default_rng(rng_seed).integers
    else:
        rng = np.random.randint

    _, num_params = _extract_unique_groups(settings)

    if calc_2nd_order and model_outputs.size % (2 * num_params + 2) == 0:
        num_samples = int(model_outputs.size / (2 * num_params + 2))
    elif not calc_2nd_order and model_outputs.size % (num_params + 2) == 0:
        num_samples = int(model_outputs.size / (num_params + 2))
    else:
        raise RuntimeError("Sample mismatch between model outputs and settings")

    model_outputs = (model_outputs - model_outputs.mean()) / model_outputs.std()

    A, B, AB, BA = _split_output_values(model_outputs, num_params, num_samples, calc_2nd_order)

    resample_indices = rng(num_samples, size=(num_samples, resamples))
    z_score = norm.ppf(0.5 + confidence / 2)

    results = _initialize_sensitivity_storage(num_params, resamples, keep_resamples, calc_2nd_order)

    for j in range(num_params):
        results["S1"][j] = _compute_first_order_sensitivity(A, AB[:, j], B)
        results["ST"][j] = _compute_total_order_sensitivity(A, AB[:, j], B)

        if keep_resamples:
            results["S1_conf_all"][:, j] = _compute_first_order_sensitivity(
                A[resample_indices], AB[resample_indices, j], B[resample_indices])
            results["ST_conf_all"][:, j] = _compute_total_order_sensitivity(
                A[resample_indices], AB[resample_indices, j], B[resample_indices])

            results["S1_conf"][j] = z_score * np.std(results["S1_conf_all"][:, j], ddof=1)
            results["ST_conf"][j] = z_score * np.std(results["ST_conf_all"][:, j], ddof=1)
        else:
            results["S1_conf"][j] = np.nan
            results["ST_conf"][j] = np.nan

    return results


def _split_output_values(output_values, num_params, num_samples, calc_2nd_order):
    """
    Split model outputs into A, B, AB, and BA matrices for Sobol analysis.

    This function reorganizes the flat output array from structured Sobol sampling
    into separate matrices needed for index computation. The structured sampling
    interleaves A, AB, BA, and B matrices to enable efficient variance decomposition.

    Parameters
    ----------
    output_values : np.ndarray, shape (N*(2D+2),) or (N*(D+2),)
        Flattened model outputs from Sobol sample matrix.
    num_params : int
        Number of parameters (D) in the analysis.
    num_samples : int
        Base number of samples (N).
    calc_2nd_order : bool
        Whether second-order samples (BA) are included.

    Returns
    -------
    A : np.ndarray, shape (N,)
        Outputs for first parameter matrix.
    B : np.ndarray, shape (N,)
        Outputs for second parameter matrix.
    AB : np.ndarray, shape (N, D)
        Outputs for hybrid matrices (A with columns from B).
    BA : np.ndarray, shape (N, D) or None
        Outputs for hybrid matrices (B with columns from A), None if calc_2nd_order=False.

    Notes
    -----
    **Matrix Structure**:
    For each sample i, outputs are organized as:
        [A_i, AB_i1, AB_i2, ..., AB_iD, (BA_i1, ..., BA_iD), B_i]

    This structure enables computation of:
        - S1_j from covariance of A and AB_j
        - ST_j from variance of (A - AB_j)
        - S2_jk from covariance of AB_j and BA_k (if calc_2nd_order=True)

    **Step Size**:
        - With 2nd order: 2D + 2
        - Without 2nd order: D + 2

    See Also
    --------
    sample : Generates structured sample matrix.
    _sobol_analysis : Uses split matrices for index computation.

    Examples
    --------
    >>> outputs = np.random.randn(800)  # 100*(2*3+2)
    >>> A, B, AB, BA = _split_output_values(outputs, 3, 100, True)
    >>> print(A.shape)  # (100,)
    >>> print(AB.shape)  # (100, 3)
    >>> print(BA.shape)  # (100, 3)
    """
    AB_matrix = np.zeros((num_samples, num_params))
    BA_matrix = np.zeros((num_samples, num_params)) if calc_2nd_order else None
    step_size = 2 * num_params + 2 if calc_2nd_order else num_params + 2

    A = output_values[::step_size]
    B = output_values[(step_size - 1)::step_size]

    for j in range(num_params):
        AB_matrix[:, j] = output_values[(j + 1)::step_size]
        if calc_2nd_order:
            BA_matrix[:, j] = output_values[(j + 1 + num_params)::step_size]

    return A, B, AB_matrix, BA_matrix


def _initialize_sensitivity_storage(num_params, resample_count, keep_resamples, calc_2nd_order):
    """
    Initialize storage dictionaries for Sobol sensitivity results.

    Creates pre-allocated numpy arrays for storing sensitivity indices, confidence
    intervals, and optional bootstrap resamples. This avoids repeated memory
    allocation during bootstrap iterations.

    Parameters
    ----------
    num_params : int
        Number of parameters in the analysis.
    resample_count : int
        Number of bootstrap resamples for confidence intervals.
    keep_resamples : bool
        Whether to store all bootstrap samples (memory intensive).
    calc_2nd_order : bool
        Whether to allocate storage for second-order indices.

    Returns
    -------
    results : dict
        Dictionary with pre-allocated arrays:
            - 'S1' : np.ndarray, shape (num_params,) — First-order indices
            - 'ST' : np.ndarray, shape (num_params,) — Total indices
            - 'S1_conf' : np.ndarray, shape (num_params,) — S1 confidence intervals
            - 'ST_conf' : np.ndarray, shape (num_params,) — ST confidence intervals
            - 'S1_conf_all' : np.ndarray, shape (resample_count, num_params) — All S1 resamples (if keep_resamples)
            - 'ST_conf_all' : np.ndarray, shape (resample_count, num_params) — All ST resamples (if keep_resamples)
            - 'S2' : np.ndarray, shape (num_params, num_params) — Second-order indices (if calc_2nd_order)
            - 'S2_conf' : np.ndarray, shape (num_params, num_params) — S2 confidence (if calc_2nd_order)

    Notes
    -----
    **Memory Considerations**:
    With keep_resamples=True and many parameters/resamples:
        Memory ≈ 2 * num_params * resample_count * 8 bytes

    Example: 100 resamples, 20 params: ~32 KB per analysis point.
    For 305 time points: ~10 MB total.

    **Second-Order Indices**:
    S2 matrices are initialized with np.nan to distinguish zero interactions
    from uncomputed values.

    See Also
    --------
    _sobol_analysis : Uses this storage during computation.

    Examples
    --------
    >>> results = _initialize_sensitivity_storage(5, 100, True, True)
    >>> print(results['S1'].shape)  # (5,)
    >>> print(results['S1_conf_all'].shape)  # (100, 5)
    >>> print(results['S2'].shape)  # (5, 5)
    """
    results = {
        "S1": np.zeros(num_params),
        "S1_conf": np.zeros(num_params),
        "ST": np.zeros(num_params),
        "ST_conf": np.zeros(num_params)
    }

    if keep_resamples:
        results["S1_conf_all"] = np.zeros((resample_count, num_params))
        results["ST_conf_all"] = np.zeros((resample_count, num_params))

    if calc_2nd_order:
        results["S2"] = np.full((num_params, num_params), np.nan)
        results["S2_conf"] = np.full((num_params, num_params), np.nan)

    return results


def _compute_first_order_sensitivity(A, AB, B):
    """
    Compute first-order Sobol sensitivity index using Saltelli estimator.

    The first-order index S1_j measures the main effect contribution of parameter j
    to output variance, excluding all interaction terms. This implementation uses
    the efficient covariance-based estimator from Saltelli et al. (2010).

    Parameters
    ----------
    A : np.ndarray, shape (N,) or (B, N)
        Model outputs for first parameter matrix (or bootstrap resamples).
    AB : np.ndarray, shape (N,) or (B, N)
        Model outputs for hybrid matrix (A with column j from B).
    B : np.ndarray, shape (N,) or (B, N)
        Model outputs for second parameter matrix.

    Returns
    -------
    S1 : float or np.ndarray, shape (B,)
        First-order Sobol index (or bootstrap estimates if inputs are 2D).

    Notes
    -----
    **Estimator Formula**:
        \[
        \hat{S}_j = \frac{\text{Cov}(Y_B, Y_{AB_j})}{\text{Var}(Y)}
        \]
    where the covariance and variance are estimated from samples.

    **Implementation**:
    Uses the numerically stable form:
        \[
        \hat{S}_j = \frac{\frac{1}{N}\sum_i Y_B^{(i)} (Y_{AB_j}^{(i)} - Y_A^{(i)})}{\text{Var}([Y_A, Y_B])}
        \]

    **Interpretation**:
        - S1_j = 0: Parameter j has no main effect
        - S1_j = 1: All variance explained by j alone
        - S1_j < 0: Possible due to sampling noise (treat as ≈0)
        - ∑S1 > 1: Impossible (indicates numerical issues)
        - ∑S1 < 1: Model has interactions

    **Bootstrap Support**:
    If inputs are 2D (B bootstrap resamples × N samples), computes B
    independent S1 estimates for confidence interval estimation.

    References
    ----------
    .. [1] Saltelli, A., et al. (2010).
       Variance based sensitivity analysis of model output.
       *Computer Physics Communications*, 181(2), 259-270.

    See Also
    --------
    _compute_total_order_sensitivity : Total index computation.
    _sobol_analysis : Calls this function for each parameter.

    Examples
    --------
    >>> A = np.random.randn(1000)
    >>> B = np.random.randn(1000)
    >>> AB = 0.8 * A + 0.2 * np.random.randn(1000)
    >>> S1 = _compute_first_order_sensitivity(A, AB, B)
    >>> print(f"S1 ≈ 0.64: {S1:.3f}")  # Expected ~0.64
    """
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)


def _compute_total_order_sensitivity(A, AB, B):
    """
    Compute total-order Sobol sensitivity index.

    The total index ST_j measures the total contribution of parameter j to output
    variance, including its main effect and all interactions with other parameters.
    This implementation uses the efficient variance-based estimator.

    Parameters
    ----------
    A : np.ndarray, shape (N,) or (B, N)
        Model outputs for first parameter matrix.
    AB : np.ndarray, shape (N,) or (B, N)
        Model outputs for hybrid matrix (A with column j from B).
    B : np.ndarray, shape (N,) or (B, N)
        Model outputs for second parameter matrix.

    Returns
    -------
    ST : float or np.ndarray, shape (B,)
        Total Sobol index (or bootstrap estimates).

    Notes
    -----
    **Estimator Formula**:
        \[
        \hat{ST}_j = \frac{E[(Y_A - Y_{AB_j})^2] / 2}{\text{Var}(Y)}
        \]

    **Interpretation**:
        - ST_j = 0: Parameter j has no effect (main or interaction)
        - ST_j = 1: All variance involves j
        - ST_j ≈ S1_j: j acts independently (no interactions)
        - ST_j >> S1_j: j has strong interactions
        - ST_j - S1_j: Interaction contribution

    **Complementary Index**:
    The quantity (1 - ST_j) represents the variance that would remain if
    parameter j were known exactly (all uncertainty in j removed).

    **Bootstrap Support**:
    Handles 2D inputs for confidence interval estimation.

    References
    ----------
    .. [1] Homma, T., & Saltelli, A. (1996).
       Importance measures in global sensitivity analysis of nonlinear models.
       *Reliability Engineering & System Safety*, 52(1), 1-17.

    See Also
    --------
    _compute_first_order_sensitivity : First-order index.
    _sobol_analysis : Calls this for each parameter.

    Examples
    --------
    >>> # Parameter with both main effect and interactions
    >>> A = np.random.randn(1000)
    >>> B = np.random.randn(1000)
    >>> AB = 0.8 * A + 0.3 * A * B[:1000] + 0.2 * np.random.randn(1000)
    >>> ST = _compute_total_order_sensitivity(A, AB, B)
    >>> print(f"ST > S1 due to interactions: {ST:.3f}")
    """
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)


def _validate_group_membership(settings):
    """
    Check if group-based sensitivity analysis is requested and valid.

    Grouped sensitivity analysis treats multiple parameters as a single entity,
    useful for analyzing categories of parameters (e.g., all kinetic parameters
    vs all transport parameters).

    Parameters
    ----------
    settings : dict
        Sobol problem with optional 'groups' key.

    Returns
    -------
    has_groups : bool
        True if valid group definitions exist, False otherwise.

    Notes
    -----
    **Group Requirements**:
        - 'groups' key must exist in settings
        - Must have 2+ distinct groups
        - Cannot be identical to individual parameter names

    **Use Case Example**:
    For a reactor model with parameters [k1, k2, Ea1, Ea2, D1, D2]:
        groups = ['kinetic', 'kinetic', 'kinetic', 'kinetic', 'transport', 'transport']

    This computes sensitivity indices for:
        - Kinetic parameters as a group
        - Transport parameters as a group

    See Also
    --------
    _generate_group_matrix : Creates group membership matrix.
    sample : Uses groups for structured sampling.

    Examples
    --------
    >>> settings = {
    ...     'names': ['k1', 'k2', 'D'],
    ...     'groups': ['kinetic', 'kinetic', 'transport']
    ... }
    >>> print(_validate_group_membership(settings))  # True
    >>>
    >>> settings['groups'] = settings['names']  # No grouping
    >>> print(_validate_group_membership(settings))  # False
    """
    group_definitions = settings.get("groups")
    if not group_definitions or group_definitions == settings["names"]:
        return False
    return len(set(group_definitions)) > 1


def _rescale_parameters(sample_set, problem_def):
    """
    Transform samples from [0,1] to physical parameter space with optional distributions.

    This function applies bounds and statistical distributions to rescale normalized
    Sobol samples [0,1] into physical parameter ranges. It supports uniform and
    various non-uniform distributions.

    Parameters
    ----------
    sample_set : np.ndarray, shape (N, D)
        Sobol samples in [0,1] hypercube.
    problem_def : dict
        Problem definition with:
            - 'bounds' : list[tuple] — Parameter bounds
            - 'dists' : list[str], optional — Distribution types
            - 'sample_scaled' : bool — Flag set to True after scaling

    Returns
    -------
    sample_set : np.ndarray, shape (N, D)
        Rescaled samples in physical parameter space.

    Notes
    -----
    **Distribution Types**:
        - 'unif': Uniform over [min, max]
        - 'norm': Normal with mean=min, std=max
        - 'lognorm': Log-normal (log(X) ~ Normal)
        - 'triang': Triangular with mode=max, scale=min
        - 'logunif': Log-uniform over [min, max]

    **In-Place Modification**:
    If no 'dists' specified, modifies sample_set in-place for efficiency.
    Otherwise, creates new array.

    **Side Effect**:
    Sets problem_def['sample_scaled'] = True to prevent double-scaling.

    See Also
    --------
    _apply_bounds : Uniform rescaling.
    _apply_distribution : Non-uniform distributions.
    sample : Calls this function after generating Sobol sequence.

    Examples
    --------
    >>> samples = np.array([[0.1, 0.5], [0.9, 0.3]])
    >>> problem = {
    ...     'bounds': [(1, 10), (100, 500)],
    ...     'dists': ['unif', 'norm']  # Uniform and Normal
    ... }
    >>> scaled = _rescale_parameters(samples, problem)
    >>> print(scaled[0])  # [1.9, ~300] (Normal centered at 100)
    """
    boundaries = problem_def["bounds"]
    dist_definitions = problem_def.get("dists")

    if dist_definitions is None:
        _apply_bounds(sample_set, boundaries)
    else:
        sample_set = _apply_distribution(sample_set, boundaries, dist_definitions)

    problem_def["sample_scaled"] = True
    return sample_set


def _apply_bounds(sample_set, bounds):
    """
    Rescale uniform [0,1] samples to specified bounds via linear transformation.

    Applies the transformation: x_scaled = x * (upper - lower) + lower
    for each parameter. Modifies sample_set in-place for efficiency.

    Parameters
    ----------
    sample_set : np.ndarray, shape (N, D)
        Samples in [0,1] hypercube (modified in-place).
    bounds : list[tuple[float, float]]
        [(lower1, upper1), (lower2, upper2), ...] for each parameter.

    Raises
    ------
    ValueError
        If any lower bound >= upper bound.

    Notes
    -----
    **In-Place Operation**:
    For memory efficiency with large sample sets, rescaling is done in-place
    using NumPy's out parameter.

    **Uniform Distribution**:
    Maintains uniformity: if X ~ Uniform[0,1], then
    a + (b-a)X ~ Uniform[a,b].

    See Also
    --------
    _rescale_parameters : Main rescaling function.
    _apply_distribution : Non-uniform distributions.

    Examples
    --------
    >>> samples = np.array([[0.0, 0.5], [1.0, 0.25]])
    >>> bounds = [(10, 20), (0, 100)]
    >>> _apply_bounds(samples, bounds)
    >>> print(samples)
    # [[10, 50], [20, 25]]
    """
    boundary_matrix = np.array(bounds)
    lower_bounds = boundary_matrix[:, 0]
    upper_bounds = boundary_matrix[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are invalid: upper bound must be greater than lower bound")

    np.add(
        np.multiply(sample_set, (upper_bounds - lower_bounds), out=sample_set),
        lower_bounds,
        out=sample_set,
    )


def _apply_distribution(sample_set, bounds, distributions):
    """
    Transform uniform [0,1] samples to specified non-uniform distributions.

    Uses inverse CDF (quantile) transformation to convert uniform samples
    into samples from target distributions. Each parameter can have a different
    distribution type.

    Parameters
    ----------
    sample_set : np.ndarray, shape (N, D)
        Uniform samples in [0,1] hypercube.
    bounds : list[tuple]
        Parameter bounds. Interpretation depends on distribution type.
    distributions : list[str]
        Distribution types: 'unif', 'norm', 'lognorm', 'triang', 'logunif'.

    Returns
    -------
    scaled_params : np.ndarray, shape (N, D)
        Samples from specified distributions.

    Raises
    ------
    ValueError
        If unsupported distribution type specified.

    Notes
    -----
    **Inverse Transform Sampling**:
    If U ~ Uniform[0,1] and F is CDF, then F^(-1)(U) follows distribution F.

    **Distribution Parameters** (for bounds = (a, b)):
        - 'unif': min=a, max=b
        - 'norm': mean=a, std=b
        - 'lognorm': log_mean=a, log_std=b
        - 'triang': scale=a, mode=b
        - 'logunif': min=a, max=b (log-scale uniform)

    **Log-Uniform**:
    Useful for parameters spanning orders of magnitude (e.g., rate constants).
    Uniform on log-scale: log(X) ~ Uniform[log(a), log(b)].

    See Also
    --------
    _rescale_parameters : Main rescaling function.
    _apply_bounds : Uniform distribution.

    Examples
    --------
    >>> samples = np.array([[0.5], [0.9]])
    >>> bounds = [(100, 10)]  # mean=100, std=10 for Normal
    >>> dists = ['norm']
    >>> scaled = _apply_distribution(samples, bounds, dists)
    >>> # scaled ≈ [[100], [112.8]]  # 90th percentile
    """
    boundary_matrix = np.array(bounds, dtype=object)
    scaled_params = np.empty_like(sample_set)

    for i in range(scaled_params.shape[1]):
        lower, upper = boundary_matrix[i][0], boundary_matrix[i][1]

        if distributions[i] == "triang":
            scaled_params[:, i] = sp.stats.triang.ppf(
                sample_set[:, i], c=upper, scale=lower
            )
        elif distributions[i] == "unif":
            scaled_params[:, i] = sample_set[:, i] * (upper - lower) + lower
        elif distributions[i] == "logunif":
            scaled_params[:, i] = sp.stats.loguniform.ppf(sample_set[:, i], a=lower, b=upper)
        elif distributions[i] == "norm":
            scaled_params[:, i] = sp.stats.norm.ppf(sample_set[:, i], loc=lower, scale=upper)
        elif distributions[i] == "lognorm":
            scaled_params[:, i] = np.exp(sp.stats.norm.ppf(sample_set[:, i], loc=lower, scale=upper))
        else:
            raise ValueError("Unsupported distribution type. Choose from ['unif', 'triang', 'norm', 'lognorm']")

    return scaled_params


def _extract_unique_groups(settings):
    """
    Extract unique group names and count from sensitivity analysis settings.

    For grouped sensitivity analysis, identifies the distinct groups defined
    for parameters. For individual analysis, treats each parameter as its own group.

    Parameters
    ----------
    settings : dict
        Sobol problem with:
            - 'names' : list[str] — Parameter names
            - 'groups' : list[str], optional — Group assignments

    Returns
    -------
    unique_groups : list[str]
        List of unique group names.
    num_groups : int
        Number of unique groups.

    Notes
    -----
    **Default Behavior**:
    If 'groups' not specified, returns parameter names as groups (individual analysis).

    **Example**:
    For parameters ['k1', 'k2', 'D1', 'D2'] with groups ['kinetic', 'kinetic', 'transport', 'transport']:
        Returns: (['kinetic', 'transport'], 2)

    See Also
    --------
    _generate_group_matrix : Creates membership matrix from groups.
    _validate_group_membership : Checks if grouping is valid.

    Examples
    --------
    >>> settings = {
    ...     'names': ['k1', 'k2', 'D'],
    ...     'groups': ['kinetic', 'kinetic', 'transport']
    ... }
    >>> groups, n = _extract_unique_groups(settings)
    >>> print(groups)  # ['kinetic', 'transport']
    >>> print(n)  # 2
    """
    group_members = settings.get("groups", settings["names"])
    unique_group_set = list(pd.unique(np.asarray(group_members)))
    return unique_group_set, len(unique_group_set)


def _generate_group_matrix(groups):
    """
    Create binary membership matrix for grouped sensitivity analysis.

    Constructs a matrix indicating which parameters belong to which groups.
    This enables computation of group-level sensitivity indices by summing
    contributions of group members.

    Parameters
    ----------
    groups : list[str]
        Group assignment for each parameter. Length = number of parameters.

    Returns
    -------
    membership_matrix : np.ndarray, shape (n_params, n_groups)
        Binary matrix where membership_matrix[i, j] = 1 if parameter i belongs to group j.
    unique_groups : np.ndarray, shape (n_groups,)
        Array of unique group names.

    Notes
    -----
    **Matrix Structure**:
    Each row corresponds to a parameter, each column to a group.
    Element (i,j) = 1 if parameter i is in group j, else 0.

    **Example**:
    For params [k1, k2, D1, D2] with groups ['kinetic', 'kinetic', 'transport', 'transport']:
        membership_matrix = [[1, 0],
                            [1, 0],
                            [0, 1],
                            [0, 1]]
        unique_groups = ['kinetic', 'transport']

    **Use in Sampling**:
    The membership matrix guides construction of hybrid sample matrices
    for group-level sensitivity.

    See Also
    --------
    _extract_unique_groups : Gets unique group list.
    sample : Uses membership matrix for grouped sampling.

    Examples
    --------
    >>> groups = ['A', 'A', 'B', 'A']
    >>> matrix, names = _generate_group_matrix(groups)
    >>> print(matrix)
    # [[1 0]
    #  [1 0]
    #  [0 1]
    #  [1 0]]
    >>> print(names)  # ['A' 'B']
    """
    group_array = np.asarray(groups)
    unique_group_list = pd.unique(group_array)
    group_index_map = {group_name: idx for idx, group_name in enumerate(unique_group_list)}
    membership_matrix = np.zeros((len(groups), len(unique_group_list)), dtype=int)

    for param_idx, group_name in enumerate(groups):
        membership_matrix[param_idx, group_index_map[group_name]] = 1

    return membership_matrix, unique_group_list


def sample(settings, N, calc_second_order=True, scramble=True, skip_values=0, seed=None):
    """
    Generate structured Sobol sample matrix for variance-based sensitivity analysis.

    Creates a low-discrepancy quasi-random sample matrix using Sobol sequences,
    structured to enable efficient computation of first-order and total-order
    sensitivity indices. Supports grouped analysis and various sampling options.

    Parameters
    ----------
    settings : dict
        Sobol problem definition:
            - 'num_vars' : int — Number of parameters
            - 'bounds' : list[tuple] — [(min, max), ...] for each parameter
            - 'dists' : list[str], optional — Distribution types
            - 'groups' : list[str], optional — Group assignments
    N : int
        Base number of samples. Total simulations = N × (2D + 2) where D = num_vars.
    calc_second_order : bool, optional
        Generate samples for second-order indices (default: True).
        If False, reduces to N × (D + 2) simulations.
    scramble : bool, optional
        Apply Owen scrambling to Sobol sequence (default: True).
    skip_values : int, optional
        Skip initial Sobol points (default: 0).
    seed : int, optional
        Random seed for scrambling (default: None).

    Returns
    -------
    sample_matrix : np.ndarray, shape ((2D+2)*N, D) or ((D+2)*N, D)
        Structured sample matrix with interleaved A, AB, BA (optional), B matrices.

    Notes
    -----
    **Sample Matrix Structure**:
    For each base sample i, generates:
        [A_i, AB_i1, ..., AB_iD, (BA_i1, ..., BA_iD), B_i]
    where AB_ij is A with column j replaced by B.

    **Sobol Sequences**:
    Low-discrepancy quasi-random sequences with superior space-filling vs random:
        - Uniform coverage of parameter space
        - Faster convergence of Monte Carlo estimates
        - Deterministic (given seed)

    **Total Simulations**:
        - With 2nd order: N × (2D + 2)
        - Without 2nd order: N × (D + 2)

    Example for N=100, D=5:
        - With 2nd order: 1,200 simulations
        - Without: 700 simulations

    **Scrambling**:
    Owen scrambling randomizes while preserving low-discrepancy properties.
    Enables confidence interval estimation via multiple scrambled sequences.

    See Also
    --------
    _rescale_parameters : Applies bounds and distributions.
    _generate_group_matrix : For grouped analysis.
    sensa : Main function that calls this.

    Examples
    --------
    >>> problem = {
    ...     'num_vars': 3,
    ...     'bounds': [(0, 1), (100, 500), (1e3, 1e6)]
    ... }
    >>> samples = sample(problem, N=100, seed=42)
    >>> print(samples.shape)  # (800, 3) = 100*(2*3+2)
    >>>
    >>> # With distributions
    >>> problem['dists'] = ['unif', 'norm', 'logunif']
    >>> samples = sample(problem, N=100)
    """
    num_vars = settings["num_vars"]
    groups = _validate_group_membership(settings)
    sobol_sequence = qmc.Sobol(d=2 * num_vars, scramble=scramble, seed=seed)

    if skip_values > 0 and isinstance(skip_values, int):
        sobol_sequence.fast_forward(skip_values)
    elif skip_values < 0 or not isinstance(skip_values, int):
        raise ValueError("skip_values must be a non-negative integer.")

    base_samples = sobol_sequence.random(N)

    if not groups:
        dim_groups = settings["num_vars"]
    else:
        group_matrix, group_names = _generate_group_matrix(groups)
        dim_groups = len(group_names)

    total_samples = 2 * dim_groups + 2 if calc_second_order else dim_groups + 2
    sample_matrix = np.zeros((total_samples * N, num_vars))

    index = 0
    for i in range(N):
        sample_matrix[index] = base_samples[i, :num_vars]  # A
        index += 1

        for k in range(dim_groups):
            for j in range(num_vars):
                sample_matrix[index, j] = base_samples[i, j + num_vars] if (not groups and j == k) else base_samples[i, j]
            index += 1

        if calc_second_order:
            for k in range(dim_groups):
                for j in range(num_vars):
                    sample_matrix[index, j] = base_samples[i, j] if (not groups and j == k) else base_samples[i, j + num_vars]
                index += 1

        sample_matrix[index] = base_samples[i, num_vars:]  # B
        index += 1

    return _rescale_parameters(sample_matrix, settings)


def _process_sample_chunk(sobol_sample_chunk, solver, t, phisc, phitsc, theta, thetac,
                          piecewise_func, tsc, sobol_problem, phi_nom, phit_nom, system,
                          var_sensitivity, par_sensitivity, models):
    """
    Execute model simulations for a chunk of Sobol samples (parallel worker function).

    This function is called by parallel workers to process a subset of the total
    Sobol sample matrix. It extracts parameters and variables from each sample,
    runs the forward model, and collects time-variant outputs.

    Parameters
    ----------
    sobol_sample_chunk : np.ndarray, shape (n_chunk, D)
        Subset of Sobol samples to process.
    solver : str
        Model/solver name.
    t : list[float]
        Normalized time points [0, 1].
    phisc : dict[str, float]
        Scaling factors for time-invariant inputs (all 1.0 for GSA).
    phitsc : dict[str, float]
        Scaling factors for time-variant inputs (all 1.0 for GSA).
    theta : list[float]
        Nominal parameter values.
    thetac : list[float]
        Parameter scaling factors (all 1.0 for GSA).
    piecewise_func : dict[str, str]
        CVP methods (all 'no_CVP' for GSA).
    tsc : int
        Time scaling factor.
    sobol_problem : dict
        Problem definition with 'names' and 'num_vars'.
    phi_nom : list[float]
        Nominal time-invariant input values.
    phit_nom : list[float]
        Nominal time-variant input values.
    system : dict
        System configuration.
    var_sensitivity : bool
        Whether variables are included in sensitivity analysis.
    par_sensitivity : bool
        Whether parameters are included in sensitivity analysis.
    models : dict
        Model definitions.

    Returns
    -------
    results : list[dict]
        List of simulation results (tv_ophi dicts) for each sample.

    Notes
    -----
    **Sample Interpretation**:
    Sobol samples contain either:
        - Parameters only (if par_sensitivity=True, var_sensitivity=False)
        - Variables only (if var_sensitivity=True, par_sensitivity=False)
        - Both parameters and variables (if both=True)

    The function splits each sample accordingly and assigns to theta, phi, phit.

    **Nominal Values**:
    When not varying a quantity (e.g., variables when par_sensitivity=True only),
    nominal values are used from gsa configuration.

    **Parallel Execution**:
    This function is designed to be called by ProcessPoolExecutor workers.
    Each worker processes an independent chunk without shared memory.

    See Also
    --------
    sensa : Splits samples into chunks and submits to workers.
    simula : Forward model called for each sample.
    _process_results : Aggregates results from all chunks.

    Examples
    --------
    >>> # Typically called internally by ProcessPoolExecutor
    >>> chunk = sobol_samples[0:100]  # First 100 samples
    >>> results = _process_sample_chunk(
    ...     chunk, 'M1', t, phisc, phitsc, theta, thetac,
    ...     cvp, tsc, problem, phi_nom, phit_nom, system,
    ...     True, True, models
    ... )
    >>> print(len(results))  # 100
    >>> print(results[0].keys())  # ['CA', 'CB', 'T']
    """
    results = []

    for params in sobol_sample_chunk:
        if par_sensitivity and not var_sensitivity:
            phi = {key: value for key, value in zip(system['tii'].keys(), phi_nom)}
            phit = {key: value for key, value in zip(system['tvi'].keys(), phit_nom)}
            params_split = params[:len(theta)]
        elif var_sensitivity and not par_sensitivity:
            params_split = theta
            var_split = params
            phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
            phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}
        else:
            params_split = params[:len(theta)]
            var_split = params[len(theta):]
            phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
            phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}

        phit = {key: float(value) for key, value in phit.items()}

        tv_ophi, ti_ophi, phit_interp = simula(
            t, {}, phisc, phi, phit, tsc, params_split, thetac,
            piecewise_func, phitsc, solver, system, models
        )

        results.append(tv_ophi)

    return results


def _process_results(solver, results, t, sobol_problem_solver, plt_show):
    """
    Aggregate simulation results and compute time-varying Sobol indices.

    After all Sobol samples have been simulated, this function organizes outputs
    by response variable, computes Sobol indices at each time point, and optionally
    generates plots.

    Parameters
    ----------
    solver : str
        Model/solver name for labeling.
    results : list[dict]
        List of simulation results. Each dict maps response names to time series.
    t : list[float]
        Time points (normalized [0, 1]).
    sobol_problem_solver : dict
        Sobol problem definition for this solver.
    plt_show : bool
        Whether to generate Sobol plots.

    Returns
    -------
    SA : dict[str, list[dict]]
        Sensitivity analysis results:
            - Keys: response variable names
            - Values: lists of index dicts, one per time point (excluding t=0)
            Each dict contains 'S1', 'ST', 'S1_conf', 'ST_conf' arrays.

    Notes
    -----
    **Time Point Exclusion**:
    The first time point (t=0) is excluded from analysis because:
        - Initial conditions are fixed (no parameter influence yet)
        - Zero variance leads to undefined Sobol indices
        - Transient behavior starts after t=0

    **Result Organization**:
    For each response variable:
        1. Extract all simulation outputs (N*(2D+2) trajectories)
        2. At each time point j, form output vector Y[j]
        3. Compute Sobol indices from Y[j]
        4. Store in time series

    **Plotting**:
    If plt_show=True, calls plot_sobol_results() for each response,
    creating time-evolution plots of sensitivity indices.

    See Also
    --------
    _sobol_analysis : Computes indices from output vector.
    plot_sobol_results : Visualization function.
    sensa : Main function that aggregates results from all solvers.

    Examples
    --------
    >>> # After running all simulations
    >>> SA = _process_results('M1', results, t, problem, True)
    >>>
    >>> # Access indices for response 'CA' at time point 50
    >>> indices_t50 = SA['CA'][50]
    >>> print(f"S1: {indices_t50['S1']}")
    >>> print(f"ST: {indices_t50['ST']}")
    >>>
    >>> # Plot shows how parameter importance changes over time
    """
    sobol_analysis_results = {}
    SA = {}
    sobol_analysis_results_excluded = {}

    for response_key in results[0].keys():
        response_values = np.array([result[response_key] for result in results])

        sobol_analysis_results[solver] = [
            _sobol_analysis(sobol_problem_solver, response_values[:, j])
            for j in range(len(t))
        ]

        t_excluded = t[1:]
        sobol_analysis_results_excluded[solver] = sobol_analysis_results[solver][1:]

        if plt_show:
            plot_sobol_results(
                t_excluded, sobol_analysis_results_excluded[solver],
                sobol_problem_solver, solver, response_key
            )

        SA[response_key] = sobol_analysis_results_excluded[solver]

    return SA