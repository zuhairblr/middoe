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
    Perform Sobol sensitivity analysis for the given settings and model structure.

    Parameters:
    gsa (dict): Dictionary containing the settings for the global sensitivity analysis.
    models (dict): Dictionary containing the settings for the modelling process.
    system (dict): Dictionary containing the structure of the model.
    framework_settings (dict): Dictionary containing the settings for the framework.

    Returns:
    tuple: A tuple containing the Sobol analysis results and the Sobol problem definition.
    """
    phi_nom = gsa['tii_n']  # Nominal values for time-invariant variables
    phit_nom = gsa['tvi_n']  # Nominal values for time-variant variables
    sampling = gsa['samp']
    var_sensitivity = gsa['var_s']
    par_sensitivity = gsa['par_s']
    plt_show = gsa['plt']
    active_solvers = models['can_m']
    theta_parameters = models['theta']
    thetamaxd= models['t_u']
    thetamind= models['t_l']
    cvp_initial = {
        var: 'no_CVP'
        for var in system['tvi'].keys()
    }
    # piecewise_func = system['tv_iphi']['initial_piecewise_func'][0]
    parallel_value = gsa.get('multi', False)

    # Retrieve time information from model structure and ensure it's an integer
    time_field = system['t_s'][1]
    t = list(np.linspace(0, 1, 305))  # Ensure `t` is a list of floats
    # t = {key: list(np.linspace(0, 1, 305)) for key in system['tv_iphi'].keys()}
    tsc = int(time_field)  # Ensure `tsc` is an integer
    sobol_problem = {}
    sa = {}

    if isinstance(parallel_value, float):
        # MULTI-CORE EXECUTION
        power = parallel_value  # e.g., 0.7 means use 70% of CPU cores
        total_cpus = os.cpu_count()
        num_workers = int(np.floor(total_cpus * power))  # Use 70% of available CPUs

        # Use ProcessPoolExecutor for multiprocessing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for solver in active_solvers:
                print(f'Running GSA-Sobol for model: {solver} in parallel')
                theta = [float(val) for val in theta_parameters[solver]]
                thetamax = [float(max_val) * float(theta_val) for max_val, theta_val in zip(thetamaxd[solver], theta)]
                thetamin = [float(min_val) * float(theta_val) for min_val, theta_val in zip(thetamind[solver], theta)]

                # Construct scaling factors
                phisc = {key: 1.0 for key in system['tii'].keys()}  # Time-invariant input scaling factors
                phitsc = {key: 1.0 for key in system['tvi'].keys()}  # Time-variant input scaling factors
                thetac = [1.0 for _ in theta]  # Parameter scaling factors

                # Prepare variable bounds
                if isinstance(gsa.get('var_d'), float):
                    phi_bounds = [
                        (gsa['tii_n'][i] / gsa['var_d'],
                         gsa['tvi_n'][i] * gsa['var_d'])
                        for i in range(len(gsa['tii_n']))
                    ]
                    phit_bounds = [
                        (gsa['tii_n'][i] / gsa['var_d'],
                         gsa['tvi_n'][i] * gsa['var_d'])
                        for i in range(len(gsa['tvi_n']))
                    ]
                else:
                    phi_bounds = [
                        (var_attrs['min'], var_attrs['max']) for var_attrs in system['tii'].values()
                    ]
                    phit_bounds = [
                        (var_attrs['min'], var_attrs['max']) for var_attrs in system['tvi'].values()
                    ]

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
                    # sobol_problem[model]['names'].extend(system['ti_iphi']['var'] + system['tv_iphi']['var'])
                    sobol_problem[solver]['names'].extend(
                        list(system['tii'].keys()) + list(system['tvi'].keys()))

                    sobol_problem[solver]['bounds'].extend(phi_bounds)
                    sobol_problem[solver]['bounds'].extend(phit_bounds)

                sobol_samples = sample(sobol_problem[solver], N=sampling)

                # Split sobol_samples into chunks for parallel processing
                chunk_size = len(sobol_samples) // num_workers
                sobol_sample_chunks = [sobol_samples[i:i + chunk_size] for i in range(0, len(sobol_samples), chunk_size)]

                # Submit tasks for each chunk of sobol_samples for parallel processing
                futures = [executor.submit(_process_sample_chunk, chunk, solver, t, phisc, phitsc, theta, thetac, cvp_initial, tsc, sobol_problem[solver], phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models) for chunk in sobol_sample_chunks]

                # Collect results from all chunks
                results = []
                for future in futures:
                    results.extend(future.result())

                sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)

    else:
        # SINGLE-CORE EXECUTION
        for solver in active_solvers:
            print(f'Running GSA-Sobol for model: {solver} in single-core')
            theta = [float(val) for val in theta_parameters[solver]]
            thetamax = [float(max_val) * float(theta_val) for max_val, theta_val in zip(thetamaxd[solver], theta)]
            thetamin = [float(min_val) * float(theta_val) for min_val, theta_val in zip(thetamind[solver], theta)]

            phisc = {key: 1.0 for key in system['tii'].keys()}  # Time-invariant input scaling factors
            phitsc = {key: 1.0 for key in system['tvi'].keys()}  # Time-variant input scaling factors
            thetac = [1.0 for _ in theta]  # Parameter scaling factors

            if isinstance(gsa.get('var_d'), float):
                phi_bounds = [
                    (phi_nom[i] / gsa['var_d'], phi_nom[i] * gsa['var_d'])
                    for i, _ in enumerate(system['tii'].keys())
                ]
                phit_bounds = [
                    (phit_nom[i] / gsa['var_d'], phit_nom[i] * gsa['var_d'])
                    for i, _ in enumerate(system['tvi'].keys())
                ]
            else:
                phi_bounds = [
                    (attrs['min'], attrs['max']) for attrs in system['tii'].values()
                ]
                phit_bounds = [
                    (attrs['min'], attrs['max']) for attrs in system['tvi'].values()
                ]

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
                # sobol_problem[model]['names'].extend(system['ti_iphi']['var'] + system['tv_iphi']['var'])
                sobol_problem[solver]['names'].extend(
                    list(system['tii'].keys()) + list(system['tvi'].keys()))

                sobol_problem[solver]['bounds'].extend(phi_bounds)
                sobol_problem[solver]['bounds'].extend(phit_bounds)

            sobol_samples = sample(sobol_problem[solver], N=sampling)

            # Process samples in single-core mode (no parallel processing)
            results = _process_sample_chunk(sobol_samples, solver, t, phisc, phitsc, theta, thetac, cvp_initial, tsc, sobol_problem[solver], phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models)

            sa[solver] = _process_results(solver, results, t, sobol_problem[solver], plt_show)

    # Construct return dictionary
    sobol_output = {
        'analysis': sa,
        'problem': sobol_problem
    }

    return sobol_output


def _sobol_analysis(
        settings, model_outputs, calc_2nd_order=True, resamples=100, confidence=0.95,
        use_parallel=False, n_cores=None, keep_resamples=False, rng_seed=None):
    """
    Perform Sobol sensitivity analysis on the given model outputs.

    Parameters:
    settings (dict): Dictionary containing the settings for the analysis.
    model_outputs (numpy.ndarray): Array of model outputs to analyze.
    calc_2nd_order (bool, optional): Whether to calculate second-order indices. Default is True.
    resamples (int, optional): Number of resamples for confidence interval estimation. Default is 100.
    confidence (float, optional): Confidence level for the intervals. Default is 0.95.
    use_parallel (bool, optional): Whether to use parallel processing. Default is False.
    n_cores (int, optional): Number of cores to use for parallel processing. Default is None.
    keep_resamples (bool, optional): Whether to keep resamples for confidence interval calculation. Default is False.
    rng_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    dict: Dictionary containing the Sobol sensitivity indices and confidence intervals.
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
        raise RuntimeError("Sample mismatch between the model outputs and settings")

    model_outputs = (model_outputs - model_outputs.mean()) / model_outputs.std()

    A, B, AB, BA = _split_output_values(model_outputs, num_params, num_samples, calc_2nd_order)

    resample_indices = rng(num_samples, size=(num_samples, resamples))
    z_score = norm.ppf(0.5 + confidence / 2)

    results = _initialize_sensitivity_storage(num_params, resamples, keep_resamples, calc_2nd_order)

    for j in range(num_params):
        # First-order Sobol index
        results["S1"][j] = _compute_first_order_sensitivity(A, AB[:, j], B)

        # Total-order Sobol index
        results["ST"][j] = _compute_total_order_sensitivity(A, AB[:, j], B)

        if keep_resamples:
            # Only calculate confidence intervals if keep_resamples is True
            results["S1_conf_all"][:, j] = _compute_first_order_sensitivity(
                A[resample_indices], AB[resample_indices, j], B[resample_indices])
            results["ST_conf_all"][:, j] = _compute_total_order_sensitivity(
                A[resample_indices], AB[resample_indices, j], B[resample_indices])

            # Confidence intervals
            results["S1_conf"][j] = z_score * np.std(results["S1_conf_all"][:, j], ddof=1)
            results["ST_conf"][j] = z_score * np.std(results["ST_conf_all"][:, j], ddof=1)
        else:
            # Set confidence intervals to zero or NaN if keep_resamples is False
            results["S1_conf"][j] = np.nan
            results["ST_conf"][j] = np.nan

    return results

def _split_output_values(output_values, num_params, num_samples, calc_2nd_order):
    """
    Split the output values into matrices A, B, AB, and BA for Sobol sensitivity analysis.

    Parameters:
    output_values (numpy.ndarray): Array of model outputs to be split.
    num_params (int): Number of parameters in the model.
    num_samples (int): Number of samples in the output values.
    calc_2nd_order (bool): Whether to calculate second-order indices.

    Returns:
    tuple: A tuple containing matrices A, B, AB, and BA.
           - A (numpy.ndarray): Matrix of output values for the first set of samples.
           - B (numpy.ndarray): Matrix of output values for the second set of samples.
           - AB (numpy.ndarray): Matrix of output values for the first-order indices.
           - BA (numpy.ndarray or None): Matrix of output values for the second-order indices (if calc_2nd_order is True).
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
    Initialize storage for sensitivity analysis results.

    Parameters:
    num_params (int): Number of parameters in the model.
    resample_count (int): Number of resamples for confidence interval estimation.
    keep_resamples (bool): Whether to keep resamples for confidence interval calculation.
    calc_2nd_order (bool): Whether to calculate second-order indices.

    Returns:
    dict: Dictionary containing initialized storage for sensitivity indices and confidence intervals.
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
    Compute the first-order Sobol sensitivity index.

    Parameters:
    A (numpy.ndarray): Matrix of output values for the first set of samples.
    AB (numpy.ndarray): Matrix of output values for the first-order indices.
    B (numpy.ndarray): Matrix of output values for the second set of samples.

    Returns:
    numpy.ndarray: First-order Sobol sensitivity index.
    """
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)

def _compute_total_order_sensitivity(A, AB, B):
    """
    Compute the total-order Sobol sensitivity index.

    Parameters:
    A (numpy.ndarray): Matrix of output values for the first set of samples.
    AB (numpy.ndarray): Matrix of output values for the first-order indices.
    B (numpy.ndarray): Matrix of output values for the second set of samples.

    Returns:
    numpy.ndarray: Total-order Sobol sensitivity index.
    """
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)

def _validate_group_membership(settings):
    """
    Validate the group membership definitions in the settings.

    Parameters:
    settings (dict): Dictionary containing the settings for the analysis.

    Returns:
    bool: True if the group definitions are valid, False otherwise.
    """
    group_definitions = settings.get("groups")
    if not group_definitions or group_definitions == settings["names"]:
        return False
    return len(set(group_definitions)) > 1


def _rescale_parameters(sample_set, problem_def):
    """
    Rescale the sample set based on the problem definition.

    Parameters:
    sample_set (numpy.ndarray): The set of samples to be rescaled.
    problem_def (dict): The problem definition containing bounds and optional distributions.

    Returns:
    numpy.ndarray: The rescaled sample set.
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
    Rescale the sample set based on the provided bounds.

    Parameters:
    sample_set (numpy.ndarray): The set of samples to be rescaled.
    bounds (list of tuple): List of tuples containing the lower and upper bounds for each parameter.

    Raises:
    ValueError: If any lower bound is greater than or equal to the corresponding upper bound.
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
    Rescale the sample set based on the provided bounds and distributions.

    Parameters:
    sample_set (numpy.ndarray): The set of samples to be rescaled.
    bounds (list of tuple): List of tuples containing the lower and upper bounds for each parameter.
    distributions (list of str): List of distribution types for each parameter.

    Returns:
    numpy.ndarray: The rescaled sample set.
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
    Extract unique group names from the settings.

    Parameters:
    settings (dict): Dictionary containing the settings for the analysis.

    Returns:
    tuple: A tuple containing the list of unique group names and their count.
    """
    group_members = settings.get("groups", settings["names"])
    unique_group_set = list(pd.unique(np.asarray(group_members)))

    return unique_group_set, len(unique_group_set)


def _generate_group_matrix(groups):
    """
    Generate a membership matrix for the given groups.

    Parameters:
    groups (list): List of group names for each parameter.

    Returns:
    tuple: A tuple containing the membership matrix and the list of unique group names.
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
    Generate a sample matrix for Sobol sensitivity analysis.

    Parameters:
    settings (dict): Dictionary containing the settings for the analysis.
    N (int): Number of samples to generate.
    calc_second_order (bool, optional): Whether to calculate second-order indices. Default is True.
    scramble (bool, optional): Whether to scramble the Sobol sequence. Default is True.
    skip_values (int, optional): Number of values to skip in the Sobol sequence. Default is 0.
    seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    numpy.ndarray: The generated sample matrix.
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
        sample_matrix[index] = base_samples[i, :num_vars]  # Copy A matrix
        index += 1

        for k in range(dim_groups):
            for j in range(num_vars):
                sample_matrix[index, j] = base_samples[i, j + num_vars] if (not groups and j == k) else base_samples[
                    i, j]
            index += 1

        if calc_second_order:
            for k in range(dim_groups):
                for j in range(num_vars):
                    sample_matrix[index, j] = base_samples[i, j] if (not groups and j == k) else base_samples[
                        i, j + num_vars]
                index += 1

        sample_matrix[index] = base_samples[i, num_vars:]  # Copy B matrix
        index += 1

    return _rescale_parameters(sample_matrix, settings)


def _process_sample_chunk(sobol_sample_chunk, solver, t, phisc, phitsc, theta, thetac, piecewise_func, tsc, sobol_problem, phi_nom, phit_nom, system, var_sensitivity, par_sensitivity, models):
    """
    Process a chunk of Sobol samples for sensitivity analysis.

    Parameters:
    sobol_sample_chunk (numpy.ndarray): Chunk of Sobol samples to process.
    model (function): Solver function to use for simulation.
    simulate (function): Simulation function to use.
    t (dictionary): Dictionary of time points for the simulation for each variable.
    phisc (dict): Dictionary of scaling factors for time-invariant variables.
    phitsc (dict): Dictionary of scaling factors for time-variant variables.
    theta (list): List of parameter values.
    thetac (list): List of scaling factors for parameters.
    piecewise_func (dictionary): Piecewise function to use in the simulation for each time-variant variable.
    tsc (int): Time scaling factor.
    sobol_problem (dict): Dictionary containing the Sobol problem definition.
    phi_nom (list): List of nominal values for time-invariant variables.
    phit_nom (list): List of nominal values for time-variant variables.
    system (dict): Dictionary containing the model structure.
    var_sensitivity (bool): Whether to perform variable sensitivity analysis.
    par_sensitivity (bool): Whether to perform parameter sensitivity analysis.

    Returns:
    list: List of simulation results for each set of parameters in the chunk.
    """
    results = []

    for params in sobol_sample_chunk:
        if par_sensitivity and not var_sensitivity:
            # For parameter sensitivity only
            phi = {key: value for key, value in zip(system['tii'].keys(), phi_nom)}
            phit = {key: value for key, value in zip(system['tvi'].keys(), phit_nom)}
            params_split = params[:len(theta)]
        elif var_sensitivity and not par_sensitivity:
            # For variable sensitivity only
            params_split = theta  # Use nominal theta
            var_split = params
            phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
            phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}
        else:
            # For both parameter and variable sensitivity
            params_split = params[:len(theta)]
            var_split = params[len(theta):]
            phi = {key: value for key, value in zip(system['tii'].keys(), var_split[:len(phi_nom)])}
            phit = {key: value for key, value in zip(system['tvi'].keys(), var_split[len(phi_nom):])}

        # Ensure tvi values are floats
        phit = {key: float(value) for key, value in phit.items()}

        tv_ophi, ti_ophi, phit_interp = simula(t, {}, phisc, phi, phit, tsc, params_split, thetac, piecewise_func, phitsc, solver, system, models)

        results.append(tv_ophi)

    return results


def _process_results(solver, results, t, sobol_problem_solver, plt_show):
    """
    Process the results of Sobol sensitivity analysis.

    Parameters:
    model (str): The model used for the analysis.
    results (list): List of simulation results for each set of parameters.
    t (list): List of time points for the simulation.
    sobol_problem_solver (dict): Dictionary containing the Sobol problem definition for the model.
    framework_settings (dict): Dictionary containing the settings for the framework.

    Returns:
    tuple: A tuple containing the Sobol analysis results and the Sobol analysis results excluding the first time point.
    """
    sobol_analysis_results = {}
    SA = {}
    sobol_analysis_results_excluded = {}

    for response_key in results[0].keys():
        response_values = np.array([result[response_key] for result in results])

        sobol_analysis_results[solver] = [_sobol_analysis(sobol_problem_solver, response_values[:, j]) for j in range(len(t))]
        t_excluded = t[1:]
        sobol_analysis_results_excluded[solver] = sobol_analysis_results[solver][1:]
        if plt_show:
            plot_sobol_results(t_excluded, sobol_analysis_results_excluded[solver], sobol_problem_solver, solver, response_key)
        SA[response_key] = sobol_analysis_results_excluded[solver]
    return SA
