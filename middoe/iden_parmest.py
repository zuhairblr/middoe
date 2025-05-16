from scipy.optimize import minimize
from middoe.krnl_simula import simula
import numpy as np
from scipy.optimize import differential_evolution
from middoe.iden_utils import _initialize_dictionaries
import time
import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

def parmest(system, models, iden_opt, data, case=None):
    """
    Main function to perform parameter estimation.

    Parameters:
    system (dict): User provided - Dictionary containing the model structure.
    models (dict): User provided - Dictionary containing the modelling settings.
    iden_opt (dict): User provided - Dictionary containing the estimation settings.
    data (dict): Dictionary containing the data for optimization, from experiments.
    run_solver (function): Model simulator function.

    Returns:
    tuple: Results of the optimization and theta parameters.
    """

    # Unpack design settings
    _initialize_dictionaries(models, iden_opt)

    active_models = models['can_m']
    theta_parameters = models['theta']
    bound_max = models['t_u']
    bound_min = models['t_l']
    mutation = models['mutation']
    method = iden_opt['meth']
    objf = iden_opt['ob']


    if case is None:
        # Check if 'normalized_parameters' exists in models
        if 'normalized_parameters' in models:
            x0_dict = models['normalized_parameters']
        else:
            x0_dict = iden_opt['x0']
    elif case == 'freeze':
        x0_dict = iden_opt['x0']
    logging= iden_opt['log']

    # Call runner function
    results = _runner(
        active_models, theta_parameters, bound_max, bound_min, mutation,
        objf, x0_dict, method, data, system, models, logging
    )

    # Report results
    _report_optimization_results(results, logging)

    return results

def _objective(
    theta,
    data,
    active_models,
    x0,
    thetac,
    thetas,
    system,
    models,
    **kwargs
):
    """
    Objective function for optimization in parameter estimation.

    Parameters:
    ----------
    theta : list
        List of parameters to optimize.
    data : dict
        Dictionary containing the data for optimization.
    active_models : list
        List of active solvers.
    x0 : list
        Initial guess for the parameters.
    thetac : list
        List of parameter scalars.
    thetas : list
        List indicating which parameters are to be optimized.
    system : dict
        Dictionary that describes the structure of the model
        (variables, measured or not, uncertainties, etc.).
    run_solver : function
        Model simulator function.
    models : dict
        Settings for running the model.
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    ----------
    tuple
        (data, metrics)

        data : dict
            The original data (unmodified).
        metrics : dict
            A dictionary containing combined metrics:
            {
                'LS': float,       # classical R² over *all* scaled data
                'MLE': float,      # negative log-likelihood with known sigmas
                'Chi': float,      # Chi-squared
                'JWLS': float,     # Weighted least squares
                'R2_responses': {var1: R²_for_var1, var2: ..., ...}
            }
    """

    if len(theta) != len(x0):
        raise ValueError("Length of theta and x0 must be the same")

    # Convert to list (in case it's a numpy array)
    theta = theta.tolist()

    # Replace elements in theta with x0 where thetas is False
    theta = [theta[i] if thetas[i] else x0[i] for i in range(len(thetas))]

    # -------------------------------------------------------------------------
    # Prepare structures for storing global data (across sheets) that we'll use
    # to compute overall metrics at the end.
    # We'll keep both unscaled (for computing R²) and scaled or weighted sets.
    # -------------------------------------------------------------------------
    # For unweighted global R² across *all* variables (scaled by max-abs):
    global_y_true = {}
    global_y_pred = {}
    maxv = {}  # max absolute value per sheet & variable

    # For computing per-variable R², as well as JWLS and MLE,
    # we want to track each measurement along with its sigma.
    # var_measurements[var] = list of (y_true_i, y_pred_i, sigma_i)
    var_measurements = {}

    # -------------------------------------------------------------------------
    # Extract time-variant and time-invariant measured variables from system
    # -------------------------------------------------------------------------
    # By default, we consider only measured (get('measured', True) == True).
    tv_iphi_vars = list(system['tvi'].keys())
    ti_iphi_vars = list(system['tii'].keys())
    tv_ophi_vars = [
        var for var in system['tvo'].keys()
        if system['tvo'][var].get('meas', True)
    ]
    ti_ophi_vars = [
        var for var in system['tio'].keys()
        if system['tio'][var].get('meas', True)
    ]

    # -------------------------------------------------------------------------
    # Attempt to retrieve standard deviations for each measured variable.
    # If not found, default to 1.0.
    # We'll store them in a dict so that for each variable we know its sigma.
    # Note: The same sigma is used for all time steps for that variable (assuming
    # a single std. dev. per variable).
    # -------------------------------------------------------------------------
    std_dev = {}
    # TV outputs
    for var in system['tvo'].keys():
        if system['tvo'][var].get('meas', True):
            sigma = system['tvo'][var].get('unc', None)
            if sigma is None or np.isnan(sigma):
                sigma = 1.0
            std_dev[var] = sigma
    # TI outputs
    for var in system['tio'].keys():
        if system['tio'][var].get('meas', True):
            sigma = system['tio'][var].get('unc', None)
            if sigma is None or np.isnan(sigma):
                sigma = 1.0
            std_dev[var] = sigma

    # -------------------------------------------------------------------------
    # Initialize scaling factors for time-variant and time-invariant inputs
    # (used when calling the model).
    # -------------------------------------------------------------------------
    phisc = {var: 1 for var in ti_iphi_vars}   # Scaling for TI inputs
    phitsc = {var: 1 for var in tv_iphi_vars}  # Scaling for TV inputs
    tsc = 1  # Time-scaling factor

    # -------------------------------------------------------------------------
    # Main loop over each sheet of data
    # -------------------------------------------------------------------------
    for sheet_name, sheet_data in data.items():

        # Gather the time vectors for each measured TV output
        # (e.g., "MES_X:var" columns hold the times where that output was measured).
        t_values = {
            var: np.array(sheet_data[f"MES_X:{var}"])[~np.isnan(sheet_data[f"MES_X:{var}"])]
            for var in tv_ophi_vars
            if f"MES_X:{var}" in sheet_data
        }

        # swps_data will hold piecewise time + level data for piecewise functions
        # (time-variant inputs).
        swps_data = {}
        for var in tv_iphi_vars:
            if f'{var}t' not in sheet_data or f'{var}l' not in sheet_data:
                pass
            else:
                valid_t = np.array(sheet_data[f'{var}t'])[~np.isnan(sheet_data[f'{var}t'])]
                valid_l = np.array(sheet_data[f'{var}l'])[~np.isnan(sheet_data[f'{var}l'])]
                if valid_t.size > 0 and valid_l.size > 0:
                    swps_data[f'{var}t'] = valid_t
                    swps_data[f'{var}l'] = valid_l

        # Extract input variables (time-invariant and time-variant)
        ti_iphi_data = {
            var: np.array(sheet_data.get(var, [np.nan]))[0]
            for var in ti_iphi_vars
        }
        tv_iphi_data = {
            var: np.array(sheet_data.get(var, []))[~np.isnan(sheet_data.get(var, []))]
            for var in tv_iphi_vars
        }

        # Extract output variables that are measured (time-variant and time-invariant).
        tv_ophi_data = {
            var: np.array(sheet_data.get(f"MES_Y:{var}", []))[~np.isnan(sheet_data.get(f"MES_Y:{var}", []))]
            for var in tv_ophi_vars
            if f"MES_Y:{var}" in sheet_data
        }
        ti_ophi_data = {
            var: np.array(sheet_data.get(f"MES_Y:{var}", [np.nan]))[0]
            for var in ti_ophi_vars
            if f"MES_Y:{var}" in sheet_data
        }

        # Initialize maxv dict for this sheet
        maxv.setdefault(sheet_name, {})

        # Construct cvp for tv_iphi variables (piecewise control or similar).
        cvp = {}
        for var in system['tvi'].keys():
            cvp_col = f"CVP:{var}"
            if cvp_col not in sheet_data:
                raise ValueError(f"Missing '{cvp_col}' column in sheet: {sheet_name}")
            cvp[var] = sheet_data[cvp_col].iloc[0]

        # We'll use the first model in active_models, or you could loop over them.
        solver_name = active_models[0]

        if solver_name:
            if "X:all" not in sheet_data:
                raise ValueError(f"Missing 'X:all' column in sheet: {sheet_name}")

            # Read the time data (for the entire simulation) from "X:all"
            t_valuese = np.array(sheet_data["X:all"])[~np.isnan(sheet_data["X:all"])]
            t_valuese = np.unique(t_valuese)  # ensure unique & sorted

            # Run model
            tv_ophi_sim, ti_ophi_sim, phit_interp = simula(
                t_valuese,
                swps_data,
                ti_iphi_data,
                phisc,
                phitsc,
                tsc,
                theta,
                thetac,
                cvp,
                tv_iphi_data,
                solver_name,
                system,
                models
            )

            # Filter model outputs at the measurement times for each tv_ophi_vars
            tv_ophi_filtered = {}
            indices = {}
            for var in tv_ophi_vars:
                if var not in t_values:
                    raise ValueError(f"No time values found for variable '{var}' in t_values")
                t_values_var = t_values[var]

                # Find indices in the full simulation time that match the
                # measurement times for this variable.
                indices[var] = np.isin(t_valuese, t_values_var)

                if var in tv_ophi_sim:
                    tv_ophi_filtered[var] = np.array(tv_ophi_sim[var])[indices[var]]
                else:
                    raise KeyError(f"Variable '{var}' not found in tv_ophi_sim output")

            # -------------------------------------------------------------
            # Collect measured vs. predicted for TIME-VARIANT outputs
            # -------------------------------------------------------------
            for var in tv_ophi_vars:
                y_true = tv_ophi_data.get(var, np.array([]))
                y_pred = tv_ophi_filtered.get(var, np.zeros_like(y_true))

                # Match lengths
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

                # Avoid zero-division by taking max-abs for scaling
                max_val = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
                if max_val == 0:
                    max_val = 1.0

                # Store the max val
                maxv[sheet_name][var] = max_val

                # Scale data (for the global "scaled" approach to LS)
                y_true_scaled = y_true / max_val
                y_pred_scaled = y_pred / max_val

                # Collect scaled data for overall R² across *all* variables
                global_y_true.setdefault(var, []).extend(y_true_scaled)
                global_y_pred.setdefault(var, []).extend(y_pred_scaled)

                # Collect all points (unscaled) + sigma for weighted metrics
                # Use the standard deviation if available, else 1.0
                sigma_var = std_dev.get(var, 1.0)
                for yt, yp in zip(y_true, y_pred):
                    var_measurements.setdefault(var, []).append((yt, yp, sigma_var))

            # -------------------------------------------------------------
            # Collect measured vs. predicted for TIME-INVARIANT outputs
            # -------------------------------------------------------------
            for var in ti_ophi_vars:
                y_true = ti_ophi_data.get(var, np.nan)
                y_pred = ti_ophi_sim.get(var, np.nan)

                # If either is NaN, skip
                if not np.isnan(y_true) and not np.isnan(y_pred):
                    max_val = max(abs(y_true), abs(y_pred))
                    if max_val == 0:
                        max_val = 1.0

                    maxv[sheet_name][var] = max_val

                    y_true_scaled = y_true / max_val
                    y_pred_scaled = y_pred / max_val

                    global_y_true.setdefault(var, []).append(y_true_scaled)
                    global_y_pred.setdefault(var, []).append(y_pred_scaled)

                    sigma_var = std_dev.get(var, 0)
                    var_measurements.setdefault(var, []).append((y_true, y_pred, sigma_var))

    # -------------------------------------------------------------------------
    # Compute combined metrics
    # -------------------------------------------------------------------------

    # 1) Combine *scaled* data for a global "LS" metric (classical R²):
    all_y_true_scaled = []
    all_y_pred_scaled = []
    for var in global_y_true:
        y_true_list = np.array(global_y_true[var])
        y_pred_list = np.array(global_y_pred[var])
        min_len = min(len(y_true_list), len(y_pred_list))
        y_true_list = y_true_list[:min_len]
        y_pred_list = y_pred_list[:min_len]
        all_y_true_scaled.extend(y_true_list)
        all_y_pred_scaled.extend(y_pred_list)

    all_y_true_scaled = np.array(all_y_true_scaled)
    all_y_pred_scaled = np.array(all_y_pred_scaled)
    min_len = min(len(all_y_true_scaled), len(all_y_pred_scaled))
    all_y_true_scaled = all_y_true_scaled[:min_len]
    all_y_pred_scaled = all_y_pred_scaled[:min_len]

    # Classical R² on the scaled data:
    if len(all_y_true_scaled) > 1:
        ss_res = np.sum((all_y_true_scaled - all_y_pred_scaled) ** 2)
        mean_y_true = np.mean(all_y_true_scaled)
        ss_tot = np.sum((all_y_true_scaled - mean_y_true) ** 2)
        LS = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    else:
        # Not enough points to compute R²
        LS = np.nan

    # 2) Weighted LS (JWLS) and Weighted MLE across *all* unscaled data:
    #    We'll flatten everything: for each var, we had (y_true, y_pred, sigma).
    #    JWLS = sum( ((y_true - y_pred)/sigma)**2 )
    #    MLE  = sum( 0.5 * [ ln(2π) + ln(sigma^2) + ((y_true - y_pred)/sigma)^2 ] )
    #         = 0.5 * Σ [ ln(2π·sigma_i^2) + ((y_i - yhat_i)/sigma_i)^2 ]
    # Note: This is the negative log-likelihood for i.i.d normal with known sigmas.
    all_residuals_w = []
    sum_mle = 0.0
    sum_chi = 0.0

    # We'll also store R² for each var individually (unweighted, standard definition).
    r2_responses = {}

    for var, measurements in var_measurements.items():
        if len(measurements) == 0:
            continue

        # Separate out arrays
        y_true_arr = np.array([m[0] for m in measurements])
        y_pred_arr = np.array([m[1] for m in measurements])
        sigma_arr   = np.array([m[2] for m in measurements])

        # Weighted residual
        resid = (y_true_arr - y_pred_arr) / sigma_arr
        all_residuals_w.extend(resid)

        # Weighted MLE sum
        # MLE = 0.5 * Σ [ ln(2π sigma^2) + (resid)^2 ]
        # where resid = (y_true - y_pred)/sigma
        # => (resid)^2 = ((y_true - y_pred)/sigma)^2
        # => (y_true - y_pred)^2 / sigma^2
        # We also handle varying sigma_i
        for i in range(len(y_true_arr)):
            sigma_i = sigma_arr[i]
            diff_i = (y_true_arr[i] - y_pred_arr[i])
            sum_mle += 0.5 * (
                np.log(2 * np.pi * sigma_i**2) + (diff_i**2 / sigma_i**2)
            )

        # For Chi-squared
        #   Chi = Σ ( (y_true - y_pred)^2 / |y_pred| ), was unscaled
        #   but in many contexts, "chi-squared" can also be with known sigma
        #   => Σ ( (y_true - y_pred)^2 / sigma^2 ).
        # The user’s code used denominator = abs(y_pred). We'll keep that,
        # but you could adapt if you wanted sigma-based.
        # We'll continue with the user’s definition:
        denom = np.abs(y_pred_arr)
        denom[denom == 0] = 1e-8  # avoid div-zero
        chi_part = np.sum((y_true_arr - y_pred_arr) ** 2 / denom)
        sum_chi += chi_part

        # Compute unweighted R² for *this* var
        if len(y_true_arr) > 1:
            ss_res_var = np.sum((y_true_arr - y_pred_arr)**2)
            mean_var = np.mean(y_true_arr)
            ss_tot_var = np.sum((y_true_arr - mean_var)**2)
            r2_var = 1 - (ss_res_var / ss_tot_var) if ss_tot_var != 0 else 0.0
        else:
            r2_var = np.nan

        r2_responses[var] = r2_var

    # JWLS = Σ residual^2 where residual = (y_true - y_pred) / sigma
    all_residuals_w = np.array(all_residuals_w)
    WLS = np.sum(all_residuals_w**2)

    # Weighted MLE (already accumulated in sum_mle)
    MLE = sum_mle

    # 3) The original code's "Chi" we keep as well (sum of (res/something)).
    Chi = sum_chi

    # -------------------------------------------------------------------------
    # Store final metrics
    # -------------------------------------------------------------------------
    metrics = {
        'LS' : LS,        # classical R² on scaled data across all variables
        'MLE': MLE,       # negative log-likelihood with known sigmas
        'Chi': Chi,       # as originally defined
        'WLS': WLS,     # weighted LS with known sigmas
        'R2_responses': r2_responses  # dict of per-variable R²
    }

    return data, metrics


def _objective_function(theta, data, model, x0, thetac, objf, thetas, system, models, logging):
    """
    Wrapper function for the objective function to be used in optimization.

    Parameters:
    theta (list): List of parameters to optimize.
    data (dict): Dictionary containing the experimental data for optimization.
    model (str): Name of the model to be used.
    x0 (list): Initial guess for the parameters.
    thetac (list): List of parameter scalors.
    nd (int): Number of data points, for time spanning.
    objf (str): Name of the objective function to minimize.
    thetas (list): List indicating which parameters are to be optimized.
    design_variables (dict): Dictionary containing design variables, part of the model structure.
    run_solver (function): Model simulator function.

    Returns:
    float: Value of the objective function.
    """
    start_time = time.time()  # Start timer
    # Call the objective function
    optimized_data, metrics = _objective(
        theta, data, [model], x0, thetac, thetas, system, models
    )
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    if logging:
        print(f"Objective function: '{objf}'| model '{model}' | CPU time {elapsed_time:.4f} seconds.")


    # Extract the metrics needed for the optimization
    LS, MLE, Chi, WLS = metrics['LS'], metrics['MLE'], metrics['Chi'], metrics['WLS']

    # Determine which objective function to minimize
    if objf == 'LS':
        return -LS
    elif objf == 'MLE':
        return -MLE
    elif objf == 'Chi':
        return -Chi
    elif objf == 'WLS':
        return WLS
    else:
        raise ValueError(f"Unknown objective function: {objf}")


def _runner(active_models, theta_parameters, bound_max, bound_min, mutation, objf, x0_dict, method, data, system, models, logging):
    """
    Runner function to perform optimization using different solvers.

    Parameters:
    active_models (list): List of active models.
    theta_parameters (dict): Dictionary of theta parameters for each model.
    bound_max (dict): Dictionary of maximum bounds for parameters of each model.
    bound_min (dict): Dictionary of minimum bounds for parameters of each model.
    mutation (dict): Dictionary indicating which parameters are to be optimized for each model.
    nd2 (int): Number of data points, for time spanning.
    objf (str): Objective function to minimize.
    x0_dict (dict): Dictionary of initial guesses for each model.
    method (str): Optimization method ('Local' or 'Global').
    data (dict): Dictionary containing the data for optimization.
    design_variables (dict): Dictionary containing design variables.
    run_solver (function): Model simulator function.

    Returns:
    dict: Results of the optimization for each model.
    """
    results = {}
    x0 = {model: x0_dict[model] for model in active_models}

    for solver in active_models:
        thetac = theta_parameters[solver]
        thetamax = bound_max[solver]
        thetamin = bound_min[solver]
        thetas = mutation[solver]
        initial_x0 = x0[solver]
        narrow_factor = 1e-40

        bounds = [
            (x * (1 - narrow_factor), x * (1 + narrow_factor)) if not theta else (thetmin, thetmax)
            for x, theta, thetmax, thetmin in zip(initial_x0, thetas, thetamax, thetamin)
        ]

        if method == 'Ls':
            result = minimize(
                _objective_function,
                initial_x0,
                args=(data, solver, initial_x0, thetac, objf, thetas, system, models, logging),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 10000000, 'disp': False},
                tol=1e-4
            )

        if method == 'Ln':
            result = minimize(
                _objective_function,
                initial_x0,
                args=(data, solver, initial_x0, thetac, objf, thetas, system,  models, logging),
                method='Nelder-Mead',
                options={'maxiter': 100000, 'disp': False, 'fatol': 1e-6}
            )


        elif method == 'G':
            result = differential_evolution(
                _objective_function,
                bounds=bounds,
                args=(data, solver, initial_x0, thetac, objf, thetas, system, models, logging),
                maxiter=1000,
                popsize=18,
                tol=1e-6,
                strategy='best1bin',
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=True,
                updating='deferred',
                workers=-1,  # Enable parallel processing
                disp=False
            )

        # Store results for each model
        results[solver] = {
            'optimization_result': result,
        }
    return results


def _report_optimization_results(results, logging):
    """
    Report the optimization results.

    Parameters:
    results (dict): Dictionary containing the results of the optimization.
    logging (bool): Flag to indicate whether to log the results.
    """
    if logging:
        for solver, solver_results in results.items():
            print(f"parameter estimation for model {solver} concluded- success: {solver_results['optimization_result'].success}")
            print()



