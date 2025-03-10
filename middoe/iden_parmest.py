from scipy.optimize import minimize
import numpy as np
from scipy.optimize import differential_evolution
from middoe.iden_utils import _initialize_dictionaries

def Parmest(model_structure, modelling_settings, estimation_settings, data, run_solver, case=None):
    """
    Main function to perform parameter estimation.

    Parameters:
    model_structure (dict): User provided - Dictionary containing the model structure.
    modelling_settings (dict): User provided - Dictionary containing the modelling settings.
    estimation_settings (dict): User provided - Dictionary containing the estimation settings.
    data (dict): Dictionary containing the data for optimization, from experiments.
    run_solver (function): Model simulator function.

    Returns:
    tuple: Results of the optimization and theta parameters.
    """

    # Unpack design settings
    _initialize_dictionaries(modelling_settings, estimation_settings)

    active_solvers = modelling_settings['active_solvers']
    theta_parameters = modelling_settings['theta_parameters']
    bound_max = modelling_settings['bound_max']
    bound_min = modelling_settings['bound_min']
    mutation = modelling_settings['mutation']
    method = estimation_settings['method']
    objf = estimation_settings['objf']


    if case is None:
        # Check if 'normalized_parameters' exists in modelling_settings
        if 'normalized_parameters' in modelling_settings:
            x0_dict = modelling_settings['normalized_parameters']
        else:
            x0_dict = estimation_settings['x0']
    elif case == 'freeze':
        x0_dict = estimation_settings['x0']
    logging= estimation_settings['logging']

    # Call runner function
    results = _runner(
        active_solvers, theta_parameters, bound_max, bound_min, mutation,
        objf, x0_dict, method, data, model_structure, run_solver, modelling_settings
    )

    # Report results
    _report_optimization_results(results, logging)

    return results

# def _objective(theta, data, active_solvers, x0, thetac, thetas, model_structure, run_solver, modelling_settings, **kwargs):
#     """
#     Objective function for optimization in parameter estimation.
#
#     Parameters:
#     theta (list): List of parameters to optimize.
#     data (dict): Dictionary containing the data for optimization.
#     active_solvers (list): List of active solvers.
#     x0 (list): Initial guess for the parameters.
#     thetac (list): List of parameter scalors.
#     nd (int): time discrete points.
#     thetas (list): List indicating which parameters are to be optimized.
#     design_variables (dict): Dictionary containing design variables , part of the model structure.
#     run_solver (function): Model simulator function.
#     **kwargs: Additional keyword arguments.
#
#     Returns:
#     tuple: Optimized data and metrics.
#     """
#     if len(theta) != len(x0):
#         raise ValueError("Length of theta and x0 must be the same")
#
#     theta = theta.tolist()
#
#     # Replace elements in theta with x0 where thetas is False
#     theta = [theta[i] if thetas[i] else x0[i] for i in range(len(thetas))]
#
#     global_y_true = {}
#     global_y_pred = {}
#     maxv = {}  # To store max values per sheet and variable
#
#     # Extract variables from design_variables
#     tv_iphi_vars = list(model_structure['tv_iphi'].keys())
#     ti_iphi_vars = list(model_structure['ti_iphi'].keys())
#     tv_ophi_vars = [var for var in model_structure['tv_ophi'].keys() if model_structure['tv_ophi'][var].get('measured', True)]
#     ti_ophi_vars = [var for var in model_structure['ti_ophi'].keys() if model_structure['ti_ophi'][var].get('measured', True)]
#
#     std_dev = {
#         var: model_structure['tv_ophi'][var]['unc']
#         for var in model_structure['tv_ophi'].keys()
#         if model_structure['tv_ophi'][var].get('measured', True)
#     }
#
#     # Initialize scaling factors
#     phisc = {var: 1 for var in ti_iphi_vars}  # Scaling factor for time-invariant inputs
#     phitsc = {var: 1 for var in tv_iphi_vars}  # Scaling factor for time-variant inputs
#     tsc = 1  # Time-scaling factor
#
#     # Loop through data
#     for sheet_name, sheet_data in data.items():
#         t_values = {
#             var: np.array(sheet_data[f"MES_X:{var}"])[~np.isnan(sheet_data[f"MES_X:{var}"])]
#             for var in tv_ophi_vars
#             if f"MES_X:{var}" in sheet_data
#         }
#         # Process time-variant and time-invariant input variables
#         swps_data = {}
#         for var in tv_iphi_vars:
#             if f'{var}t' not in sheet_data or f'{var}l' not in sheet_data:
#                 pass
#             else:
#                 # Filter out NaN values for time ('t') and level ('l') variables
#                 valid_t = np.array(sheet_data[f'{var}t'])[~np.isnan(sheet_data[f'{var}t'])]
#                 valid_l = np.array(sheet_data[f'{var}l'])[~np.isnan(sheet_data[f'{var}l'])]
#                 # Only add to swps_data if both valid_t and valid_l have valid (non-empty) data
#                 if valid_t.size > 0 and valid_l.size > 0:
#                     swps_data[f'{var}t'] = valid_t
#                     swps_data[f'{var}l'] = valid_l
#
#         # Extract input variables (time-invariant and time-variant)
#         ti_iphi_data = {var: np.array(sheet_data.get(var, [np.nan]))[0] for var in ti_iphi_vars}
#         tv_iphi_data = {var: np.array(sheet_data.get(var, []))[~np.isnan(sheet_data.get(var, []))] for var in
#                         tv_iphi_vars}
#
#         # Extract time-variant output variables for measured variables
#         tv_ophi_data = {
#             var: np.array(sheet_data.get(f"MES_Y:{var}", []))[~np.isnan(sheet_data.get(f"MES_Y:{var}", []))]
#             for var in tv_ophi_vars
#             if f"MES_Y:{var}" in sheet_data
#         }
#
#         # Extract time-invariant output variables for measured variables
#         ti_ophi_data = {
#             var: np.array(sheet_data.get(f"MES_Y:{var}", [np.nan]))[0]
#             for var in ti_ophi_vars
#             if f"MES_Y:{var}" in sheet_data
#         }
#
#         # Initialize maxv for this sheet
#         maxv.setdefault(sheet_name, {})
#
#         # Piecewise function check
#         # Construct cvp_doe dynamically for tv_iphi variables
#         cvp = {}
#
#         # Filter tv_iphi variables
#         tv_iphi_vars = model_structure['tv_iphi'].keys()
#
#         # Loop through tv_iphi variables
#         for var in tv_iphi_vars:
#             cvp_col = f"CVP:{var}"  # Expect column like "CVP:u1", "CVP:u2", etc.
#             if cvp_col not in sheet_data:
#                 raise ValueError(f"Missing '{cvp_col}' column in sheet: {sheet_name}")
#             # Assign the value from the sheet data
#             cvp[var] = sheet_data[cvp_col].iloc[0]
#
#         solver_name = active_solvers[0]
#
#         if solver_name:
#             # Ensure the `X:all` column exists in the data
#             if "X:all" not in sheet_data:
#                 raise ValueError(f"Missing 'X:all' column in sheet: {sheet_name}")
#
#             # Read the time data directly from the `X:all` column
#             t_valuese = np.array(sheet_data["X:all"])[~np.isnan(sheet_data["X:all"])]
#
#             # Ensure time values are unique and sorted
#             t_valuese = np.unique(t_valuese)
#
#             # Pass all required arguments to the solver
#             tv_ophi, ti_ophi, phit_interp = run_solver(
#                 t_valuese, swps_data, ti_iphi_data, phisc, phitsc, tsc, theta, thetac, cvp,
#                 tv_iphi_data, solver_name, model_structure, modelling_settings
#             )
#             tv_ophi_filtered = {}
#             indices={}
#             for var in tv_ophi_vars:
#                 if var not in t_values:
#                     raise ValueError(f"No time values found for variable '{var}' in t_values")
#
#                 # Get the specific time values for this variable
#                 t_values_var = t_values[var]
#
#                 # Find indices in the full simulation time (`t_valuese`) that match the specific times for this variable
#                 indices[var] = np.isin(t_valuese, t_values_var)
#
#                 # Filter the corresponding output values in `tv_ophi`
#                 if var in tv_ophi:
#                     tv_ophi_filtered[var] = np.array(tv_ophi[var])[indices[var]]
#                 else:
#                     raise KeyError(f"Variable '{var}' not found in tv_ophi")
#
#             # Collect true and predicted values, and apply scaling per sheet and variable
#             for var in tv_ophi_vars:
#                 # Get true values
#                 y_true = tv_ophi_data.get(var, np.array([]))
#                 # Get predicted values
#                 y_pred = tv_ophi_filtered.get(var, np.zeros_like(y_true))
#
#                 # Ensure matching lengths
#                 min_len = min(len(y_true), len(y_pred))
#                 y_true = y_true[:min_len]
#                 y_pred = y_pred[:min_len]
#
#                 # Find max value between true and predicted for this sheet and variable
#                 max_val = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
#                 if max_val == 0:
#                     max_val = 1  # Avoid division by zero
#
#                 # Store max_val
#                 maxv.setdefault(sheet_name, {})[var] = max_val
#
#                 # Scale the data
#                 y_true_scaled = y_true / max_val
#                 y_pred_scaled = y_pred / max_val
#
#                 # Store scaled data
#                 global_y_true.setdefault(var, []).extend(y_true_scaled)
#                 global_y_pred.setdefault(var, []).extend(y_pred_scaled)
#
#             # For time-invariant variables
#             for var in ti_ophi_vars:
#                 y_true = ti_ophi_data.get(var, np.nan)
#                 y_pred = ti_ophi.get(var, np.nan)
#
#                 if not np.isnan(y_true) and not np.isnan(y_pred):
#                     # Find max value between true and pred
#                     max_val = max(abs(y_true), abs(y_pred))
#                     if max_val == 0:
#                         max_val = 1  # Avoid division by zero
#
#                     # Store max_val
#                     maxv[sheet_name][var] = max_val
#
#                     # Scale the data
#                     y_true_scaled = y_true / max_val
#                     y_pred_scaled = y_pred / max_val
#
#                     # Store scaled data
#                     global_y_true.setdefault(var, []).append(y_true_scaled)
#                     global_y_pred.setdefault(var, []).append(y_pred_scaled)
#
#     # Combine all variables' data into a single array for metrics computation
#     all_y_true = []
#     all_y_pred = []
#
#     # Handle variables
#     for var in global_y_true:
#         y_true = np.array(global_y_true[var])
#         y_pred = np.array(global_y_pred.get(var, np.zeros_like(y_true)))
#
#         # Ensure matching lengths by trimming the longer array
#         min_len = min(len(y_true), len(y_pred))
#         y_true = y_true[:min_len]
#         y_pred = y_pred[:min_len]
#
#         # Append to the combined lists
#         all_y_true.extend(y_true)
#         all_y_pred.extend(y_pred)
#
#     # Convert to NumPy arrays
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
#
#     # Ensure both arrays have the same length
#     min_len = min(len(all_y_true), len(all_y_pred))
#     all_y_true = all_y_true[:min_len]
#     all_y_pred = all_y_pred[:min_len]
#
#     # LS
#     ss_res = np.sum((all_y_true - all_y_pred) ** 2)
#     mean_y_true = np.mean(all_y_true)
#     ss_tot = np.sum((all_y_true - mean_y_true) ** 2)
#     LS = 1 - (ss_res / ss_tot)
#     # MLE
#     residuals = all_y_true - all_y_pred
#     sigma_squared_mle = np.mean(residuals ** 2)
#     MLE = (len(all_y_true) / 2) * np.log(2 * np.pi * sigma_squared_mle) + \
#           (1 / (2 * sigma_squared_mle)) * np.sum(residuals ** 2)
#     # Chi
#     denominator = np.abs(all_y_pred)
#     denominator[denominator == 0] = 1e-8  # Avoid division by zero
#     Chi = np.sum((all_y_true - all_y_pred) ** 2 / denominator)
#     # JWLS
#     JWLS = np.dot(residuals.T, residuals)
#
#     # Store metrics for the combined dataset
#     metrics = {
#         'LS': LS,
#         'MLE': MLE,
#         'Chi': Chi,
#         'JWLS': JWLS
#     }
#
#     return data, metrics

def _objective(
    theta,
    data,
    active_solvers,
    x0,
    thetac,
    thetas,
    model_structure,
    run_solver,
    modelling_settings,
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
    active_solvers : list
        List of active solvers.
    x0 : list
        Initial guess for the parameters.
    thetac : list
        List of parameter scalars.
    thetas : list
        List indicating which parameters are to be optimized.
    model_structure : dict
        Dictionary that describes the structure of the model
        (variables, measured or not, uncertainties, etc.).
    run_solver : function
        Model simulator function.
    modelling_settings : dict
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
    # Extract time-variant and time-invariant measured variables from model_structure
    # -------------------------------------------------------------------------
    # By default, we consider only measured (get('measured', True) == True).
    tv_iphi_vars = list(model_structure['tv_iphi'].keys())
    ti_iphi_vars = list(model_structure['ti_iphi'].keys())
    tv_ophi_vars = [
        var for var in model_structure['tv_ophi'].keys()
        if model_structure['tv_ophi'][var].get('measured', True)
    ]
    ti_ophi_vars = [
        var for var in model_structure['ti_ophi'].keys()
        if model_structure['ti_ophi'][var].get('measured', True)
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
    for var in model_structure['tv_ophi'].keys():
        if model_structure['tv_ophi'][var].get('measured', True):
            sigma = model_structure['tv_ophi'][var].get('unc', None)
            if sigma is None or np.isnan(sigma):
                sigma = 1.0
            std_dev[var] = sigma
    # TI outputs
    for var in model_structure['ti_ophi'].keys():
        if model_structure['ti_ophi'][var].get('measured', True):
            sigma = model_structure['ti_ophi'][var].get('unc', None)
            if sigma is None or np.isnan(sigma):
                sigma = 1.0
            std_dev[var] = sigma

    # -------------------------------------------------------------------------
    # Initialize scaling factors for time-variant and time-invariant inputs
    # (used when calling the solver).
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
        for var in model_structure['tv_iphi'].keys():
            cvp_col = f"CVP:{var}"
            if cvp_col not in sheet_data:
                raise ValueError(f"Missing '{cvp_col}' column in sheet: {sheet_name}")
            cvp[var] = sheet_data[cvp_col].iloc[0]

        # We'll use the first solver in active_solvers, or you could loop over them.
        solver_name = active_solvers[0]

        if solver_name:
            if "X:all" not in sheet_data:
                raise ValueError(f"Missing 'X:all' column in sheet: {sheet_name}")

            # Read the time data (for the entire simulation) from "X:all"
            t_valuese = np.array(sheet_data["X:all"])[~np.isnan(sheet_data["X:all"])]
            t_valuese = np.unique(t_valuese)  # ensure unique & sorted

            # Run solver
            tv_ophi_sim, ti_ophi_sim, phit_interp = run_solver(
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
                model_structure,
                modelling_settings
            )

            # Filter solver outputs at the measurement times for each tv_ophi_vars
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
    JWLS = np.sum(all_residuals_w**2)

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
        'JWLS': JWLS,     # weighted LS with known sigmas
        'R2_responses': r2_responses  # dict of per-variable R²
    }

    return data, metrics


def _objective_function(theta, data, solver, x0, thetac, objf, thetas, model_structure, run_solver, modelling_settings):
    """
    Wrapper function for the objective function to be used in optimization.

    Parameters:
    theta (list): List of parameters to optimize.
    data (dict): Dictionary containing the experimental data for optimization.
    solver (str): Name of the model to be used.
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
    # Call the objective function
    optimized_data, metrics = _objective(
        theta, data, [solver], x0, thetac, thetas, model_structure, run_solver, modelling_settings
    )

    # Extract the metrics needed for the optimization
    LS, MLE, Chi, JWLS = metrics['LS'], metrics['MLE'], metrics['Chi'], metrics['JWLS']

    # Determine which objective function to minimize
    if objf == 'LS':
        return -LS
    elif objf == 'MLE':
        return -MLE
    elif objf == 'Chi':
        return -Chi
    elif objf == 'JWLS':
        return JWLS
    else:
        raise ValueError(f"Unknown objective function: {objf}")


def _runner(active_solvers, theta_parameters, bound_max, bound_min, mutation, objf, x0_dict, method, data, model_structure, run_solver, modelling_settings):
    """
    Runner function to perform optimization using different solvers.

    Parameters:
    active_solvers (list): List of active models.
    theta_parameters (dict): Dictionary of theta parameters for each solver.
    bound_max (dict): Dictionary of maximum bounds for parameters of each solver.
    bound_min (dict): Dictionary of minimum bounds for parameters of each solver.
    mutation (dict): Dictionary indicating which parameters are to be optimized for each solver.
    nd2 (int): Number of data points, for time spanning.
    objf (str): Objective function to minimize.
    x0_dict (dict): Dictionary of initial guesses for each solver.
    method (str): Optimization method ('Local' or 'Global').
    data (dict): Dictionary containing the data for optimization.
    design_variables (dict): Dictionary containing design variables.
    run_solver (function): Model simulator function.

    Returns:
    dict: Results of the optimization for each solver.
    """
    results = {}
    x0 = {solver: x0_dict[solver] for solver in active_solvers}

    for solver in active_solvers:
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

        if method == 'Local':
            result = minimize(
                _objective_function,
                initial_x0,
                args=(data, solver, initial_x0, thetac, objf, thetas, model_structure, run_solver, modelling_settings),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100000, 'disp': False},
                tol=1e-6
            )

        # if method == 'Local':
        #     result = minimize(
        #         _objective_function,
        #         initial_x0,
        #         args=(data, solver, initial_x0, thetac, objf, thetas, model_structure, run_solver, modelling_settings),
        #         method='Nelder-Mead',
        #         options={'maxiter': 100000, 'disp': False, 'fatol': 1e-6}
        #     )


        elif method == 'Global':
            result = differential_evolution(
                _objective_function,
                bounds=bounds,
                args=(data, solver, initial_x0, thetac, objf, thetas, model_structure, run_solver, modelling_settings),
                maxiter=10000,
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

        # Store results for each solver
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
            print(f"Optimization results for {solver}:")
            # print("Optimal Parameters:", solver_results['optimization_result'].x)
            print("Optimization Success:", solver_results['optimization_result'].success)
            print()



