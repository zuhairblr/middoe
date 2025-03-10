import numpy as np
import scipy.stats as stats
import logging
import matplotlib.pyplot as plt
import math


# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def Uncert(data, resultpr, model_structure, modelling_settings, estimation_settings, run_solver):
    """
    Perform uncertainty analysis on the optimization results.

    Parameters
    ----------
    data : dict
        Dictionary containing the experimental data.
    resultpr : dict
        Dictionary containing the results of the optimization from parameter estimation.
    model_structure : dict
        User-provided dictionary containing the model structure.
    modelling_settings : dict
        User-provided dictionary containing the modelling settings.
    estimation_settings : dict
        User-provided dictionary containing the estimation settings.
    run_solver : function
        Simulator function to run the model.

    Returns
    ----------
    tuple
        (resultun2, theta_parameters, solver_parameters, scaled_params, observed_values)

        Where resultun2 is a dictionary keyed by solver name, containing:
          'LS', 'MLE', 'MSE', 'Chi', 'LSA', 'JWLS', 'R2_responses', etc.
    """
    import numpy as np

    # Gather variable names
    tv_iphi_vars = list(model_structure.get('tv_iphi', {}).keys())
    ti_iphi_vars = list(model_structure.get('ti_iphi', {}).keys())
    tv_ophi_vars = [
        var for var in model_structure.get('tv_ophi', {}).keys()
        if model_structure['tv_ophi'][var].get('measured', True)
    ]
    ti_ophi_vars = [
        var for var in model_structure.get('ti_ophi', {}).keys()
        if model_structure['ti_ophi'][var].get('measured', True)
    ]

    # Attempt to retrieve standard deviations for each measured variable.
    # If not found or NaN, default to 1.0.
    # (Although we don't use them directly here, we show how you'd gather them.)
    std_dev = {}
    for var in model_structure.get('tv_ophi', {}):
        if model_structure['tv_ophi'][var].get('measured', True):
            sigma = model_structure['tv_ophi'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma
    for var in model_structure.get('ti_ophi', {}):
        if model_structure['ti_ophi'][var].get('measured', True):
            sigma = model_structure['ti_ophi'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma

    # Estimation settings
    eps = estimation_settings.get('eps', None)
    logging = estimation_settings.get('logging', False)

    # Modelling settings
    mutation = modelling_settings.get('mutation', {})
    theta_parameters = modelling_settings.get('theta_parameters', {})

    resultun = {}

    # We'll keep a placeholder to store the last-run 'observed_values'
    # so that we can return it at the end.
    observed_values = None

    # Loop over solvers in the results dictionary
    for solver, solver_results in resultpr.items():

        thetac = theta_parameters.get(solver, [])
        thetas = mutation.get(solver, [])
        initial_x0 = solver_results['optimization_result'].x

        # If eps is not provided, perform (optional) FDM mesh dependency test
        # in your own custom function:
        if eps is None:
            eps = _perform_fdm_mesh_dependency_test(
                initial_x0,
                thetac,
                solver,
                model_structure,
                modelling_settings,
                run_solver,
                tv_iphi_vars,
                ti_iphi_vars,
                tv_ophi_vars,
                ti_ophi_vars,
                data
            )

        # ---------------------------------------------------------------------
        # Call the core metrics function (updated to handle Weighted LS, MLE, etc.)
        # ---------------------------------------------------------------------
        (
            optimized_data,
            LS,
            MLE,
            Chi,
            LSA,
            V_matrix,
            CI,
            t_values,
            t_m,
            tv_input_m,
            ti_input_m,
            tv_output_m,
            ti_output_m,
            JWLS,
            obs,
            MSE,
            R2_responses
        ) = _uncert_metrics(
            initial_x0,
            data,
            [solver],
            initial_x0,
            thetac,
            eps,
            thetas,
            ti_iphi_vars,
            tv_iphi_vars,
            tv_ophi_vars,
            ti_ophi_vars,
            run_solver,
            model_structure,
            modelling_settings
        )
        observed_values = obs  # store how many measurements used, etc.

        resultun[solver] = {
            'optimization_result': solver_results['optimization_result'],
            'data': optimized_data,
            'LS': LS,                     # Overall R² across all scaled data
            'MLE': MLE,                   # Weighted negative log-likelihood
            'MSE': MSE,                   # Mean squared error (unweighted)
            'Chi': Chi,                   # Sum((res^2) / abs(pred))
            'LSA': LSA,                   # Jacobian (sensitivities)
            'JWLS': JWLS,                 # Weighted least squares
            'R2_responses': R2_responses, # dictionary of per-variable R²
            'V_matrix': np.array(V_matrix),
            'CI': CI,
            't_values': t_values,
            't_m': t_m,
            'tv_input_m': tv_input_m,
            'ti_input_m': ti_input_m,
            'tv_output_m': tv_output_m,
            'ti_output_m': ti_output_m,
            'estimations_normalized': initial_x0,
            'found_eps': eps
        }

    # Summarize results for all solvers
    (
        scaled_params,
        solver_parameters,
        solver_cov_matrices,
        solver_confidence_intervals,
        resultun2
    ) = _report(
        resultun,
        mutation,
        theta_parameters,
        modelling_settings,
        logging
    )

    return resultun2, theta_parameters, solver_parameters, scaled_params, observed_values

def _uncert_metrics(
    theta,
    data,
    active_solvers,
    x0,
    thetac,
    eps,
    thetas,
    ti_iphi_vars,
    tv_iphi_vars,
    tv_ophi_vars,
    ti_ophi_vars,
    run_solver,
    model_structure,
    modelling_settings
):
    """
    Calculate the uncertainty metrics for the optimization results.

    Parameters
    ----------
    theta : list
        Estimations (normalized).
    data : dict
        Dictionary containing the experimental data.
    active_solvers : list
        List of active solvers.
    x0 : list
        Estimations (normalized).
    thetac : list
        List of theta parameter scalars.
    eps : float
        Epsilon value for perturbations in sensitivity analysis.
    thetas : list
        List of boolean values indicating which parameters are active.
    ti_iphi_vars : list
        List of time-invariant input variables.
    tv_iphi_vars : list
        List of time-variant input variables.
    tv_ophi_vars : list
        List of time-variant output variables.
    ti_ophi_vars : list
        List of time-invariant output variables.
    run_solver : function
        Simulator function.
    model_structure : dict
        Dictionary describing the model structure.
    modelling_settings : dict
        Dictionary describing the modelling settings.

    Returns
    ----------
    tuple
        (data, LS, MLE, Chi, LSA, V_matrix, CI, t_values,
         t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m, JWLS, obs, MSE, R2_responses)
    """
    import numpy as np
    from scipy import stats

    if len(theta) != len(x0):
        raise ValueError("Length of theta and x0 must be the same")

    theta = theta.tolist()

    # Replace elements in theta with x0 where thetas is False
    theta = [
        theta[i] if thetas[i] else x0[i]
        for i in range(len(thetas))
    ]

    # -------------------------------------------------------------------------
    # Build a dictionary of standard deviations (sigmas) for each measured var.
    # Fallback to sigma=1.0 if none is provided or if not measured.
    # -------------------------------------------------------------------------
    std_dev = {}
    for var in model_structure.get('tv_ophi', {}):
        if model_structure['tv_ophi'][var].get('measured', True):
            sigma = model_structure['tv_ophi'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma

    for var in model_structure.get('ti_ophi', {}):
        if model_structure['ti_ophi'][var].get('measured', True):
            sigma = model_structure['ti_ophi'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma

    # -------------------------------------------------------------------------
    # We will store scaled data (for the overall "LS" R²) and also store all
    # (unscaled) data with sigmas for Weighted LS and Weighted MLE.
    # -------------------------------------------------------------------------
    global_y_true = {}
    global_y_pred = {}
    maxv = {}

    # For Weighted metrics and per-variable R²:
    var_measurements = {}  # var -> list of (y_true, y_pred, sigma)

    # Initialize the LSA array for sensitivity analysis
    param_mask = np.array(thetas)
    LSA = np.zeros((0, np.sum(param_mask)))  # LSA for active parameters

    # Initialize scaling factors
    phisc = {var: 1 for var in ti_iphi_vars}
    phitsc = {var: 1 for var in tv_iphi_vars}
    tsc = 1

    # Dictionaries to store time, input/output for each sheet
    t_m = {}
    tv_input_m = {}
    ti_input_m = {}
    tv_output_m = {}
    ti_output_m = {}

    # -------------------------------------------------------------------------
    # Main loop over each sheet in data
    # -------------------------------------------------------------------------
    for sheet_name, sheet_data in data.items():

        # Gather times for each measured time-variant output
        t_values = {
            var: np.array(sheet_data[f"MES_X:{var}"])[~np.isnan(sheet_data[f"MES_X:{var}"])]
            for var in tv_ophi_vars
            if f"MES_X:{var}" in sheet_data
        }

        # Build piecewise (sweep) data for time-variant inputs
        swps_data = {}
        for var in tv_iphi_vars:
            if f'{var}t' in sheet_data and f'{var}l' in sheet_data:
                valid_t = np.array(sheet_data[f'{var}t'])[~np.isnan(sheet_data[f'{var}t'])]
                valid_l = np.array(sheet_data[f'{var}l'])[~np.isnan(sheet_data[f'{var}l'])]
                if valid_t.size > 0 and valid_l.size > 0:
                    swps_data[f'{var}t'] = valid_t
                    swps_data[f'{var}l'] = valid_l

        # Extract TI and TV input data
        ti_iphi_data = {
            var: np.array(sheet_data.get(var, [np.nan]))[0]
            for var in ti_iphi_vars
        }
        tv_iphi_data_ = {
            var: np.array(sheet_data.get(var, []))[~np.isnan(sheet_data.get(var, []))]
            for var in tv_iphi_vars
        }

        # Extract measured TV/TI outputs
        tv_ophi_data_ = {
            var: np.array(sheet_data.get(f"MES_Y:{var}", []))[~np.isnan(sheet_data.get(f"MES_Y:{var}", []))]
            for var in tv_ophi_vars
            if f"MES_Y:{var}" in sheet_data
        }
        ti_ophi_data_ = {
            var: np.array(sheet_data.get(f"MES_Y:{var}", [np.nan]))[0]
            for var in ti_ophi_vars
            if f"MES_Y:{var}" in sheet_data
        }

        # Initialize maxv for this sheet
        maxv.setdefault(sheet_name, {})

        # Build the cvp dict (e.g., piecewise controls) for each tv_iphi variable
        cvp = {}
        for var in tv_iphi_vars:
            cvp_col = f"CVP:{var}"
            if cvp_col not in sheet_data:
                raise ValueError(f"Missing '{cvp_col}' column in sheet: {sheet_name}")
            cvp[var] = sheet_data[cvp_col].iloc[0]

        solver_name = active_solvers[0]
        if solver_name:

            # Get the overall time vector
            if "X:all" not in sheet_data:
                raise ValueError(f"Missing 'X:all' column in sheet: {sheet_name}")

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
                tv_iphi_data_,
                solver_name,
                model_structure,
                modelling_settings
            )

            # Filter solver outputs at the measurement times for TV outputs
            indices = {}
            tv_ophi_filtered = {}
            for var in tv_ophi_vars:
                if var not in t_values:
                    raise ValueError(
                        f"No time values found for variable '{var}' in t_values"
                    )
                t_values_var = t_values[var]
                indices[var] = np.isin(t_valuese, t_values_var)

                if var in tv_ophi_sim:
                    tv_ophi_filtered[var] = np.array(tv_ophi_sim[var])[indices[var]]
                else:
                    raise KeyError(f"Variable '{var}' not found in tv_ophi_sim")

            # Now we gather the "combined" y_true and y_pred (scaled) for building LSA
            y_model_combined = []
            y_true_combined = []

            # -------------------------------------------------------------
            # TIME-VARIANT outputs
            # -------------------------------------------------------------
            for var in tv_ophi_vars:
                y_true_arr = tv_ophi_data_.get(var, np.array([]))
                y_pred_arr = tv_ophi_filtered.get(var, np.zeros_like(y_true_arr))

                # Match lengths
                min_len_var = min(len(y_true_arr), len(y_pred_arr))
                y_true_arr = y_true_arr[:min_len_var]
                y_pred_arr = y_pred_arr[:min_len_var]

                # Avoid zero division for scaling
                max_val = max(
                    np.max(np.abs(y_true_arr)),
                    np.max(np.abs(y_pred_arr))
                )
                if max_val == 0:
                    max_val = 1.0

                maxv[sheet_name][var] = max_val

                # Scale (for overall "LS" R²)
                y_true_scaled = y_true_arr / max_val
                y_pred_scaled = y_pred_arr / max_val

                # Store scaled data for overall R²:
                global_y_true.setdefault(var, []).extend(y_true_scaled)
                global_y_pred.setdefault(var, []).extend(y_pred_scaled)

                # Append to the combined arrays
                y_true_combined.extend(y_true_scaled)
                y_model_combined.extend(y_pred_scaled)

                # Also store unscaled data + sigma for Weighted calculations
                sigma_var = std_dev.get(var, 1.0)
                for yt, yp in zip(y_true_arr, y_pred_arr):
                    var_measurements.setdefault(var, []).append((yt, yp, sigma_var))

            # -------------------------------------------------------------
            # TIME-INVARIANT outputs
            # -------------------------------------------------------------
            for var in ti_ophi_vars:
                y_true_val = ti_ophi_data_.get(var, np.nan)
                y_pred_val = ti_ophi_sim.get(var, np.nan)

                if not np.isnan(y_true_val) and not np.isnan(y_pred_val):
                    max_val = max(abs(y_true_val), abs(y_pred_val))
                    if max_val == 0:
                        max_val = 1.0

                    maxv[sheet_name][var] = max_val

                    y_true_scaled = y_true_val / max_val
                    y_pred_scaled = y_pred_val / max_val

                    global_y_true.setdefault(var, []).append(y_true_scaled)
                    global_y_pred.setdefault(var, []).append(y_pred_scaled)

                    y_true_combined.append(y_true_scaled)
                    y_model_combined.append(y_pred_scaled)

                    sigma_var = std_dev.get(var, 0)
                    var_measurements.setdefault(var, []).append(
                        (y_true_val, y_pred_val, sigma_var)
                    )

            # -------------------------------------------------------------
            # Build the local sensitivity matrix (LSAe) for this sheet
            # -------------------------------------------------------------
            LSAe = np.zeros((len(y_model_combined), np.sum(param_mask)))

            for i, param_idx in enumerate(np.where(param_mask)[0]):
                modified_theta = theta.copy()
                modified_theta[param_idx] += eps

                # Run solver with the perturbed parameter
                tv_ophi_mod, ti_ophi_mod, _ = run_solver(
                    t_valuese,
                    swps_data,
                    ti_iphi_data,
                    phisc,
                    phitsc,
                    tsc,
                    modified_theta,
                    thetac,
                    cvp,
                    tv_iphi_data_,
                    solver_name,
                    model_structure,
                    modelling_settings
                )

                # Filter them at measurement times
                tv_ophi_mod_filtered = {}
                for var in tv_ophi_vars:
                    tv_ophi_mod_filtered[var] = np.array(tv_ophi_mod[var])[indices[var]]

                # Build y_model_modified_combined
                y_model_modified_combined = []
                # TV
                for var in tv_ophi_vars:
                    y_pred_mod = tv_ophi_mod_filtered[var]
                    # match length
                    y_pred_mod = y_pred_mod[: min_len_var]  # be cautious with indexing
                    # scale by the same max_val
                    this_max_val = maxv[sheet_name][var]
                    y_model_modified_combined.extend(y_pred_mod / this_max_val)

                # TI
                for var in ti_ophi_vars:
                    y_pred_mod = ti_ophi_mod.get(var, np.nan)
                    if not np.isnan(y_pred_mod):
                        this_max_val = maxv[sheet_name][var]
                        y_model_modified_combined.append(y_pred_mod / this_max_val)

                # Sensitivity: (f(theta+eps) - f(theta)) / eps
                LSAe[:, i] = (
                    np.array(y_model_modified_combined) - np.array(y_model_combined)
                ) / eps

            # Append to global LSA
            LSA = np.vstack([LSA, LSAe])

            # Store time and input/output
            t_m[sheet_name] = t_valuese
            tv_input_m[sheet_name] = tv_iphi_data_
            ti_input_m[sheet_name] = ti_iphi_data
            tv_output_m[sheet_name] = tv_ophi_sim
            ti_output_m[sheet_name] = ti_ophi_sim

    # -------------------------------------------------------------------------
    # Now compute the final metrics
    # -------------------------------------------------------------------------
    # 1) Combine scaled data for an overall R² (LS)
    all_y_true_scaled = []
    all_y_pred_scaled = []
    for var in global_y_true:
        arr_true = np.array(global_y_true[var])
        arr_pred = np.array(global_y_pred[var])
        nmin = min(len(arr_true), len(arr_pred))
        arr_true = arr_true[:nmin]
        arr_pred = arr_pred[:nmin]
        all_y_true_scaled.extend(arr_true)
        all_y_pred_scaled.extend(arr_pred)

    all_y_true_scaled = np.array(all_y_true_scaled)
    all_y_pred_scaled = np.array(all_y_pred_scaled)
    nmin = min(len(all_y_true_scaled), len(all_y_pred_scaled))
    all_y_true_scaled = all_y_true_scaled[:nmin]
    all_y_pred_scaled = all_y_pred_scaled[:nmin]

    obs = len(all_y_true_scaled)  # number of total scaled measurements
    if obs < 1:
        raise ValueError("No valid measurements found; cannot compute metrics.")

    # Classical R² on scaled data
    ss_res_scaled = np.sum((all_y_true_scaled - all_y_pred_scaled) ** 2)
    mean_scaled = np.mean(all_y_true_scaled)
    ss_tot_scaled = np.sum((all_y_true_scaled - mean_scaled) ** 2)
    LS = 1 - (ss_res_scaled / ss_tot_scaled) if ss_tot_scaled != 0 else np.nan

    # -------------------------------------------------------------------------
    # Weighted metrics: var_measurements has the unscaled data + sigma
    # We'll compute:
    #   JWLS = Σ ( (y_i - yhat_i)/sigma_i )^2
    #   MLE = 0.5 * Σ [ ln(2πσ_i^2) + ((y_i - yhat_i)/σ_i)^2 ]
    #   R2_responses[var] = classical R2 (unweighted) for each variable
    # -------------------------------------------------------------------------

    sum_jwls = 0.0
    sum_mle = 0.0
    sum_chi = 0.0

    R2_responses = {}

    for var, meas_list in var_measurements.items():
        if len(meas_list) == 0:
            continue

        y_true_arr = np.array([m[0] for m in meas_list])
        y_pred_arr = np.array([m[1] for m in meas_list])
        sigma_arr  = np.array([m[2] for m in meas_list])

        # Weighted residual (y_true - y_pred)/sigma
        resid = (y_true_arr - y_pred_arr) / sigma_arr
        sum_jwls += np.sum(resid**2)

        # Weighted MLE
        for i in range(len(y_true_arr)):
            diff_i = y_true_arr[i] - y_pred_arr[i]
            sig_i = sigma_arr[i]
            # Negative log-likelihood for N(y_pred, sig^2)
            #  = 0.5 * [ ln(2π sig^2) + ((y_true - y_pred)/sig)^2 ]
            sum_mle += 0.5 * (
                math.log(2 * math.pi * (sig_i**2)) + (diff_i**2 / sig_i**2)
            )

        # "Chi" in your original code was sum of ( (y_true - y_pred)^2 / abs(y_pred) ).
        denom = np.abs(y_pred_arr)
        denom[denom == 0] = 1e-8
        chi_part = np.sum((y_true_arr - y_pred_arr) ** 2 / denom)
        sum_chi += chi_part

        # Per-variable unweighted R²
        if len(y_true_arr) > 1:
            ss_res_var = np.sum((y_true_arr - y_pred_arr) ** 2)
            mean_var = np.mean(y_true_arr)
            ss_tot_var = np.sum((y_true_arr - mean_var) ** 2)
            r2_var = 1 - (ss_res_var / ss_tot_var) if ss_tot_var != 0 else 0.0
        else:
            r2_var = np.nan

        R2_responses[var] = r2_var

    JWLS = sum_jwls
    MLE = sum_mle
    Chi = sum_chi

    # -------------------------------------------------------------------------
    # Return also the unweighted residual-based MSE as in original code
    # -------------------------------------------------------------------------
    # (all_y_true_scaled and all_y_pred_scaled are scaled, but let's check the original unscaled approach)
    # The code below in original snippet was using `residuals = all_y_true - all_y_pred`.
    # We'll replicate that logic for continuity. However, the "all_y_true" and "all_y_pred"
    # we had in the original snippet were scaled. Let's keep that so MSE is consistent
    # with original approach. If you want unscaled MSE, you'd have to combine unscaled data
    # from var_measurements.
    residuals_scaled = (all_y_true_scaled - all_y_pred_scaled)
    ss_res = np.sum(residuals_scaled**2)
    MSE = ss_res / obs

    # -------------------------------------------------------------------------
    # The rest of the code: building covariance matrix from LSA
    # -------------------------------------------------------------------------
    dof = obs - np.sum(param_mask)
    if dof <= 0:
        # Edge case: more parameters than data points, or no data
        dof = max(1, obs)  # to avoid invalid dof

    # Critical t-value for 95% CI
    trv = stats.t.ppf(1 - (1 - 0.95) / 2, dof)

    J_transpose = np.transpose(LSA)
    J_dot = np.dot(J_transpose, LSA).astype(np.float64)

    # Regularization if near-singular
    if np.linalg.cond(J_dot) > 1e10:
        print("Matrix is near singular, adding regularization.")
        J_dot += np.eye(J_dot.shape[0]) * 1e-10

    # In many approaches, we might use the Weighted version to compute sigma,
    # but we'll match your original approach: sigma from unweighted residual.
    sigma_ls = ss_res / dof
    try:
        J_inverse = np.linalg.inv(J_dot)
    except np.linalg.LinAlgError:
        # fallback
        J_inverse = np.linalg.pinv(J_dot)

    V_matrix = sigma_ls * J_inverse
    diagonal_values = np.diag(V_matrix)
    sqrt_diagonal_values = np.sqrt(diagonal_values)

    # Confidence Intervals (CI) for each active parameter
    CI = sqrt_diagonal_values * trv

    # t-values for the active parameters
    param_values = np.array(theta)[param_mask]
    t_values = param_values / (CI + 1e-12)  # avoid zero-division

    # -------------------------------------------------------------------------
    # Return everything, adding R2_responses at the end
    # -------------------------------------------------------------------------
    return (
        data,
        LS,
        MLE,
        Chi,
        LSA,
        V_matrix,
        CI,
        t_values,
        t_m,
        tv_input_m,
        ti_input_m,
        tv_output_m,
        ti_output_m,
        JWLS,
        obs,
        MSE,
        R2_responses
    )

def _report(result, mutation, theta_parameters, modelling_settings, logging):
    """
    Report the uncertainty and parameter estimation results.

    Parameters:
    ----------
    result : dict
        Dictionary containing the results of the optimization.
    mutation : dict
        Dictionary indicating which parameters are to be optimized for each solver.
    theta_parameters : dict
        Dictionary of theta parameters for each solver.
    modelling_settings : dict
        User-provided dictionary containing the modelling settings.
    logging : bool
        Flag to indicate whether to log the results.

    Returns:
    ----------
    tuple
        (scaled_params, solver_parameters, solver_cov_matrices,
         solver_confidence_intervals, result)
    """
    solver_parameters = {}
    solver_cov_matrices = {}
    solver_confidence_intervals = {}
    scaled_thetaest = {}
    scaled_params = {}
    sum_of_chi_squared = sum(1 / (res['Chi']) ** 2 for res in result.values())

    for solver, solver_results in result.items():
        # Extract mutation mask and positions of active parameters
        mutation_mask = mutation.get(solver, [True] * len(solver_results['optimization_result'].x))
        original_positions = [i for i, active in enumerate(mutation_mask) if active]

        # Scale estimated parameters
        scaled_thetaest[solver] = [
            param * scale for param, scale in zip(solver_results['optimization_result'].x, theta_parameters[solver])
        ]

        if logging:
            scaled_params[solver] = [float(param) for param in scaled_thetaest[solver]]
            print(f"Estimated parameters of {solver}: {scaled_params[solver]}")
            print(f"True parameters of {solver}: {theta_parameters[solver]}")
            print(f"LS objective function value for {solver}: {solver_results['LS']}")

        # Update modelling settings with V_matrix for the solver
        modelling_settings['V_matrix'][solver] = solver_results['V_matrix']

        # Map CI, t-values, and CI ratios to their original parameter positions
        CI_mapped = {pos: solver_results['CI'][i] for i, pos in enumerate(original_positions)}
        t_values_mapped = {pos: solver_results['t_values'][i] for i, pos in enumerate(original_positions)}
        CI_ratio_mapped = {
            pos: ci * 100 / param if param != 0 else float('inf')
            for pos, ci, param in zip(original_positions, solver_results['CI'], solver_results['optimization_result'].x)
        }

        # Log t-values if requested
        if logging:
            print(f"T-values of solver {solver}: {solver_results['t_values']}")

        # Store parameters, covariance matrices, and confidence intervals
        solver_parameters[solver] = solver_results['optimization_result'].x
        solver_cov_matrices[solver] = solver_results['V_matrix']
        solver_confidence_intervals[solver] = CI_mapped
        solver_results['P'] = ((1 / (solver_results['Chi']) ** 2) / sum_of_chi_squared) * 100

        # Log R² for responses
        if 'R2_responses' in solver_results:
            r2_responses = solver_results['R2_responses']
            if logging:
                print(f"R² values for responses in solver {solver}:")
                for var, r2_value in r2_responses.items():
                    print(f"  {var}: {r2_value:.4f}")

            # Optionally store R² values in result for external use
            solver_results['R2_responses_summary'] = {
                var: round(r2_value, 4) for var, r2_value in r2_responses.items()
            }

    return scaled_params, solver_parameters, solver_cov_matrices, solver_confidence_intervals, result

def _perform_fdm_mesh_dependency_test(theta, thetac, solver, model_structure, modelling_settings, run_solver, tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars, data):
    """
    Perform FDM mesh dependency test to determine a suitable epsilon for sensitivity analysis.

    Parameters:
    theta (list): Initial parameter estimates.
    thetac (list): Scaling factors for parameters.
    solver (str): Solver name.
    model_structure (dict): Model structure.
    modelling_settings (dict): Modelling settings.
    run_solver (function): Simulator function to run the model.
    tv_iphi_vars (list): List of time-variant input variables.
    ti_iphi_vars (list): List of time-invariant input variables.
    tv_ophi_vars (list): List of time-variant output variables.
    ti_ophi_vars (list): List of time-invariant output variables.
    data (dict): Experimental data.

    Returns:
    float: Optimal epsilon value for sensitivity analysis.
    """
    logger.info(f"Performing FDM mesh dependency test for solver {solver}.")
    eps_values = np.logspace(-8, -1, 20)
    determinant_changes = []

    for eps in eps_values:
        try:
            result = _uncert_metrics(
                theta, data, [solver], theta, thetac, eps, [True] * len(theta),
                ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars, run_solver, model_structure, modelling_settings
            )
            V_matrix = result[5]  # Adjust index based on the correct position of V_matrix in the returned tuple
            determinant_changes.append(np.linalg.det(V_matrix))
        except np.linalg.LinAlgError:
            determinant_changes.append(np.nan)

    determinant_changes = np.array(determinant_changes)

    # Plot determinant variations
    plt.figure(figsize=(8, 6))
    plt.plot(eps_values, determinant_changes, marker='o', label='Det(V_matrix)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Determinant of Variance-Covariance Matrix')
    plt.title(f'Epsilon Dependency Test for Solver {solver}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{solver}_eps_dependency_test.png")
    plt.show()

    # Find epsilon with the lowest variation compared to neighbors
    stable_region = np.where(np.isfinite(determinant_changes))[0]

    if stable_region.size == 0:
        raise ValueError(f"Could not determine a stable epsilon region for solver {solver}.")

    lowest_variation_index = None
    min_variation = np.inf

    for i in range(1, len(stable_region) - 1):
        prev_val = determinant_changes[stable_region[i - 1]]
        current_val = determinant_changes[stable_region[i]]
        next_val = determinant_changes[stable_region[i + 1]]

        variation = abs(current_val - prev_val) + abs(current_val - next_val)
        if variation < min_variation:
            min_variation = variation
            lowest_variation_index = stable_region[i]

    optimal_eps = eps_values[lowest_variation_index]
    logger.info(f"Optimal epsilon selected for solver {solver}: {optimal_eps}")

    return optimal_eps

