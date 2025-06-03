import numpy as np
from middoe.krnl_simula import simula
import scipy.stats as stats
import logging
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path


# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def uncert(data, resultpr, system, models, iden_opt):
    """
    Perform uncertainty analysis on the optimization results.

    Parameters
    ----------
    data : dict
        Dictionary containing the experimental data.
    resultpr : dict
        Dictionary containing the results of the optimization from parameter estimation.
    system : dict
        User-provided dictionary containing the model structure.
    models : dict
        User-provided dictionary containing the modelling settings.
    iden_opt : dict
        User-provided dictionary containing the estimation settings.
    run_solver : function
        Simulator function to run the model.

    Returns
    ----------
    tuple
        (resultun2, theta_parameters, solver_parameters, scaled_params, observed_values)

        Where resultun2 is a dictionary keyed by model name, containing:
          'LS', 'MLE', 'MSE', 'Chi', 'LSA', 'JWLS', 'R2_responses', etc.
    """


    # Gather variable names
    tv_iphi_vars = list(system.get('tvi', {}).keys())
    ti_iphi_vars = list(system.get('tii', {}).keys())
    tv_ophi_vars = [
        var for var in system.get('tvo', {}).keys()
        if system['tvo'][var].get('meas', True)
    ]
    ti_ophi_vars = [
        var for var in system.get('tio', {}).keys()
        if system['tio'][var].get('meas', True)
    ]

    # Attempt to retrieve standard deviations for each measured variable.
    # If not found or NaN, default to 1.0.
    # (Although we don't use them directly here, we show how you'd gather them.)
    std_dev = {}
    for var in system.get('tvo', {}):
        if system['tvo'][var].get('meas', True):
            sigma = system['tvo'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma
    for var in system.get('tio', {}):
        if system['tio'][var].get('meas', True):
            sigma = system['tio'][var].get('unc', None)
            if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
                sigma = 1.0
            std_dev[var] = sigma

    # Estimation settings
    eps = iden_opt.get('eps', None)
    if isinstance(eps, dict):
        epsf = eps.copy()  # user provided dict
    else:
        epsf = {solver: eps for solver in resultpr}  # use same float for all
    logging = iden_opt.get('log', False)

    # Modelling settings
    mutation = models.get('mutation', {})
    theta_parameters = models.get('theta', {})

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
        if epsf[solver] is None:
            epsf[solver] = _fdm_mesh_independency(initial_x0, thetac, solver, system, models, tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars, data)


        # ---------------------------------------------------------------------
        # Call the core metrics function (updated to handle Weighted LS, MLE, etc.)
        # ---------------------------------------------------------------------
        (
            optimized_data,
            LS,
            MLE,
            Chi,
            LSA,
            Z,
            V_matrix,
            CI,
            t_values,
            t_m,
            tv_input_m,
            ti_input_m,
            tv_output_m,
            ti_output_m,
            WLS,
            obs,
            MSE,
            R2_responses
        ) = _uncert_metrics(
            initial_x0,
            data,
            [solver],
            initial_x0,
            thetac,
            epsf,
            thetas,
            ti_iphi_vars,
            tv_iphi_vars,
            tv_ophi_vars,
            ti_ophi_vars,
            system,
            models
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
            'Z': Z,                       # Jacobian normalized by std deviations
            'WLS': WLS,                   # Weighted least squares
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
            'found_eps': epsf
        }

    iden_opt['eps'] = epsf
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
        models,
        logging
    )

    # Construct return dictionary
    uncert_output = {
        'results': resultun2,
        'theta_parameters': theta_parameters,
        'solver_parameters': solver_parameters,
        'scaled_params': scaled_params,
        'obs': observed_values
    }

    return uncert_output




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
    system,
    models
):
    """
    Calculate uncertainty metrics for parameter estimation results.

    Parameters
    ----------
    theta : list
        Estimated (normalized) parameter values.
    data : dict
        Experimental dataset across sheets.
    active_solvers : list
        Active solver identifiers.
    x0 : list
        Normalized initial guess of parameters.
    thetac : list
        Parameter scaling constants.
    eps : dict
        Perturbation size for sensitivity analysis by solver.
    thetas : list of bool
        Active parameter flags.
    ti_iphi_vars : list
        Time-invariant input variable names.
    tv_iphi_vars : list
        Time-variant input variable names.
    tv_ophi_vars : list
        Time-variant output variable names.
    ti_ophi_vars : list
        Time-invariant output variable names.
    system : dict
        System structure including variable definitions.
    models : dict
        Modelling configuration and external functions.

    Returns
    -------
    tuple
        A collection of results: (data, LS, MLE, Chi, LSA, V_matrix, CI,
        t_values, t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m,
        JWLS, obs, MSE, R2_responses)
    """
    if len(theta) != len(x0):
        raise ValueError("Length of theta and x0 must match.")

    theta = [theta[i] if thetas[i] else x0[i] for i in range(len(thetas))]
    std_dev, global_y_true, global_y_pred, maxv = {}, {}, {}, {}
    var_measurements = {}
    LSA = np.zeros((0, np.sum(thetas)))

    phisc = {var: 1 for var in ti_iphi_vars}
    phitsc = {var: 1 for var in tv_iphi_vars}

    t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m = {}, {}, {}, {}, {}
    tsc = 1

    for group in ['tvo', 'tio']:
        for var in system.get(group, {}):
            if system[group][var].get('meas', True):
                sigma = system[group][var].get('unc', 1.0)
                std_dev[var] = 1.0 if sigma is None or np.isnan(sigma) else sigma

    for sheet_name, sheet_data in data.items():
        t_values = {
            var: np.array(sheet_data[f"MES_X:{var}"])[~np.isnan(sheet_data[f"MES_X:{var}"])]
            for var in tv_ophi_vars if f"MES_X:{var}" in sheet_data
        }

        swps_data = {}
        for var in tv_iphi_vars:
            if f'{var}t' in sheet_data and f'{var}l' in sheet_data:
                times = np.array(sheet_data[f'{var}t'])[~np.isnan(sheet_data[f'{var}t'])]
                levels = np.array(sheet_data[f'{var}l'])[~np.isnan(sheet_data[f'{var}l'])]
                if times.size and levels.size:
                    swps_data[f'{var}t'] = times
                    swps_data[f'{var}l'] = levels

        ti_data = {var: np.array(sheet_data.get(var, [np.nan]))[0] for var in ti_iphi_vars}
        tv_data = {var: np.array(sheet_data.get(var, []))[~np.isnan(sheet_data.get(var, []))] for var in tv_iphi_vars}
        tv_out_data = {var: np.array(sheet_data.get(f"MES_Y:{var}", []))[~np.isnan(sheet_data.get(f"MES_Y:{var}", []))] for var in tv_ophi_vars if f"MES_Y:{var}" in sheet_data}
        ti_out_data = {var: np.array(sheet_data.get(f"MES_Y:{var}", [np.nan]))[0] for var in ti_ophi_vars if f"MES_Y:{var}" in sheet_data}

        maxv.setdefault(sheet_name, {})

        cvp = {var: sheet_data[f"CVP:{var}"].iloc[0] for var in tv_iphi_vars if f"CVP:{var}" in sheet_data}

        solver_name = active_solvers[0]
        if not solver_name or "X:all" not in sheet_data:
            continue

        t_vals = np.unique(np.array(sheet_data["X:all"])[~np.isnan(sheet_data["X:all"])]).tolist()

        tv_pred, ti_pred, _ = simula(
            t_vals, swps_data, ti_data, phisc, phitsc, tsc,
            theta, thetac, cvp, tv_data, solver_name, system, models
        )

        indices, tv_filt = {}, {}
        for var in tv_ophi_vars:
            if var in t_values:
                indices[var] = np.isin(t_vals, t_values[var])
                if var in tv_pred:
                    tv_filt[var] = np.array(tv_pred[var])[indices[var]]

        y_true_combined, y_model_combined = [], []
        for var in tv_ophi_vars:
            y_true_arr = tv_out_data.get(var, np.array([]))
            y_pred_arr = tv_filt.get(var, np.zeros_like(y_true_arr))
            n = min(len(y_true_arr), len(y_pred_arr))
            y_true_arr, y_pred_arr = y_true_arr[:n], y_pred_arr[:n]
            max_val = max(np.max(np.abs(y_true_arr)), np.max(np.abs(y_pred_arr)), 1.0)
            maxv[sheet_name][var] = max_val

            global_y_true.setdefault(var, []).extend(y_true_arr / max_val)
            global_y_pred.setdefault(var, []).extend(y_pred_arr / max_val)
            y_true_combined.extend(y_true_arr / max_val)
            y_model_combined.extend(y_pred_arr / max_val)

            sigma = std_dev.get(var, 1.0)
            for yt, yp in zip(y_true_arr, y_pred_arr):
                var_measurements.setdefault(var, []).append((yt, yp, sigma))

        for var in ti_ophi_vars:
            yt, yp = ti_out_data.get(var, np.nan), ti_pred.get(var, np.nan)
            if not np.isnan(yt) and not np.isnan(yp):
                max_val = max(abs(yt), abs(yp), 1.0)
                maxv[sheet_name][var] = max_val
                global_y_true.setdefault(var, []).append(yt / max_val)
                global_y_pred.setdefault(var, []).append(yp / max_val)
                y_true_combined.append(yt / max_val)
                y_model_combined.append(yp / max_val)
                var_measurements.setdefault(var, []).append((yt, yp, std_dev.get(var, 1.0)))

        # LSA
        LSAe = np.zeros((len(y_model_combined), np.sum(thetas)))
        for i, idx in enumerate(np.where(thetas)[0]):
            perturbed_theta = theta.copy()
            perturbed_theta[idx] += eps[solver_name]
            tv_mod, ti_mod, _ = simula(
                t_vals, swps_data, ti_data, phisc, phitsc, tsc,
                perturbed_theta, thetac, cvp, tv_data, solver_name, system, models
            )
            mod_combined = []
            for var in tv_ophi_vars:
                yp_mod = np.array(tv_mod[var])[indices[var]][:len(tv_filt[var])]
                mod_combined.extend(yp_mod / maxv[sheet_name][var])
            for var in ti_ophi_vars:
                yp_mod = ti_mod.get(var, np.nan)
                if not np.isnan(yp_mod):
                    mod_combined.append(yp_mod / maxv[sheet_name][var])
            LSAe[:, i] = (np.array(mod_combined) - np.array(y_model_combined)) / eps[solver_name]
        LSA = np.vstack([LSA, LSAe])
        t_m[sheet_name] = t_vals
        tv_input_m[sheet_name] = tv_data
        ti_input_m[sheet_name] = ti_data
        tv_output_m[sheet_name] = tv_pred
        ti_output_m[sheet_name] = ti_pred

    all_y_true, all_y_pred = [], []
    for var in global_y_true:
        y_t, y_p = np.array(global_y_true[var]), np.array(global_y_pred[var])
        n = min(len(y_t), len(y_p))
        all_y_true.extend(y_t[:n])
        all_y_pred.extend(y_p[:n])

    all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)
    obs = len(all_y_true)
    ss_res_scaled = np.sum((all_y_true - all_y_pred) ** 2)
    ss_tot_scaled = np.sum((all_y_true - np.mean(all_y_true)) ** 2)
    LS = 1 - (ss_res_scaled / ss_tot_scaled) if ss_tot_scaled != 0 else np.nan

    sum_wls = sum_mle = sum_chi = 0.0
    R2_responses = {}
    for var, vals in var_measurements.items():
        y_t, y_p, sigma = zip(*vals)
        y_t, y_p, sigma = np.array(y_t), np.array(y_p), np.array(sigma)
        resid = (y_t - y_p) / sigma
        resid2 = (y_t - y_p)**2 / sigma**2
        sum_wls += np.sum(resid2**2)
        sum_mle += np.sum(0.5 * (np.log(2 * math.pi * sigma**2) + resid**2))
        sigma_safe = np.where(sigma == 0, 1e-8, sigma)
        sum_chi += np.sum(((y_t - y_p) ** 2) / sigma_safe**2)

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        R2_responses[var] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    residuals_scaled = all_y_true - all_y_pred
    MSE = np.sum(residuals_scaled**2) / obs

    dof = max(obs - np.sum(thetas), 1)
    trv = stats.t.ppf(0.975, dof)
    J = LSA
    JtJ = np.dot(J.T, J).astype(np.float64)
    if np.linalg.cond(JtJ) > 1e10:
        JtJ += np.eye(JtJ.shape[0]) * 1e-10

    sigma_ls = np.sum(residuals_scaled**2) / dof
    try:
        V_matrix = sigma_ls * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        V_matrix = sigma_ls * np.linalg.pinv(JtJ)

    diag = np.diag(V_matrix)
    CI = trv * np.sqrt(diag)
    param_values = np.array(theta)[thetas]
    t_values = param_values / (CI + 1e-12)

    # Compute Z matrix
    theta_std = np.sqrt(np.diag(V_matrix))
    response_stds = []
    for var, vals in var_measurements.items():
        response_stds.extend([sigma for (_, _, sigma) in vals])
    response_stds = np.array(response_stds)
    theta_std_safe = np.where(theta_std == 0, 1e-12, theta_std)
    response_stds_safe = np.where(response_stds == 0, 1e-12, response_stds)

    Z = LSA / np.outer(response_stds_safe, theta_std_safe)

    return (
        data, LS, sum_mle, sum_chi, LSA, Z, V_matrix, CI, t_values,
        t_m, tv_input_m, ti_input_m, tv_output_m, ti_output_m,
        sum_wls, obs, MSE, R2_responses
    )





def _report(result, mutation, theta_parameters, models, logging):
    """
    Report the uncertainty and parameter estimation results.

    Parameters:
    ----------
    result : dict
        Dictionary containing the results of the optimization.
    mutation : dict
        Dictionary indicating which parameters are to be optimized for each model.
    theta_parameters : dict
        Dictionary of theta parameters for each model.
    models : dict
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

        # Update modelling settings with V_matrix for the model
        models['V_matrix'][solver] = solver_results['V_matrix']

        # Map CI, t-values, and CI ratios to their original parameter positions
        CI_mapped = {pos: solver_results['CI'][i] for i, pos in enumerate(original_positions)}
        t_values_mapped = {pos: solver_results['t_values'][i] for i, pos in enumerate(original_positions)}
        CI_ratio_mapped = {
            pos: ci * 100 / param if param != 0 else float('inf')
            for pos, ci, param in zip(original_positions, solver_results['CI'], solver_results['optimization_result'].x)
        }

        # Log t-values if requested
        if logging:
            print(f"T-values of model {solver}: {solver_results['t_values']}")

        # Store parameters, covariance matrices, and confidence intervals
        solver_parameters[solver] = solver_results['optimization_result'].x
        solver_cov_matrices[solver] = solver_results['V_matrix']
        solver_confidence_intervals[solver] = CI_mapped
        solver_results['P'] = ((1 / (solver_results['Chi']) ** 2) / sum_of_chi_squared) * 100
        print(f"P-value of model:{solver} is {solver_results['P']} for model discrimination")

        # Log R² for responses
        if 'R2_responses' in solver_results:
            r2_responses = solver_results['R2_responses']
            if logging:
                print(f"R2 values for responses in model {solver}:")
                for var, r2_value in r2_responses.items():
                    print(f"  {var}: {r2_value:.4f}")

            # Optionally store R² values in result for external use
            solver_results['R2_responses_summary'] = {
                var: round(r2_value, 4) for var, r2_value in r2_responses.items()
            }

    return scaled_params, solver_parameters, solver_cov_matrices, solver_confidence_intervals, result



def _fdm_mesh_independency(theta, thetac, solver, system, models,
                           tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars, data):
    """
    Perform FDM mesh dependency test to determine a suitable epsilon for sensitivity analysis.

    Parameters:
    theta (list): Initial parameter estimates.
    thetac (list): Scaling factors for parameters.
    solver (str): Solver name.
    system (dict): Model structure.
    models (dict): Modelling settings.
    tv_iphi_vars (list): Time-variant input variable names.
    ti_iphi_vars (list): Time-invariant input variable names.
    tv_ophi_vars (list): Time-variant output variable names.
    ti_ophi_vars (list): Time-invariant output variable names.
    data (dict): Experimental data.

    Returns:
    float: Optimal epsilon value for sensitivity analysis.
    """
    logger.info(f"Performing FDM mesh dependency test for model {solver}.")

    eps_values = np.logspace(-8, -1, 20)
    determinant_changes = []

    for eps in eps_values:
        try:
            result = _uncert_metrics(
                theta, data, [solver], theta, thetac, {solver: eps}, [True] * len(theta),
                ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars, system, models
            )
            V_matrix = result[6]  # Adjust index if needed
            det_val = np.linalg.det(V_matrix)
            determinant_changes.append(det_val)
        except np.linalg.LinAlgError:
            logger.warning(f"LinAlgError at eps={eps:.1e} for {solver}")
            determinant_changes.append(np.nan)

    determinant_changes = np.array(determinant_changes)

    # Plot determinant variations
    folder = Path.cwd() / "meshindep_plots"
    folder.mkdir(exist_ok=True)
    filename_base = folder / f"{solver}_eps_dependency_test"

    plt.figure(figsize=(8, 6))
    plt.plot(eps_values, determinant_changes, marker='o', label='Det(V_matrix)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Determinant of Variance-Covariance Matrix')
    plt.title(f'Epsilon Dependency Test for model {solver}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300)
    if plt.get_backend() != 'agg':  # Show only in interactive environments
        plt.show()
    plt.close()

    # Save data to Excel
    df = pd.DataFrame({
        'Epsilon': eps_values,
        'Determinant': determinant_changes
    })
    df.to_excel(f"{filename_base}.xlsx", index=False)

    # Determine optimal epsilon based on stability in the determinant
    stable_region = np.where(np.isfinite(determinant_changes))[0]

    if stable_region.size == 0:
        raise ValueError(f"Could not determine a stable epsilon region for model {solver}.")

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
    logger.info(f"Optimal epsilon selected for model {solver}: {optimal_eps:.2e}")

    return optimal_eps