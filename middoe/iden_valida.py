import copy
from middoe.iden_utils import validation_R2
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
import numpy as np


def validation(data_storage, system, models, iden_opt, round_data):
    """
    Perform leave-one-out validation to evaluate the generalization of the parameter estimation.

    This function splits the experimental data into training and validation sets for each fold,
    performs parameter estimation on the training set, and evaluates the model's performance
    on the validation set. It computes metrics such as R2 and MSE for both prediction and validation.

    Parameters
    ----------
    data_storage : dict
        Experimental data observations, where each key corresponds to a dataset.
    system : dict
        Structure of the model, including variables and their properties.
    models : dict
        Settings related to the modelling process, including theta parameters and mutation masks.
    iden_opt : dict
        Settings for the estimation process, including active solvers and plotting options.
    round_data : dict
        Data from previous rounds of the estimation process, including optimization results.

    Returns
    -------
    dict
        A dictionary containing the validation results, including:
        - R2_prd : dict
            R2 values for predictions in each fold.
        - R2_val : dict
            R2 values for validations in each fold.
        - R2_stats : dict
            Mean and standard deviation of R2 values for predictions and validations.
        - MSE_stats : dict
            Mean and standard deviation of MSE values for predictions and validations.
    """
    iden_optc = copy.deepcopy(iden_opt)
    iden_optc['var-cov'] = 'M'
    # Initialize cross-validation results
    sheet_names = list(data_storage.keys())
    # sheet_names = list(data_storage.keys())[1:-1]
    n_sheets = len(sheet_names)
    R2_prd, R2_val, parameters, MSE_pred, MSE_val = {}, {}, {}, {}, {}
    models['mutation'] = {
        solver: round_data[list(round_data.keys())[-1]]['result'][solver]['optimization_result']['activeparams']
        for solver in models['can_m']
    }

    for i in range(n_sheets):
        # Split data into training and validation sets
        validation_sheet = sheet_names[i]
        training_sheets = [sheet for sheet in sheet_names if sheet != validation_sheet]

        training_data = {sheet: data_storage[sheet] for sheet in training_sheets}
        validation_data = {validation_sheet: data_storage[validation_sheet]}

        print(f"Running validation fold {i + 1}/{n_sheets}...")
        print(f"Validation sheet: {validation_sheet}")

        resultpr = parmest(system, models, iden_opt, training_data, case='strov')
        uncert_results_pred = uncert(training_data, resultpr, system, models, iden_optc)

        resultun_pred = uncert_results_pred['results']
        scaled_params_pred = {solver: resultun_pred[solver]['optimization_result']['scpr'] for solver in resultun_pred}

        uncert_results_vals = uncert(validation_data, resultpr, system, models, iden_optc)
        resultun_val = uncert_results_vals['results']

        # plotting1 = Plotting_Results(models, f'val{i+1}')  # Instantiate Plotting class
        # plotting1.fit_plot(validation_data, resultun_val, system)
        R2_prd[i + 1], R2_val[i + 1], R2_ref, MSE_pred[i + 1], MSE_val[i + 1], MSE_ref = compute_validation_error(
            resultun_pred, resultun_val, scaled_params_pred, round_data)

    # Ensure the 'modelling' directory exists
    validation_R2(R2_prd, R2_val, R2_ref, case='R2')
    validation_R2(MSE_pred, MSE_val, MSE_ref, case='MSE')
    # validation_params(parameters, ref_params)

    # Compute average and std dev for each solver
    R2_stats, MSE_stats = {}, {}
    solvers = models['can_m']

    for solver in solvers:
        pred_r2_values = [R2_prd[i + 1][solver] for i in range(n_sheets)]
        val_r2_values = [R2_val[i + 1][solver] for i in range(n_sheets)]
        pred_mse_values = [MSE_pred[i + 1][solver] for i in range(n_sheets)]
        val_mse_values = [MSE_val[i + 1][solver] for i in range(n_sheets)]

        R2_stats[solver] = {
            'prediction_mean': np.mean(pred_r2_values),
            'prediction_std': np.std(pred_r2_values),
            'validation_mean': np.mean(val_r2_values),
            'validation_std': np.std(val_r2_values)
        }

        MSE_stats[solver] = {
            'prediction_mean': np.mean(pred_mse_values),
            'prediction_std': np.std(pred_mse_values),
            'validation_mean': np.mean(val_mse_values),
            'validation_std': np.std(val_mse_values)
        }

    validres = {'R2_prd': R2_prd,
                'R2_val': R2_val,
                'R2_stats': R2_stats,
                'MSE_stats': MSE_stats}

    print(validres)

    return validres


def compute_validation_error(resultun_pred, resultun_val, scaled_params, round_data):
    prediction_R2, validation_R2 = {}, {}
    prediction_MSE, validation_MSE = {}, {}
    R2_ref, MSE_ref = {}, {}

    last_key = list(round_data.keys())[-1]

    for solver in resultun_pred:
        if solver in resultun_val:
            prediction_R2[solver] = resultun_pred[solver]['R2_total']
            prediction_MSE[solver] = resultun_pred[solver]['MSE']
            validation_R2[solver] = resultun_val[solver]['R2_total']
            validation_MSE[solver] = resultun_val[solver]['MSE']
            R2_ref[solver] = round_data[last_key]['result'][solver]['R2_total']
            MSE_ref[solver] = round_data[last_key]['result'][solver]['MSE']
            # params = scaled_params
            # ref_params[solver] = round_data[last_key]['scaled_params'][solver]

    return prediction_R2, validation_R2, R2_ref, prediction_MSE, validation_MSE, MSE_ref
