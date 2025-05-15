import os
from middoe.iden_utils import validation_R2, validation_params, Plotting_Results
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert


def validation(data_storage, system, models, iden_opt, round_data):
    """
    Perform leave-one-out validation to evaluate the generalization of the parameter estimation.

    Parameters:
    data_storage (dict): Experimental data observations.
    system (dict): Structure of the model, including variables and their properties.
    models (dict): Settings related to the modelling process, including theta parameters.
    iden_opt (dict): Settings for the estimation process, including active solvers and plotting options.
    Simula (module): The simulation module used for the experiments.

    Returns:
    dict: A dictionary containing the validation results.
    """
    # Initialize cross-validation results
    sheet_names = list(data_storage.keys())
    # sheet_names = list(data_storage.keys())[1:-1]
    n_sheets = len(sheet_names)
    R2_prd, R2_val, parameters, MSE_pred, MSE_val = {}, {}, {}, {}, {}

    for i in range(n_sheets):
        # Split data into training and validation sets
        validation_sheet = sheet_names[i]
        training_sheets = [sheet for sheet in sheet_names if sheet != validation_sheet]

        training_data = {sheet: data_storage[sheet] for sheet in training_sheets}
        validation_data = {validation_sheet: data_storage[validation_sheet]}

        print(f"Running validation fold {i + 1}/{n_sheets}...")
        print(f"Validation sheet: {validation_sheet}")

        resultpr = parmest(system, models,iden_opt, training_data)
        uncert_results_pred = uncert(training_data,resultpr,system,models,iden_opt)
        resultun_pred = uncert_results_pred['results']
        scaled_params_pred = uncert_results_pred['scaled_params']

        # resultun_pred, theta_parameters_pred, solver_parameters_pred, scaled_params_pred, obs_pred = uncert(
        #     training_data,
        #     resultpr,
        #     system,
        #     models,
        #     iden_opt
        # )

        uncert_results_val = uncert(validation_data,resultpr,system,models,iden_opt)
        resultun_val = uncert_results_val['results']

        # resultun_val, theta_parameters_val, _, _, obs_val = uncert(
        #     validation_data,
        #     resultpr,
        #     system,
        #     models,
        #     iden_opt
        # )
        plotting1 = Plotting_Results(models, f'val{i+1}')  # Instantiate Plotting class
        plotting1.fit_plot(validation_data, resultun_val, system)
        R2_prd[i+1], R2_val[i+1], parameters[i+1], R2_ref, ref_params, MSE_pred[i+1], MSE_val[i+1], MSE_ref = compute_validation_error(resultun_pred, resultun_val, scaled_params_pred, round_data)


    # Ensure the 'modelling' directory exists
    validation_R2(R2_prd, R2_val, R2_ref, case='R2')
    validation_R2(MSE_pred, MSE_val, MSE_ref, case='MSE')
    validation_params(parameters, ref_params)


    return R2_prd, R2_val, parameters

def compute_validation_error(resultun_pred, resultun_val, scaled_params, round_data):
    """
    Compute validation error (MSE) between predicted and actual validation results.

    Parameters:
    resultun_pred (dict): Predicted results from the calibration data.
    resultun_val (dict): Validation results from the validation data.

    Returns:
    float: The mean squared error.
    """
    prediction_R2, validation_R2, params, R2_ref, ref_params, prediction_MSE, validation_MSE, MSE_ref = {}, {}, {}, {}, {}, {}, {}, {}
    # Get the last key in round_data
    last_key = list(round_data.keys())[-1]

    for solver in resultun_pred:
        if solver in resultun_val:
            # Extract the 'data' field safely, ensuring it's numeric
            prediction_R2[solver] = resultun_pred[solver]['LS']
            prediction_MSE[solver] = resultun_pred[solver]['MSE']
            validation_R2[solver] = resultun_val[solver]['LS']
            validation_MSE[solver] = resultun_val[solver]['MSE']
            R2_ref[solver] = round_data[last_key]['result'][solver]['LS']
            MSE_ref[solver] = round_data[last_key]['result'][solver]['MSE']
            params = scaled_params
            ref_params[solver] = round_data[last_key]['scaled_params'][solver]

    return prediction_R2, validation_R2, params, R2_ref, ref_params, prediction_MSE, validation_MSE, MSE_ref

