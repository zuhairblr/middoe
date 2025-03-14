import os
from middoe.iden_utils import validation_R2, validation_params, Plotting_Results
from middoe.iden_parmest import Parmest
from middoe.iden_uncert import Uncert


def validation(data_storage, model_structure, modelling_settings, estimation_settings, Simula, round_data, framework_settings):
    """
    Perform leave-one-out validation to evaluate the generalization of the parameter estimation.

    Parameters:
    data_storage (dict): Experimental data observations.
    model_structure (dict): Structure of the model, including variables and their properties.
    modelling_settings (dict): Settings related to the modelling process, including theta parameters.
    estimation_settings (dict): Settings for the estimation process, including active solvers and plotting options.
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

        resultpr = Parmest(
            model_structure,
            modelling_settings,
            estimation_settings,
            training_data,
            Simula,
        )

        resultun_pred, theta_parameters_pred, solver_parameters_pred, scaled_params_pred, obs_pred = Uncert(
            training_data,
            resultpr,
            model_structure,
            modelling_settings,
            estimation_settings,
            Simula
        )

        resultun_val, theta_parameters_val, _, _, obs_val = Uncert(
            validation_data,
            resultpr,
            model_structure,
            modelling_settings,
            estimation_settings,
            Simula
        )
        plotting1 = Plotting_Results(modelling_settings, framework_settings)  # Instantiate Plotting class
        plotting1.fit_plot(validation_data, resultun_val, f'val{i+1}', model_structure)
        R2_prd[i+1], R2_val[i+1], parameters[i+1], R2_ref, ref_params, MSE_pred[i+1], MSE_val[i+1], MSE_ref = compute_validation_error(resultun_pred, resultun_val, scaled_params_pred, round_data)

    base_path = framework_settings['path']
    modelling_folder = str(framework_settings['case'])   # No leading backslash here

    # Join the base path and modelling folder
    filename = os.path.join(base_path, modelling_folder)

    # Ensure the 'modelling' directory exists
    os.makedirs(filename, exist_ok=True)
    base_path = filename  # Assuming filename contains the base path
    modelling_folder = 'validation'
    full_path = os.path.join(base_path, modelling_folder)
    os.makedirs(full_path, exist_ok=True)
    validation_R2(R2_prd, R2_val, R2_ref, full_path, case='R2')
    validation_R2(MSE_pred, MSE_val, MSE_ref, full_path, case='MSE')
    validation_params(parameters, ref_params, full_path)


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

