import copy
import pickle
import os
import pandas as pd
from scipy import stats
from middoe.iden_utils import Plotting_Results
from middoe.des_md import MD
from middoe.des_pp import PP
import importlib

def fun_globalizer(func_name):
    """
    Register a proxy function in the kernel_simulator module to call the local function.

    Parameters:
    func_name (str): The name to register the function as.
    """
    kernel_simulator = importlib.import_module('idenpy.kernel_simulator')
    setattr(kernel_simulator, func_name, lambda *args, **kwargs: globals()[func_name](*args, **kwargs))

def run_mbdoe_md(design_settings, model_structure, modelling_settings, core_num, framework_settings, round):
    """
    Run the MBDOE_MD (Model Based Design of Experiments for Model Discrimination).

    Parameters:
    design_settings (dict): User provided - Design settings for the MBDOE.
    model_structure (dict): User provided - The model structure information.
    modelling_settings (dict): User provided - The settings for the modelling process.
    core_num (int): Number of cores to use.
    framework_settings (dict): User provided - Framework settings.
    round (int): The current round of the design - conduction and identification procedure.

    Returns:
    tuple: Design decisions, model discrimination objective function value, and switching points.
    """
    design_settings_copy = copy.deepcopy(design_settings)
    model_structure_copy = copy.deepcopy(model_structure)
    modelling_settings_copy = copy.deepcopy(modelling_settings)
    design_decisions, sum_squared_differences, swps = MD(design_settings_copy, model_structure_copy, modelling_settings_copy, core_num, framework_settings, round)
    return design_decisions, sum_squared_differences, swps

def run_mbdoe_pp(design_settings, model_structure, modelling_settings, core_num, framework_settings, round):
    """
    Run the MBDOE_PP (Model Based Design of Experiments for Parameter Precision).

    Parameters:
    design_settings (dict): User provided - Design settings for the MBDOE.
    model_structure (dict): User provided - The model structure information.
    modelling_settings (dict): User provided - The settings for the modelling process.
    core_num (int): Number of cores to use.
    framework_settings (dict): User provided - Framework settings.
    round (int): The current round of the design - conduction and identification procedure.

    Returns:
    tuple: Design decisions, parameter precision objective function value, and switching points.
    """
    design_settings_copy = copy.deepcopy(design_settings)
    model_structure_copy = copy.deepcopy(model_structure)
    modelling_settings_copy = copy.deepcopy(modelling_settings)
    design_decisions, pp_obj, swps = PP(design_settings_copy, model_structure_copy, modelling_settings_copy, core_num, framework_settings, round)
    return design_decisions, pp_obj, swps


def write_excel(df_combined, experiment_number, excel_path):
    """
    Write the combined DataFrame to an Excel file.

    Parameters:
    df_combined (DataFrame): Combined DataFrame to write.
    experiment_number (int or str): Experiment number to use as the sheet name.
    excel_path (str): Path to the Excel file.
    """
    experiment_number = str(experiment_number)  # Ensure experiment_number is a string
    # Check if the file already exists
    if os.path.isfile(excel_path):
        # Open the file in append mode if it exists
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            existing_sheets = writer.book.sheetnames
            if experiment_number in existing_sheets:
                # Append to the existing sheet
                df_combined.to_excel(writer, sheet_name=experiment_number, index=False)
            else:
                # Create a new sheet
                df_combined.to_excel(writer, sheet_name=experiment_number, index=False)
    else:
        # If the file does not exist, create it and write the data
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name=experiment_number, index=False)


def read_excel(file_path):
    """
    Read data from an Excel file.

    Parameters:
    file_path (str): Path to the Excel file.

    Returns:
    dict: Data read from the Excel file, with sheet names as keys.
    """
    data = pd.read_excel(file_path, sheet_name=None)
    return data

def save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, result, theta_parameters, design_type, round_data, modelling_settings, scaled_params, estimation_settings, solver_parameters, framework_settings, obs, case, data_storage, model_structure):
    """
    Save data for each round of model identification, and append to prior.

    Parameters:
    j (int): Round number, the current round of the design - conduction and identification procedure.
    ranking (list): Ranking of parameters, from estimability analysis.
    k_optimal_value (float): Optimal number of parameters to be estimated, from estimability analysis.
    rCC_values (list): Corrected critical ratios for models.
    J_k_values (list): Objectives of weighted least square method based optimization for models.
    result (dict): Result of the identification procedure (estimation and uncertainty analysis) for models.
    theta_parameters (dict): Parameters for models.
    design_type (str): Type of design (classic or DOE).
    round_data (dict): Dictionary to append the data for the current round.
    modelling_settings (dict): User provided - The settings for the modelling process.
    scaled_params (dict): Estimated parameters scaled to the original scale for each model-round observations.
    estimation_settings (dict): Settings for the estimation process, including active solvers and plotting options.
    solver_parameters (dict): Parameters for the solver.
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    obs (int): Number of observations.
    case (str): Case identifier, used for naming files and directories.
    data_storage (dict): Dictionary to store data of experiments.
    model_structure (dict): User provided - The model structure information.

    Returns:
    dict: Reference t-value for each model-round observation.
    """
    round_key = f'Round {j}'
    dof = {solver: obs - len(theta_parameters[solver]) for solver in modelling_settings['active_solvers']}
    trv = {solver: stats.t.ppf(1 - (1 - 0.95) / 2, dof[solver]) for solver in modelling_settings['active_solvers']}
    round_data[round_key] = {
        'ranking': ranking,
        'k_optimal_value': k_optimal_value,
        'rCC_values': rCC_values,
        'J_k_values': J_k_values,
        'design_type': design_type,
        'mutation': {},
        'original_positions': {},
        'trv': trv,
        'theta_parameters': theta_parameters,
        'scaled_params': scaled_params,
        'result': result,
        'estimation_settings': estimation_settings,
        'framework_settings': framework_settings,
        'model_structure': model_structure
    }

    # Save mutation information
    for solver, mutation in modelling_settings['mutation'].items():
        round_data[round_key]['mutation'][solver] = mutation

    # Save original positions information, if it exists
    for solver in modelling_settings['active_solvers']:
        if solver in modelling_settings.get('original_positions', {}):
            round_data[round_key]['original_positions'][solver] = modelling_settings['original_positions'][solver]
        else:
            round_data[round_key]['original_positions'][solver] = []  # Or handle as needed

    if j == 1:
        add_norm_par(modelling_settings)
    else:
        for solver in modelling_settings['active_solvers']:
            # Initialize 'normalized_parameters' if it doesn't exist
            if 'normalized_parameters' not in modelling_settings:
                modelling_settings['normalized_parameters'] = {}

            # Create the entry for the solver if it doesn't exist
            if solver not in modelling_settings['normalized_parameters']:
                modelling_settings['normalized_parameters'][solver] = []

            # Update the normalized parameters for the solver
            modelling_settings['normalized_parameters'][solver] = result[solver]['estimations_normalized']

    # ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
    for solver in modelling_settings['active_solvers']:
        modelling_settings['V_matrix'][solver] = result[solver]['V_matrix']

    solver_cov_matrices = {solver: result[solver]['V_matrix'] for solver in modelling_settings['active_solvers']}
    solver_confidence_intervals = {solver: result[solver]['CI'] for solver in modelling_settings['active_solvers']}
    plotting1 = Plotting_Results(modelling_settings, framework_settings)  # Instantiate Plotting class
    if estimation_settings['con_plot'] == True:
        plotting1.conf_plot(solver_parameters, solver_cov_matrices, solver_confidence_intervals, j)
    if estimation_settings['fit_plot'] == True:
        plotting1.fit_plot(data_storage, result, j, model_structure)

    return trv


def save_to_jac(results, filename):
    """
    Save results (produced from round data saved or screening-sensitivity analysis) to a .jac file.

    Parameters:
    results (dict): Results to save.
    filename (str): Path to the .jac file.
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")

def load_from_jac(filename):
    """
    Load results (produced from round data saved or screening-sensitivity analysis) from a .jac file.

    Parameters:
    filename (str): Path to the .jac file.

    Returns:
    dict or None: Loaded results, or None if the file does not exist.
    """
    # Check if the file exists before attempting to open it
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            results = pickle.load(file)
        return results
    else:
        print(f"File not found: {filename}")
        return None

def data_appender(df_combined, experiment_number, data_storage):
    """
    Append data for the current experiment to the data storage dictionary
    and return the updated data storage along with the cumulative count of observations.

    Parameters:
    df_combined (DataFrame): Data from the current experiment.
    experiment_number (int or str): The experiment number (used as a key).
    data_storage (dict): Dictionary to store data for each experiment.

    Returns:
    dict: Updated data storage with appended experiment data, and total count of observations.
    """
    # Convert experiment_number to string for consistent dictionary keys
    experiment_number = str(experiment_number)

    # Add the data for this experiment to the storage
    data_storage[experiment_number] = df_combined

    return data_storage


def add_norm_par(modelling_settings):
    """
    Add normalized parameters to the modelling settings.

    Parameters:
    modelling_settings (dict): Dictionary containing modelling settings, including 'theta_parameters'.

    Returns:
    dict: Updated modelling settings with 'normalized_parameters' added.

    Raises:
    KeyError: If 'theta_parameters' is not present in the modelling settings.
    """
    # Check if 'theta_parameters' is present in modelling_settings
    if 'theta_parameters' in modelling_settings:
        # Create 'normalized_parameters' as a dictionary with lists of ones
        modelling_settings['normalized_parameters'] = {
            key: [1] * len(value) for key, value in modelling_settings['theta_parameters'].items()
        }
    else:
        raise KeyError("The dictionary must contain 'theta_parameters' as a key.")

    return modelling_settings
