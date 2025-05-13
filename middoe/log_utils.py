import copy
import pickle
import os
import pandas as pd
from scipy import stats
from middoe.iden_utils import Plotting_Results
from middoe.des_md import MD
import importlib
from pathlib import Path

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
    des_opt (dict): User provided - Design settings for the MBDOE.
    system (dict): User provided - The model structure information.
    models (dict): User provided - The settings for the modelling process.
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

def read_excel(data_type):
    """
    Read data from 'indata.xlsx' or 'exdata.xlsx' in the current working directory.

    Parameters:
    data_type (str): Must be either 'indata' or 'exdata'.

    Returns:
    dict: Data read from the Excel file, with sheet names as keys.

    Raises:
    ValueError: If data_type is not 'indata' or 'exdata'.
    FileNotFoundError: If the expected file does not exist.
    """
    if data_type not in ['indata', 'exdata']:
        raise ValueError("data_type must be either 'indata' or 'exdata'")

    file_path = Path.cwd() / f"{data_type}.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path.name} not found in the current directory.")

    print(f"[INFO] Reading from {file_path.name}")
    data = pd.read_excel(file_path, sheet_name=None)
    return data

def save_rounds(round, result, theta_parameters, design_type, round_data, models, scaled_params, iden_opt, solver_parameters, obs, data_storage, system, ranking= None, k_optimal_value= None, rCC_values= None, J_k_values= None):
    """
    Save data for each round of model identification, and append to prior.

    Parameters:
    round (int): Round number, the current round of the design - conduction and identification procedure.
    ranking (list): Ranking of parameters, from estimability analysis.
    k_optimal_value (float): Optimal number of parameters to be estimated, from estimability analysis.
    rCC_values (list): Corrected critical ratios for models.
    J_k_values (list): Objectives of weighted least square method based optimization for models.
    result (dict): Result of the identification procedure (estimation and uncertainty analysis) for models.
    theta_parameters (dict): Parameters for models.
    design_type (str): Type of design (classic or DOE).
    round_data (dict): Dictionary to append the data for the current round.
    models (dict): User provided - The settings for the modelling process.
    scaled_params (dict): Estimated parameters scaled to the original scale for each model-round observations.
    iden_opt (dict): Settings for the estimation process, including active solvers and plotting options.
    solver_parameters (dict): Parameters for the solver.
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    obs (int): Number of observations.
    case (str): Case identifier, used for naming files and directories.
    data_storage (dict): Dictionary to store data of experiments.
    system (dict): User provided - The model structure information.

    Returns:
    dict: Reference t-value for each model-round observation.
    """
    round_key = f'Round {round}'
    dof = {solver: obs - len(theta_parameters[solver]) for solver in models['active_solvers']}
    trv = {solver: stats.t.ppf(1 - (1 - 0.95) / 2, dof[solver]) for solver in models['active_solvers']}
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
        'iden_opt': iden_opt,
        'system': system
    }

    # Save mutation information
    for solver, mutation in models['mutation'].items():
        round_data[round_key]['mutation'][solver] = mutation

    # Save original positions information, if it exists
    for solver in models['active_solvers']:
        if solver in models.get('original_positions', {}):
            round_data[round_key]['original_positions'][solver] = models['original_positions'][solver]
        else:
            round_data[round_key]['original_positions'][solver] = []  # Or handle as needed

    if round == 1:
        add_norm_par(models)
    else:
        for solver in models['active_solvers']:
            # Initialize 'normalized_parameters' if it doesn't exist
            if 'normalized_parameters' not in models:
                models['normalized_parameters'] = {}

            # Create the entry for the solver if it doesn't exist
            if solver not in models['normalized_parameters']:
                models['normalized_parameters'][solver] = []

            # Update the normalized parameters for the solver
            models['normalized_parameters'][solver] = result[solver]['estimations_normalized']

    # ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
    for solver in models['active_solvers']:
        models['V_matrix'][solver] = result[solver]['V_matrix']

    solver_cov_matrices = {solver: result[solver]['V_matrix'] for solver in models['active_solvers']}
    solver_confidence_intervals = {solver: result[solver]['CI'] for solver in models['active_solvers']}
    plotting1 = Plotting_Results(models, round)  # Instantiate Plotting class
    if iden_opt['con_plot'] == True:
        plotting1.conf_plot(solver_parameters, solver_cov_matrices, solver_confidence_intervals)
    if iden_opt['fit_plot'] == True:
        plotting1.fit_plot(data_storage, result, system)

    return trv


def save_to_jac(results, purpose):
    """
    Save results to a .jac file in the project directory using a fixed name.

    Parameters:
    results (dict): The results to save.
    purpose (str): Either 'iden' or 'sensa' to determine the file name.
    """
    try:
        if purpose == "iden":
            file_path = Path.cwd() / "iden_results.jac"
        elif purpose == "sensa":
            file_path = Path.cwd() / "sensa_results.jac"
        else:
            raise ValueError("Purpose must be 'iden' or 'sensa'")

        with open(file_path, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] Results saved to: {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

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
    models (dict): Dictionary containing modelling settings, including 'theta_parameters'.

    Returns:
    dict: Updated modelling settings with 'normalized_parameters' added.

    Raises:
    KeyError: If 'theta_parameters' is not present in the modelling settings.
    """
    # Check if 'theta_parameters' is present in models
    if 'theta_parameters' in modelling_settings:
        # Create 'normalized_parameters' as a dictionary with lists of ones
        modelling_settings['normalized_parameters'] = {
            key: [1] * len(value) for key, value in modelling_settings['theta_parameters'].items()
        }
    else:
        raise KeyError("The dictionary must contain 'theta_parameters' as a key.")

    return modelling_settings
