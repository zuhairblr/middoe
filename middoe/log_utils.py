import pickle
import os
import pandas as pd
from scipy import stats
from middoe.iden_utils import Plotting_Results
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
    solver_parameters (dict): Parameters for the model.
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    obs (int): Number of observations.
    case (str): Case identifier, used for naming files and directories.
    data_storage (dict): Dictionary to store data of experiments.
    system (dict): User provided - The model structure information.

    Returns:
    dict: Reference t-value for each model-round observation.
    """
    round_key = f'Round {round}'
    dof = {solver: obs - len(theta_parameters[solver]) for solver in models['can_m']}
    trv = {solver: stats.t.ppf(1 - (1 - 0.95) / 2, dof[solver]) for solver in models['can_m']}
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
    for solver in models['can_m']:
        if solver in models.get('original_positions', {}):
            round_data[round_key]['original_positions'][solver] = models['original_positions'][solver]
        else:
            round_data[round_key]['original_positions'][solver] = []  # Or handle as needed

    if round == 1:
        add_norm_par(models)
    else:
        for solver in models['can_m']:
            # Initialize 'normalized_parameters' if it doesn't exist
            if 'normalized_parameters' not in models:
                models['normalized_parameters'] = {}

            # Create the entry for the model if it doesn't exist
            if solver not in models['normalized_parameters']:
                models['normalized_parameters'][solver] = []

            # Update the normalized parameters for the model
            models['normalized_parameters'][solver] = result[solver]['estimations_normalized']

    # ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
    for solver in models['can_m']:
        models['V_matrix'][solver] = result[solver]['V_matrix']

    solver_cov_matrices = {solver: result[solver]['V_matrix'] for solver in models['can_m']}
    solver_confidence_intervals = {solver: result[solver]['CI'] for solver in models['can_m']}
    plotting1 = Plotting_Results(models, iden_opt['plt_s'], round)  # Instantiate Plotting class
    if iden_opt['c_plt'] == True:
        plotting1.conf_plot(solver_parameters, solver_cov_matrices, solver_confidence_intervals)
    if iden_opt['f_plt'] == True:
        plotting1.fit_plot(data_storage, result, system)

    for solver in models['can_m']:
        print(f'reference t value for model {solver} and round {round}: {trv[solver]}')
        print(f'estimated t values for model {solver} and round {round}: {result[solver]["t_values"]}')
        print(f'P-value for model {solver} and round {round}: {result[solver]["P"]}')
        print(f'eps for model {solver} and round {round}: {result[solver]["found_eps"][solver]}')
    print()
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

def load_from_jac():
    """
    Attempt to load 'round_data.jac' and 'sensa_results.jac' if available in the current directory.

    Returns:
    dict: Dictionary with keys 'iden' and 'sensa'. Each maps to the loaded object or None.
          Returns None if neither file is found.
    """
    filenames = {
        'iden': 'iden_results.jac',
        'sensa': 'sensa_results.jac'
    }

    loaded_data = {}
    found_any = False

    for key, fname in filenames.items():
        if os.path.exists(fname):
            try:
                with open(fname, 'rb') as f:
                    loaded_data[key] = pickle.load(f)
                found_any = True
                print(f"Loaded: {fname}")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                loaded_data[key] = None
        else:
            print(f"File not found: {fname}")
            loaded_data[key] = None

    return loaded_data if found_any else None

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
    if 'theta' in modelling_settings:
        # Create 'normalized_parameters' as a dictionary with lists of ones
        modelling_settings['normalized_parameters'] = {
            key: [1] * len(value) for key, value in modelling_settings['theta'].items()
        }
    else:
        raise KeyError("The dictionary must contain 'theta_parameters' as a key.")

    return modelling_settings

# def save_sobol_results_to_excel(sensa):
#     """
#     Saves Sobol analysis results to 'sobol_results.xlsx' in the current working directory.
#
#     Parameters:
#     - sensa: Dictionary containing Sobol analysis results (must have a top-level 'analysis' key).
#     """
#     if not sensa or not isinstance(sensa, dict) or 'analysis' not in sensa:
#         print("Error: Sobol results are not available. No file will be created.")
#         return
#
#     file_path = os.path.join(os.getcwd(), "sobol_results.xlsx")
#     sobol_analysis = sensa['analysis']
#
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         for model_name, model_data in sobol_analysis.items():
#             data = {'x_axis_steps': []}
#
#             # Determine if step data is a list or dict
#             steps = model_data.get(model_name)
#             if isinstance(steps, list):
#                 for step_index, step_data in enumerate(steps):
#                     data['x_axis_steps'].append(step_index)
#                     for i, st_val in enumerate(step_data.get('ST', [])):
#                         col = f"Parameter_{i + 1}"
#                         data.setdefault(col, []).append(st_val)
#             elif isinstance(steps, dict):
#                 for step, step_data in steps.items():
#                     data['x_axis_steps'].append(step)
#                     for i, st_val in enumerate(step_data.get('ST', [])):
#                         col = f"Parameter_{i + 1}"
#                         data.setdefault(col, []).append(st_val)
#
#             pd.DataFrame(data).to_excel(writer, sheet_name=model_name, index=False)
#
#     print(f"Sobol analysis results have been saved to: {file_path}")


def save_to_xlsx(sensa):
    """
    Save Sobol analysis results to 'sobol_results.xlsx' in the current directory.

    Parameters:
    - sensa: dict, with structure sensa['analysis'][model][response] = list of time-point dicts
    - sobol_problem: dict, with sobol_problem[model]['names'] = list of parameter names
    """
    if not sensa or not isinstance(sensa, dict) or 'analysis' not in sensa:
        print("Error: Sobol results not available. No file written.")
        return

    file_path = os.path.join(os.getcwd(), "sobol_results.xlsx")
    sobol_analysis = sensa['analysis']
    sheets_to_write = {}

    for model_name, response_dict in sobol_analysis.items():
        for response_key, step_data_list in response_dict.items():
            if not isinstance(step_data_list, list) or not step_data_list:
                print(f"Skipping {model_name}-{response_key}: no data.")
                continue

            num_params = len(step_data_list[0].get('ST', []))
            time = list(range(len(step_data_list)))

            try:
                names = sobol_problem[model_name]['names']
            except Exception:
                names = [f'Param_{i+1}' for i in range(num_params)]

            df_data = {'Time': time}
            for i in range(num_params):
                col_name = names[i] if i < len(names) else f'Param_{i+1}'
                df_data[col_name] = [step.get('ST', [0]*num_params)[i] for step in step_data_list]

            df = pd.DataFrame(df_data)

            sheet_name = f"{model_name}_{response_key}"
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]

            sheets_to_write[sheet_name] = df

    if not sheets_to_write:
        print("No valid sheets found. No Excel file created.")
        return

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_to_write.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Sobol results saved to: {file_path}")