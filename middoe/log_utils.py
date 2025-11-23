# #log_utils.py
#
# import pickle
# import os
# import pandas as pd
# from scipy import stats
# from middoe.iden_utils import Plotting_Results
# import importlib
# from pathlib import Path
#
# def fun_globalizer(func_name):
#     """
#     Register a proxy function in the kernel_simulator module to call the local function.
#
#     Parameters:
#     func_name (str): The name to register the function as.
#     """
#     kernel_simulator = importlib.import_module('idenpy.kernel_simulator')
#     setattr(kernel_simulator, func_name, lambda *args, **kwargs: globals()[func_name](*args, **kwargs))
#
# def read_excel():
#     """
#     Read data from 'data.xlsx' in the current working directory.
#
#     Returns:
#     dict: Data read from the Excel file, with sheet names as keys and DataFrames as values.
#
#     Raises:
#     FileNotFoundError: If data.xlsx does not exist.
#     """
#
#
#     file_path = Path.cwd() / "data.xlsx"
#
#     if not file_path.exists():
#         raise FileNotFoundError(
#             f"{file_path.name} not found in the current directory. "
#             f"Please create the data file as Excel and add it to your project repository."
#         )
#
#     print(f"[INFO] Reading from {file_path.name}")
#     data = pd.read_excel(file_path, sheet_name=None)
#     return data
#
# def save_rounds(round, result, design_type, round_data, models, iden_opt, obs, system, ranking= None, k_optimal_value= None, rCC_values= None, J_k_values= None,  best_uncert_result= None):
#     """
#     Save data for each round of model identification, and append to prior.
#
#     Parameters
#     ----------
#     round : int
#         Round number, the current round of the design - conduction and identification procedure.
#     result : dict
#         Result of the identification procedure (estimation and uncertainty analysis) for models.
#     design_type : str
#         Type of design (classic or DOE).
#     round_data : dict
#         Dictionary to append the data for the current round.
#     models : dict
#         User provided - The settings for the modelling process.
#     iden_opt : dict
#         Settings for the estimation process, including active solvers and plotting options.
#     obs : int
#         Number of observations.
#     system : dict
#         User provided - The model structure information.
#     ranking : list, optional
#         Ranking of parameters, from estimability analysis.
#     k_optimal_value : float, optional
#         Optimal number of parameters to be estimated, from estimability analysis.
#     rCC_values : list, optional
#         Corrected critical ratios for models.
#     J_k_values : list, optional
#         Objectives of weighted least square method based optimization for models.
#     best_uncert_result : dict, optional
#         Best uncertainty analysis result, if available.
#
#     Returns
#     -------
#     dict
#         Reference t-value for each model-round observation.
#     """
#     data = read_excel()
#     if best_uncert_result:
#         result = best_uncert_result['results']
#
#     round_key = f'Round {round}'
#     dof = {solver: obs - len(result[solver]['estimations']) for solver in models['can_m']}
#     trv = {solver: stats.t.ppf(1 - (1 - 0.95) / 2, dof[solver]) for solver in models['can_m']}
#     scaled_params = {solver: result[solver]['estimations'] for solver in models['can_m']}
#     round_data[round_key] = {
#         'ranking': ranking,
#         'k_optimal_value': k_optimal_value,
#         'rCC_values': rCC_values,
#         'J_k_values': J_k_values,
#         'design_type': design_type,
#         'mutation': {},
#         'original_positions': {},
#         'trv': trv,
#         'scaled_params': scaled_params,
#         'result': result,
#         'iden_opt': iden_opt,
#         'models': models,
#         'system': system,
#         'est_EA':  best_uncert_result
#     }
#
#     # Save mutation information
#     for solver, mutation in models['mutation'].items():
#         round_data[round_key]['mutation'][solver] = mutation
#
#     # Save original positions information, if it exists
#     for solver in models['can_m']:
#         if solver in models.get('original_positions', {}):
#             round_data[round_key]['original_positions'][solver] = models['original_positions'][solver]
#         else:
#             round_data[round_key]['original_positions'][solver] = []  # Or handle as needed
#
#     solver_cov_matrices = {solver: result[solver]['V_matrix'] for solver in models['can_m']}
#     solver_confidence_intervals = {solver: result[solver]['CI'] for solver in models['can_m']}
#     plotting1 = Plotting_Results(models, iden_opt['plt_s'], round)  # Instantiate Plotting class
#     if iden_opt['c_plt'] == True:
#         plotting1.conf_plot(scaled_params, solver_cov_matrices, solver_confidence_intervals)
#     if iden_opt['f_plt'] == True:
#         plotting1.fit_plot(data, result, system)
#
#     for solver in models['can_m']:
#         print(f'reference t value for model {solver} and round {round}: {trv[solver]}')
#         print(f'estimated t values for model {solver} and round {round}: {result[solver]["t_values"]}')
#         print(f'P-value for model {solver} and round {round}: {result[solver]["P"]}')
#     print()
#     return trv
#
# def save_to_jac(results, purpose):
#     """
#     Save results to a .jac file in the project directory using a fixed name.
#
#     Parameters:
#     results (dict): The results to save.
#     purpose (str): Either 'iden' or 'sensa' to determine the file name.
#     """
#     try:
#         if purpose == "iden":
#             file_path = Path.cwd() / "iden_results.jac"
#         elif purpose == "sensa":
#             file_path = Path.cwd() / "sensa_results.jac"
#         else:
#             raise ValueError("Purpose must be 'iden' or 'sensa'")
#
#         with open(file_path, 'wb') as file:
#             pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
#
#         print(f"[INFO] Results saved to: {file_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")
#
# def load_from_jac():
#     """
#     Attempt to load 'round_data.jac' and 'sensa_results.jac' if available in the current directory.
#
#     Returns:
#     dict: Dictionary with keys 'iden' and 'sensa'. Each maps to the loaded object or None.
#           Returns None if neither file is found.
#     """
#     filenames = {
#         'iden': 'iden_results.jac',
#         'sensa': 'sensa_results.jac'
#     }
#
#     loaded_data = {}
#     found_any = False
#
#     for key, fname in filenames.items():
#         if os.path.exists(fname):
#             try:
#                 with open(fname, 'rb') as f:
#                     loaded_data[key] = pickle.load(f)
#                 found_any = True
#                 print(f"Loaded: {fname}")
#             except Exception as e:
#                 print(f"Error loading {fname}: {e}")
#                 loaded_data[key] = None
#         else:
#             print(f"File not found: {fname}")
#             loaded_data[key] = None
#
#     return loaded_data if found_any else None
#
# def data_appender(df_combined, experiment_number, data_storage):
#     """
#     Append data for the current experiment to the data storage dictionary
#     and return the updated data storage along with the cumulative count of observations.
#
#     Parameters:
#     df_combined (DataFrame): Data from the current experiment.
#     experiment_number (int or str): The experiment number (used as a key).
#     data_storage (dict): Dictionary to store data for each experiment.
#
#     Returns:
#     dict: Updated data storage with appended experiment data, and total count of observations.
#     """
#     # Convert experiment_number to string for consistent dictionary keys
#     experiment_number = str(experiment_number)
#
#     # Add the data for this experiment to the storage
#     data_storage[experiment_number] = df_combined
#
#     return data_storage
#
# def add_norm_par(modelling_settings):
#     """
#     Add normalized parameters to the modelling settings.
#
#     Parameters:
#     models (dict): Dictionary containing modelling settings, including 'theta_parameters'.
#
#     Returns:
#     dict: Updated modelling settings with 'normalized_parameters' added.
#
#     Raises:
#     KeyError: If 'theta_parameters' is not present in the modelling settings.
#     """
#     # Check if 'theta_parameters' is present in models
#     if 'theta' in modelling_settings:
#         # Create 'normalized_parameters' as a dictionary with lists of ones
#         modelling_settings['normalized_parameters'] = {
#             key: [1] * len(value) for key, value in modelling_settings['theta'].items()
#         }
#     else:
#         raise KeyError("The dictionary must contain 'theta_parameters' as a key.")
#
#     return modelling_settings
#
# def save_to_xlsx(sensa):
#     """
#     Save Sobol analysis results to 'sobol_results.xlsx' in the current directory.
#
#     Parameters:
#     - sensa: dict, with structure sensa['analysis'][model][response] = list of time-point dicts
#     - sobol_problem: dict, with sobol_problem[model]['names'] = list of parameter names
#     """
#     if not sensa or not isinstance(sensa, dict) or 'analysis' not in sensa:
#         print("Error: Sobol results not available. No file written.")
#         return
#
#     file_path = os.path.join(os.getcwd(), "sobol_results.xlsx")
#     sobol_analysis = sensa['analysis']
#     sheets_to_write = {}
#
#     for model_name, response_dict in sobol_analysis.items():
#         for response_key, step_data_list in response_dict.items():
#             if not isinstance(step_data_list, list) or not step_data_list:
#                 print(f"Skipping {model_name}-{response_key}: no data.")
#                 continue
#
#             num_params = len(step_data_list[0].get('ST', []))
#             time = list(range(len(step_data_list)))
#
#             try:
#                 names = sobol_problem[model_name]['names']
#             except Exception:
#                 names = [f'Param_{i+1}' for i in range(num_params)]
#
#             df_data = {'Time': time}
#             for i in range(num_params):
#                 col_name = names[i] if i < len(names) else f'Param_{i+1}'
#                 df_data[col_name] = [step.get('ST', [0]*num_params)[i] for step in step_data_list]
#
#             df = pd.DataFrame(df_data)
#
#             sheet_name = f"{model_name}_{response_key}"
#             if len(sheet_name) > 31:
#                 sheet_name = sheet_name[:31]
#
#             sheets_to_write[sheet_name] = df
#
#     if not sheets_to_write:
#         print("No valid sheets found. No Excel file created.")
#         return
#
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         for sheet_name, df in sheets_to_write.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
#
#     print(f"Sobol results saved to: {file_path}")


#log_utils.py

import pickle
import os
import pandas as pd
from scipy import stats
from middoe.iden_utils import Plotting_Results
import importlib
from pathlib import Path


def fun_globalizer(func_name):
    """
    Register a function in the kernel_simulator module namespace for dynamic access.

    This utility enables runtime registration of functions in the kernel_simulator module,
    facilitating dynamic model loading and callback mechanisms. It's primarily used for
    custom model functions that need to be accessible globally during simulation.

    Parameters
    ----------
    func_name : str
        Name under which to register the function in kernel_simulator's namespace.
        The function with this name must exist in the current global scope.

    Returns
    -------
    None
        Function is registered as a side effect.

    Notes
    -----
    **Use Case**:
    When defining custom ODE models in a script, this function allows them to be
    dynamically registered in the kernel_simulator module, making them accessible
    to the simulation engine without modifying the core package code.

    **Registration Mechanism**:
    Uses setattr() to add a lambda wrapper that calls the local function:
        \[
        \text{kernel_simulator.func_name} = \lambda(*args, **kwargs) \rightarrow \text{globals()[func_name]}(*args, **kwargs)
        \]

    **Namespace Management**:
    The function must exist in globals() at the time of registration. This typically
    means it should be defined in the same script before calling fun_globalizer().

    **Warning**:
    This modifies the kernel_simulator module's namespace at runtime. Use with caution
    in production environments. Prefer direct model registration via models['krt'].

    See Also
    --------
    simula : Uses registered functions for model execution.

    Examples
    --------
    >>> # Define a custom model
    >>> def my_custom_model(t, y, tii, tvi, theta, t_vec):
    ...     dydt = [-theta[0] * y[0], theta[0] * y[0]]
    ...     return dydt
    >>>
    >>> # Register it globally
    >>> fun_globalizer('my_custom_model')
    >>>
    >>> # Now accessible in kernel_simulator
    >>> # (used internally by simulation engine)
    """
    kernel_simulator = importlib.import_module('idenpy.kernel_simulator')
    setattr(kernel_simulator, func_name, lambda *args, **kwargs: globals()[func_name](*args, **kwargs))


def read_excel():
    """
    Read experimental data from 'data.xlsx' in the current working directory.

    This is the standard data loading function used throughout the MBDoE package.
    It reads all sheets from an Excel file containing experimental measurements,
    input conditions, and metadata. The file must follow the standardized format
    with specific column naming conventions.

    Returns
    -------
    data : dict[str, pd.DataFrame]
        Dictionary of DataFrames:
            - Keys: Sheet names (typically experiment numbers: '1', '2', '3', ...)
            - Values: DataFrames with standardized columns:
                * 'MES_X:{var}': Measurement times for time-variant outputs
                * 'MES_Y:{var}': Measured values
                * 'MES_E:{var}': Measurement uncertainties (std dev)
                * 'X:all': Full time vector for simulation
                * Input variable columns (e.g., 'T', 'P', 'CA_0')
                * 'CVP:{var}': Control parameterization methods
                * '{var}t', '{var}l': Switching points (if applicable)

    Raises
    ------
    FileNotFoundError
        If 'data.xlsx' does not exist in current working directory.

    Notes
    -----
    **File Location**:
    The function looks for 'data.xlsx' in Path.cwd() (current working directory).
    Ensure you run your script from the project root or navigate to the correct
    directory before calling this function.

    **Data Format**:
    The Excel file should contain multiple sheets, each representing one experiment.
    Within each sheet:
        - **Measurement columns**: 'MES_X:{var}', 'MES_Y:{var}', 'MES_E:{var}'
        - **Time column**: 'X:all' (full simulation time vector)
        - **Input columns**: Time-invariant (e.g., 'P', 'T0') and time-variant (e.g., 'T')
        - **Metadata**: 'CVP:{var}' for control parameterization info
        - **Switching points**: '{var}t' (times) and '{var}l' (levels)

    **Usage Context**:
    Called by:
        - parmest(): To load data for parameter estimation
        - uncert(): To load data for uncertainty analysis
        - validation(): To split data for cross-validation
        - expera(): Writes to this file during in-silico experiments

    **Multiple Sheets**:
    Each sheet represents an independent experiment. Parameter estimation typically
    uses all sheets simultaneously to estimate a single set of parameters. The
    sheet names are arbitrary but should be unique identifiers.

    **Pandas read_excel Options**:
    Uses sheet_name=None to read all sheets at once, returning a dictionary.
    This is more efficient than reading sheets individually.

    See Also
    --------
    expera : Writes data to this Excel file.
    parmest : Primary consumer of this data.
    data_appender : Alternative for in-memory data accumulation.

    Examples
    --------
    >>> # Load experimental data
    >>> data = read_excel()
    >>> print(f"Found {len(data)} experiments")
    >>> print(f"Experiments: {list(data.keys())}")
    >>>
    >>> # Access first experiment
    >>> exp1 = data['1']
    >>> print(f"Columns: {exp1.columns.tolist()}")
    >>>
    >>> # Extract measurements for a specific variable
    >>> y1_times = exp1['MES_X:y1'].dropna().values
    >>> y1_values = exp1['MES_Y:y1'].dropna().values
    >>> y1_errors = exp1['MES_E:y1'].dropna().values
    >>> print(f"Measured {len(y1_values)} points for y1")
    """
    file_path = Path.cwd() / "data.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} not found in the current directory. "
            f"Please create the data file as Excel and add it to your project repository."
        )

    print(f"[INFO] Reading from {file_path.name}")
    data = pd.read_excel(file_path, sheet_name=None)
    return data


def save_rounds(round, result, design_type, round_data, models, iden_opt, obs, system,
                ranking=None, k_optimal_value=None, rCC_values=None, J_k_values=None,
                best_uncert_result=None):
    """
    Save and organize results from a single experimental round for post-processing.

    This function consolidates all results from one round of the MBDoE workflow
    (design → experiment → estimation → analysis) into a structured dictionary.
    It computes reference t-values, saves parameter estimates and uncertainties,
    generates plots, and prepares data for multi-round comparison.

    Parameters
    ----------
    round : int
        Round number (1, 2, 3, ...) indicating experimental campaign progression.
    result : dict[str, dict]
        Uncertainty analysis results for each model/solver containing:
            - 'estimations': Estimated parameter values
            - 'V_matrix': Covariance matrix
            - 'CI': Confidence intervals
            - 't_values': t-statistics for parameter significance
            - 'P': Model probability/weight
            - Plus all metrics from uncert()
    design_type : str
        Design category: 'classic' (preliminary) or 'DOE' (MBDoE-optimized).
    round_data : dict
        Cumulative storage dictionary for all rounds. Updated in-place.
    models : dict
        Model definitions with parameter masks, names, and settings.
    iden_opt : dict
        Identification options including plotting flags:
            - 'c_plt': bool — Generate confidence plots
            - 'f_plt': bool — Generate fit plots
            - 'plt_s': str — Plotting style/folder
    obs : int
        Total number of observations (measurements) across all experiments.
    system : dict
        System configuration with variable definitions.
    ranking : list[str], optional
        Parameter ranking from estimability analysis (most to least estimable).
    k_optimal_value : int, optional
        Optimal number of parameters to estimate based on rCC analysis.
    rCC_values : dict[str, list[float]], optional
        Revised Critical Ratio values for each model: {solver: [rCC_1, rCC_2, ...]}.
    J_k_values : dict[str, list[float]], optional
        Objective function values for each k: {solver: [J_1, J_2, ...]}.
    best_uncert_result : dict, optional
        If estimability analysis was performed, the uncertainty results for the
        optimal parameter subset. Overrides 'result' if provided.

    Returns
    -------
    trv : dict[str, float]
        Reference t-values for 95% confidence level:
            \[
            t_{ref} = t_{0.975, \nu} \quad \text{where } \nu = n_{obs} - n_{params}
            \]
        Keys: solver names. Used to assess parameter significance.

    Side Effects
    ------------
    - Updates round_data dict with complete round information
    - Generates confidence plots (if iden_opt['c_plt']=True)
    - Generates fit plots (if iden_opt['f_plt']=True)
    - Prints summary statistics (t-values, P-values) to console

    Notes
    -----
    **Round Data Structure**:
    round_data['Round i'] contains:
        - 'ranking': Parameter estimability ranking
        - 'k_optimal_value': Optimal number of parameters
        - 'rCC_values': Corrected Critical Ratios
        - 'J_k_values': Objective function values
        - 'design_type': 'classic' or 'DOE'
        - 'mutation': Parameter activity masks
        - 'original_positions': Parameter reordering info
        - 'trv': Reference t-values
        - 'scaled_params': Estimated parameter values
        - 'result': Full uncertainty analysis results
        - 'iden_opt': Copy of identification options
        - 'models': Copy of model definitions
        - 'system': Copy of system configuration
        - 'est_EA': Estimability analysis results

    **Reference t-Values**:
    Computed from Student's t-distribution:
        \[
        t_{ref} = t_{1 - \alpha/2, \nu} \quad \text{with } \alpha = 0.05, \nu = n_{obs} - n_{params}
        \]
    Used to assess parameter significance:
        - If \( |t_{param}| > t_{ref} \): Parameter is significant
        - If \( |t_{param}| < t_{ref} \): Parameter is non-significant (poorly determined)

    **Plotting**:
    Two types of plots are generated (if enabled):
        1. **Confidence plots**: 2D ellipsoids showing parameter correlations
        2. **Fit plots**: Model predictions vs experimental data

    Plots are saved to folders named by round and solver.

    **Estimability Analysis Integration**:
    If best_uncert_result is provided, it indicates that estimability analysis
    was performed to select a subset of parameters. This result is used instead
    of the full uncertainty analysis.

    **Mutation Tracking**:
    Parameter activity masks (mutation) track which parameters are estimated
    (True) vs fixed (False) in each round. This is crucial for sequential
    parameter estimation strategies.

    See Also
    --------
    uncert : Generates the 'result' dict.
    run_postprocessing : Uses round_data for multi-round analysis.
    Plotting_Results : Generates confidence and fit plots.

    Examples
    --------
    >>> # After completing estimation and uncertainty analysis
    >>> round_data = {}
    >>> trv = save_rounds(
    ...     round=1,
    ...     result=uncert_results,
    ...     design_type='classic',
    ...     round_data=round_data,
    ...     models=models,
    ...     iden_opt={'c_plt': True, 'f_plt': True, 'plt_s': './plots'},
    ...     obs=50,
    ...     system=system
    ... )
    >>>
    >>> print(f"Round 1 saved. Reference t-value: {trv['M1']:.3f}")
    >>> print(f"Stored data: {list(round_data['Round 1'].keys())}")
    >>>
    >>> # Check parameter significance
    >>> for i, t_val in enumerate(uncert_results['M1']['t_values']):
    ...     significant = "✓" if abs(t_val) > trv['M1'] else "✗"
    ...     print(f"  θ{i+1}: t={t_val:.2f} {significant}")
    """
    data = read_excel()
    if best_uncert_result:
        result = best_uncert_result['results']

    round_key = f'Round {round}'
    dof = {solver: obs - len(result[solver]['estimations']) for solver in models['can_m']}
    trv = {solver: stats.t.ppf(1 - (1 - 0.95) / 2, dof[solver]) for solver in models['can_m']}
    scaled_params = {solver: result[solver]['estimations'] for solver in models['can_m']}

    round_data[round_key] = {
        'ranking': ranking,
        'k_optimal_value': k_optimal_value,
        'rCC_values': rCC_values,
        'J_k_values': J_k_values,
        'design_type': design_type,
        'mutation': {},
        'original_positions': {},
        'trv': trv,
        'scaled_params': scaled_params,
        'result': result,
        'iden_opt': iden_opt,
        'models': models,
        'system': system,
        'est_EA': best_uncert_result
    }

    # Save mutation information
    for solver, mutation in models['mutation'].items():
        round_data[round_key]['mutation'][solver] = mutation

    # Save original positions information
    for solver in models['can_m']:
        if solver in models.get('original_positions', {}):
            round_data[round_key]['original_positions'][solver] = models['original_positions'][solver]
        else:
            round_data[round_key]['original_positions'][solver] = []

    solver_cov_matrices = {solver: result[solver]['V_matrix'] for solver in models['can_m']}
    solver_confidence_intervals = {solver: result[solver]['CI'] for solver in models['can_m']}

    # Generate plots
    plotting1 = Plotting_Results(models, iden_opt['plt_s'], round)
    if iden_opt['c_plt'] == True:
        plotting1.conf_plot(scaled_params, solver_cov_matrices, solver_confidence_intervals)
    if iden_opt['f_plt'] == True:
        plotting1.fit_plot(data, result, system)

    # Print summary
    for solver in models['can_m']:
        print(f'reference t value for model {solver} and round {round}: {trv[solver]}')
        print(f'estimated t values for model {solver} and round {round}: {result[solver]["t_values"]}')
        print(f'P-value for model {solver} and round {round}: {result[solver]["P"]}')
    print()

    return trv


def save_to_jac(results, purpose):
    """
    Serialize and save results to a binary .jac file using pickle.

    This function provides persistent storage for estimation or sensitivity analysis
    results, enabling later retrieval without re-running expensive computations.
    The .jac extension is a custom convention for "Job Archive" files.

    Parameters
    ----------
    results : dict
        Results dictionary to serialize. Structure depends on purpose:
            - For 'iden': round_data dict with all estimation rounds
            - For 'sensa': Sobol sensitivity analysis results
    purpose : str
        Result type, determines filename:
            - 'iden': Saves to 'iden_results.jac' (identification results)
            - 'sensa': Saves to 'sensa_results.jac' (sensitivity results)

    Returns
    -------
    None
        File is written to disk as a side effect.

    Raises
    ------
    ValueError
        If purpose is neither 'iden' nor 'sensa'.
    Exception
        If file writing fails (permissions, disk space, etc.).

    Notes
    -----
    **Pickle Protocol**:
    Uses pickle.HIGHEST_PROTOCOL for maximum compatibility with future Python
    versions while maintaining backward compatibility.

    **File Location**:
    Files are saved in Path.cwd() (current working directory). Ensure you have
    write permissions in this directory.

    **Security Warning**:
    Pickle files can execute arbitrary code during deserialization. Only load
    .jac files from trusted sources. For untrusted data, consider JSON or HDF5.

    **File Size**:
    Results can be large (MB-GB) if they contain many rounds with extensive
    covariance matrices, LSA matrices, and simulation trajectories. Monitor
    disk space usage.

    **Use Cases**:
        - **Checkpoint**: Save after each round for crash recovery
        - **Archive**: Preserve results for reproducibility
        - **Sharing**: Distribute results to collaborators
        - **Post-processing**: Load results for plotting without re-estimation

    See Also
    --------
    load_from_jac : Companion function to load saved results.
    save_rounds : Organizes data before saving.

    Examples
    --------
    >>> # After completing all estimation rounds
    >>> save_to_jac(round_data, purpose='iden')
    # [INFO] Results saved to: .../iden_results.jac
    >>>
    >>> # After sensitivity analysis
    >>> save_to_jac(sobol_results, purpose='sensa')
    # [INFO] Results saved to: .../sensa_results.jac
    >>>
    >>> # Invalid purpose
    >>> save_to_jac(results, purpose='invalid')
    # ValueError: Purpose must be 'iden' or 'sensa'
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
    Load previously saved identification and sensitivity analysis results.

    This function attempts to deserialize .jac files from the current directory,
    providing easy access to archived results without re-running expensive
    computations. It handles missing files gracefully and reports status for
    each expected file.

    Returns
    -------
    loaded_data : dict or None
        Dictionary with keys 'iden' and 'sensa':
            - loaded_data['iden']: Identification results (round_data) or None
            - loaded_data['sensa']: Sensitivity results (sobol_results) or None
        Returns None if neither file is found.

    Notes
    -----
    **Expected Files**:
        - 'iden_results.jac': Identification/estimation results
        - 'sensa_results.jac': Sensitivity analysis results

    **Return Behavior**:
        - If both files exist: Returns dict with both keys populated
        - If one file exists: Returns dict with one key populated, other is None
        - If neither exists: Returns None

    **Error Handling**:
    If a file exists but cannot be loaded (corrupt, version mismatch, etc.),
    the function prints an error message and sets that key to None, but continues
    loading other files.

    **Usage Patterns**:
        1. **Resume workflow**: Load previous rounds and continue
        2. **Post-processing only**: Load results for plotting/analysis
        3. **Comparison**: Load multiple result sets for comparison

    **Pickle Security**:
    As with save_to_jac(), only load .jac files from trusted sources.

    See Also
    --------
    save_to_jac : Companion function to save results.
    run_postprocessing : Often called after loading results.

    Examples
    --------
    >>> # Load previous results
    >>> loaded = load_from_jac()
    >>> if loaded is not None:
    ...     if loaded['iden'] is not None:
    ...         print(f"Loaded {len(loaded['iden'])} estimation rounds")
    ...         round_data = loaded['iden']
    ...     if loaded['sensa'] is not None:
    ...         print("Loaded sensitivity analysis results")
    ...         sobol_results = loaded['sensa']
    >>> else:
    ...     print("No previous results found, starting fresh")
    >>>
    >>> # Continue from last round
    >>> if loaded and loaded['iden']:
    ...     last_round = max([int(k.split()[1]) for k in loaded['iden'].keys()])
    ...     print(f"Continuing from round {last_round}")
    ...     # Prepare for round last_round + 1
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
    Accumulate experimental data in memory during sequential experiment generation.

    This function provides an alternative to Excel file accumulation for in-silico
    experiments. It's particularly useful for programmatic workflows where data
    is generated and consumed entirely in Python without intermediate file I/O.

    Parameters
    ----------
    df_combined : pd.DataFrame
        DataFrame for current experiment with standardized structure (from expera()).
    experiment_number : int or str
        Unique identifier for this experiment (converted to string internally).
    data_storage : dict[str, pd.DataFrame]
        Accumulator dictionary. Updated in-place.

    Returns
    -------
    data_storage : dict[str, pd.DataFrame]
        Updated storage dictionary with appended experiment.

    Notes
    -----
    **Usage Context**:
    Alternative to Excel-based workflow:
        - **Excel workflow**: expera() → data.xlsx → read_excel() → parmest()
        - **Memory workflow**: expera() → data_appender() → parmest(data=data_storage)

    **When to Use**:
        - Automated batch experiments
        - High-throughput screening
        - Avoiding file I/O overhead
        - Testing and validation

    **DataFrame Structure**:
    The df_combined DataFrame should match the format expected by read_excel(),
    with the same column naming conventions.

    **Memory Considerations**:
    Storing many experiments in memory can consume significant RAM. For large
    campaigns (>100 experiments), consider periodic saving to disk.

    See Also
    --------
    expera : Generates df_combined for each experiment.
    read_excel : Excel-based alternative.
    parmest : Accepts data from either source.

    Examples
    --------
    >>> # Initialize storage
    >>> data_storage = {}
    >>>
    >>> # Generate multiple experiments
    >>> for i in range(1, 6):
    ...     _, df = expera(system, models, insilicos, design_decisions, expr=i)
    ...     data_storage = data_appender(df, i, data_storage)
    >>>
    >>> print(f"Accumulated {len(data_storage)} experiments")
    >>> # {'1': DataFrame, '2': DataFrame, ..., '5': DataFrame}
    >>>
    >>> # Use directly for estimation
    >>> results = parmest(system, models, iden_opt, data=data_storage)
    """
    experiment_number = str(experiment_number)
    data_storage[experiment_number] = df_combined
    return data_storage


def add_norm_par(modelling_settings):
    """
    Initialize normalized parameter vectors for all models.

    This utility creates 'normalized_parameters' entries (all ones) from the
    'theta' parameter scaling factors. It's used during setup to establish the
    normalization convention used throughout the package.

    Parameters
    ----------
    modelling_settings : dict
        Model definitions with:
            - 'theta' : dict[str, list[float]]
                Parameter scaling factors for each model.

    Returns
    -------
    modelling_settings : dict
        Updated dictionary with added:
            - 'normalized_parameters' : dict[str, list[float]]
                Normalized parameter vectors (all ones) for each model.

    Raises
    ------
    KeyError
        If 'theta' key is not present in modelling_settings.

    Notes
    -----
    **Normalization Convention**:
    Throughout the package, parameters are represented as:
        \[
        \theta_{physical} = \theta_{normalized} \cdot \theta_c
        \]
    where \( \theta_{normalized} = [1, 1, ..., 1] \) represents the baseline
    (true or initial) parameter values.

    **Why All Ones?**:
    Using ones as the baseline simplifies:
        - Parameter estimation: Optimize \( \theta_{normalized} \) around 1.0
        - Bounds specification: Bounds become multiplicative factors (e.g., [0.5, 2.0])
        - Interpretation: Values >1 indicate parameters larger than baseline

    **Usage Context**:
    Called during model setup, typically early in the workflow before any
    estimation or design routines.

    See Also
    --------
    _construct_par : Uses normalized parameters during simulation.
    parmest : Estimates normalized parameters during optimization.

    Examples
    --------
    >>> models = {
    ...     'theta': {
    ...         'M1': [1.5e5, 2.3, 0.8],
    ...         'M2': [2.0e5, 1.8, 1.2]
    ...     },
    ...     'can_m': ['M1', 'M2']
    ... }
    >>> models = add_norm_par(models)
    >>> print(models['normalized_parameters'])
    # {'M1': [1, 1, 1], 'M2': [1, 1, 1]}
    """
    if 'theta' in modelling_settings:
        modelling_settings['normalized_parameters'] = {
            key: [1] * len(value) for key, value in modelling_settings['theta'].items()
        }
    else:
        raise KeyError("The dictionary must contain 'theta_parameters' as a key.")

    return modelling_settings


def save_to_xlsx(sensa, sobol_problem):
    """
    Export Sobol sensitivity analysis results to Excel for visualization and reporting.

    This function converts nested Sobol analysis results (time-varying sensitivity
    indices for multiple models and responses) into a well-organized Excel workbook
    with separate sheets for each model-response combination.

    Parameters
    ----------
    sensa : dict
        Sensitivity analysis results with structure:
            sensa['analysis'][model_name][response_name] = [
                {'ST': [s1, s2, ...], 'S1': [...], ...},  # time point 1
                {'ST': [s1, s2, ...], 'S1': [...], ...},  # time point 2
                ...
            ]
    sobol_problem : dict
        Sobol problem definition with parameter names:
            sobol_problem[model_name]['names'] = ['k1', 'k2', 'Ea', ...]

    Returns
    -------
    None
        Excel file is written to disk as a side effect.

    Side Effects
    ------------
    Creates 'sobol_results.xlsx' in current working directory.

    Notes
    -----
    **Excel Structure**:
    Each sheet contains:
        - Column 1: Time (or time point index)
        - Columns 2+: Total Sobol indices (ST) for each parameter

    Sheet names follow the pattern: '{model}_{response}'
    (truncated to 31 chars for Excel compatibility).

    **Total Sobol Index (ST)**:
    The function exports ST (total-order indices) rather than S1 (first-order)
    because ST captures both main effects and all interactions, providing the
    most comprehensive sensitivity information.

    **Time-Varying Sensitivity**:
    Each row represents one time point. This captures how parameter importance
    evolves during a dynamic experiment:
        - Early times: Initial condition parameters dominate
        - Middle times: Kinetic parameters dominate
        - Late times: Equilibrium parameters dominate

    **Missing Data Handling**:
        - Skips invalid or empty data
        - Uses default parameter names ('Param_1', 'Param_2', ...) if names unavailable
        - Handles mismatched data gracefully

    **Visualization**:
    The exported Excel file can be:
        - Plotted directly in Excel
        - Imported into plotting software
        - Used for publication-quality figures

    See Also
    --------
    plot_sobol_results : Visualization function from iden_utils.
    SALib.analyze.sobol : Sobol analysis computation.

    Examples
    --------
    >>> # After running Sobol analysis
    >>> save_to_xlsx(sensa_results, sobol_problem)
    # Sobol results saved to: .../sobol_results.xlsx
    >>>
    >>> # Result: Excel file with sheets like:
    >>> # - M1_CA (Concentration A sensitivity for model 1)
    >>> # - M1_CB (Concentration B sensitivity for model 1)
    >>> # - M2_T (Temperature sensitivity for model 2)
    >>> # Each with columns: Time, k1, k2, Ea, ...
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