# # krnl_expera.py
#
# from middoe.krnl_simula import simula
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import os
#
# def expera(system, models, insilicos, design_decisions, expr, swps=None):
#     """
#     Run an in-silico experiment, produce simulation results, and prepare them for reporting.
#
#     This function:
#     1. Sets up paths and simulation parameters.
#     2. Constructs scaled and unscaled variables, parameters, and switching points as needed.
#     3. Calls the in-silico simulator to generate results.
#     4. Returns the path to an Excel file and the resulting simulation DataFrame.
#
#     Parameters
#     ----------
#     system : dict
#         Structure of the model, including variables and their properties.
#         It contains keys like 'tv_iphi', 'tv_ophi', 'ti_iphi', 'ti_ophi',
#         and 'independant_variables' each with 'var' and 'attributes' sub-dicts.
#
#     models : dict
#         User-defined settings related to the modelling process, including 'theta_parameters'.
#
#     insilicos : dict
#         User-defined settings for the simulator, including:
#         - 'insilico_model': str, name of the model function to be simulated.
#         - 'classic-des': dict, containing default or "classic" design values.
#         - 'nd2': int, number of points for time discretization.
#
#     design_decisions : dict
#         Design decisions for the experiment. May contain keys like 'St', 'tii', 'tvi', 't_values'.
#
#     expr : int
#         Index for the current experiment.
#
#     swps : dict, optional
#         Switching points for the time-variant variables, unscaled. If None, empty dicts are used.
#
#     Returns
#     -------
#     tuple
#         (excel_path, df_combined)
#         excel_path : str
#             The path to the Excel file where results are to be stored.
#         df_combined : pandas.DataFrame
#             The combined results of the in-silico experiment.
#     """
#     # Extract scaling constant for independent variables
#     tsc = system['t_s'][1]
#
#
#     # Construct piecewise functions for initial and DOE cases
#     cvp_initial = {
#         var: 'no_CVP'
#         for var in system['tvi'].keys()
#     }
#     cvp_doe = {
#         var: system['tvi'][var]['cvp']
#         for var in system['tvi'].keys()
#     }
#
#     # Retrieve the model name and related simulation parameters
#     model_name = insilicos.get('tr_m')
#     errortype = insilicos.get('errt', 'abs')
#     classic_des = insilicos['prels']
#     # theta_parameters = models['theta']
#     theta_parameters = insilicos['theta']
#
#     # Standard deviations for measured tv_ophi variables
#     std_dev = {
#         var: system['tvo'][var]['unc']
#         for var in system['tvo'].keys()
#         if system['tvo'][var].get('meas', True)
#     }
#
#     # If no design decisions are provided, run the "initial" case
#     if not design_decisions:
#         # Time values for each measured variable
#         # Step 1: Create tlin
#         dt_real = system['t_r']
#         nodes = int(round(tsc / dt_real)) + 1
#         tlin = np.linspace(0, 1, nodes)
#
#         # Step 2: Generate t_values, snapping each to the closest value in tlin
#         t_values = {
#             var: [float(tlin[np.argmin(np.abs(tlin - t))]) for t in
#                   np.linspace(0, 1, system['tvo'][var]['sp'])]
#             for var in system['tvo']
#             if system['tvo'][var].get('meas', True)
#         }
#
#         # Flatten all time points
#         t_values_flat = [tp for times in t_values.values() for tp in times]
#
#         # Create unique accumulated time points for simulation
#         t_acc_unique = np.unique(np.concatenate((tlin, t_values_flat))).tolist()
#
#         # Construct variables and parameters for the "initial" case
#         phi, phit, phisc, phitsc = _construct_var(system, classic_des, expr)
#         theta, thetac = _construct_par(model_name, theta_parameters)
#
#         if swps is None:
#             swps, swpsu = {}, {}
#
#         case = 'initial'
#         df_combined = _inssimulator(
#             t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac,
#             cvp_initial, std_dev, t_acc_unique, case, model_name,
#             system, models, errortype
#         )
#
#     else:
#         # When design decisions are provided, run the "doe" case
#         St = design_decisions['St']
#         t_values = {var: np.array(St[var]) / tsc for var in St}
#
#         # Accumulated time points for DOE
#         t_acc_unique = np.array(design_decisions['t_values'])
#         case = 'doe'
#
#         # Construct scaling and parameters
#         _, _, phisc, phitsc = _construct_var(system, classic_des, expr)
#         theta, thetac = _construct_par(model_name, theta_parameters)
#
#         phi = design_decisions['tii']
#         phit = design_decisions['tvi']
#         swpsu = swps if swps is not None else {}
#
#         # Scale tii and tvi
#         phi_scaled = {
#             var: np.array(phi[var]) / phisc[var]
#             for var in phi if var in system['tii'].keys()
#         }
#         phit_scaled = {
#             var: np.array(phit[var])
#             for var in phit if var in system['tvi'].keys()
#         }
#
#         # Scale swps
#         swps_scaled = {}
#         if swpsu:
#             for var in swpsu:
#                 if var.endswith('l'):
#                     # Length-based scaling
#                     var_base = var.rstrip('l')
#                     swps_scaled[var] = np.array(swpsu[var]) / phitsc[var_base]
#                 elif var.endswith('t'):
#                     # Time-based scaling
#                     swps_scaled[var] = np.array(swpsu[var]) / tsc
#                 else:
#                     raise ValueError(f"Unrecognized variable format for {var} in swps.")
#
#         df_combined = _inssimulator(
#             t_values, swps_scaled, swpsu, phi_scaled, phisc, phit_scaled, phitsc, tsc, theta, thetac,
#             cvp_doe, std_dev, t_acc_unique, case, model_name, system, models, errortype)
#
#     # Define Excel path for in-silico data
#     excel_path = Path.cwd() / 'data.xlsx'
#
#     experiment_number = str(expr)  # Ensure experiment_number is a string
#     # Check if the file already exists
#     if os.path.isfile(excel_path):
#         # Open the file in append mode if it exists
#         with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
#             existing_sheets = writer.book.sheetnames
#             if experiment_number in existing_sheets:
#                 # Append to the existing sheet
#                 df_combined.to_excel(writer, sheet_name=experiment_number, index=False)
#             else:
#                 # Create a new sheet
#                 df_combined.to_excel(writer, sheet_name=experiment_number, index=False)
#     else:
#         # If the file does not exist, create it and write the data
#         with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
#             df_combined.to_excel(writer, sheet_name=experiment_number, index=False)
#
#     # Return Excel path and DataFrame, with summary print
#     print(f"\n[✓] In-silico data saved to: {excel_path}")
#     print(f"[INFO] Model used         : {model_name}")
#     print(f"[INFO] Design case        : {'classic/preliminary' if case == 'initial' else 'MBDoE'}")
#     print(f"[INFO] Responses simulated:")
#
#     for response in system.get('tvo', {}):
#         meas = system['tvo'][response].get('meas', True)
#         unc = system['tvo'][response].get('unc', 'N/A')
#         status = "measurable" if meas else "non-measurable"
#         print(f"   - {response:<10} | {status:<15} | std.dev = {unc}")
#
#     return excel_path, df_combined
#
#     # return df_combined
#
#
# def _construct_var(system, classic_des, j):
#     """
#     Construct and scale variable dictionaries for the simulation.
#
#     Parameters
#     ----------
#     system : dict
#         Dictionary describing the model structure.
#     classic_des : dict
#         Dictionary containing "classic" design values for variables.
#     j : int
#         Index of the current experiment.
#
#     Returns
#     -------
#     tuple
#         (tii, tvi, phisc, phitsc):
#         - tii : dict
#             Time-invariant variable values scaled by their maxima.
#         - tvi : dict
#             Time-variant variable values scaled by their maxima.
#         - phisc : dict
#             Scaling factors (max values) for time-invariant variables.
#         - phitsc : dict
#             Scaling factors (max values) for time-variant variables.
#     """
#     phit = {}
#     phisc = {}
#     phitsc = {}
#     phi = {}
#
#     # Time-variant input variables (tv_iphi)
#     for var in system['tvi'].keys():
#         max_val = system['tvi'][var]['max']
#         classic_value = classic_des.get(str(j), {}).get(var, max_val)
#         phitsc[var] = max_val
#         phit[var] = classic_value / max_val
#
#     # Time-invariant input variables (ti_iphi)
#     for var in system['tii'].keys():
#         max_val = system['tii'][var]['max']
#         classic_value = classic_des.get(str(j), {}).get(var, max_val)
#         phisc[var] = max_val
#         phi[var] = classic_value / max_val
#
#     return phi, phit, phisc, phitsc
#
#
# def _construct_par(model_name, theta_parameters):
#     """
#     Construct parameter arrays for the simulation, given the model name and theta parameters.
#
#     Parameters
#     ----------
#     model_name : str
#         The name of the model.
#     theta_parameters : dict
#         Dictionary mapping model names to theta scaling parameters.
#
#     Returns
#     -------
#     tuple
#         (theta, thetac):
#         - theta : list
#             Normalized parameter values (e.g., [1, 1, ...]).
#         - thetac : list
#             Actual parameter scaling values for the given model.
#
#     Raises
#     ------
#     ValueError
#         If parameters cannot be constructed or accessed.
#     """
#     try:
#         # thetac = theta_parameters.get(model_name, [])
#         thetac = theta_parameters
#         theta = [1] * len(thetac)
#         return theta, thetac
#     except Exception as e:
#         raise ValueError(f"Unable to adjust theta or thetac for model {model_name}: {e}")
#
#
# def _inssimulator(t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac, cvp, std_dev,
#                   t_valuesc, case, model_name, system, models, errortype):
#     """
#     Conduct the in-silico experiments by setting up the simulation environment,
#     running the simulations, adding noise, and saving the results.
#
#     Returns one combined DataFrame with sections:
#     - var_t, var_noisy for each tv_ophi variable (each with its own length).
#     - t_global, tii(s), tvi(s)/phit_interp(s), ti_ophi (aligned with t_valuesc).
#     - piecewise_func in a separate section.
#     - swpsu in another separate section.
#
#     All in one DataFrame, separated by blank columns.
#
#     Parameters
#     ----------
#     t_values : dict
#         Dictionary of time values keyed by dependent variable name,
#         e.g. {'y1': [0, 0.25, 0.5, 1.0], 'y2': [0, 0.5, 1.0]}.
#     swps : dict
#         Scaled switching points for the time-variant variables.
#     swpsu : dict
#         Unscaled switching points for the time-variant variables.
#     phi, phisc : dict
#         Dictionaries of time-invariant variables and their scaling factors.
#     phit, phitsc : dict
#         Dictionaries of time-variant variables and their scaling factors.
#     tsc : float
#         Scaling factor for time.
#     theta, thetac : list
#         Parameter vectors.
#     cvp : dict
#         Dictionary of piecewise functions defining variable trajectories.
#     std_dev : dict
#         Dictionary of standard deviations keyed by variable name.
#     t_valuesc : list
#         Global set of time points for simulation.
#     case : str
#         Indicates the type of run ('initial', 'doe', etc.).
#     model_name : str
#         The name of the model being simulated.
#     system : dict
#         Defines the structure of the model, including variable attributes.
#     models : dict
#         Additional modelling settings.
#     errortype : str
#         Type of error to be applied ('abs' for absolute, 'rel' for relative).
#
#     Returns
#     -------
#     pd.DataFrame
#         A combined DataFrame with simulation results, noisy measurements, piecewise function info, and switching points.
#
#     Raises
#     ------
#     ValueError
#         If the case is unsupported or if the number of independent variables does not match the dependent variables.
#     KeyError
#         If a dependent variable is missing in the simulation results.
#     """
#
#     # Run the model to get tv_ophi, ti_ophi, phit_interp
#     if case == 'initial':
#         tv_ophi, ti_ophi, phit_interp = simula(
#             t_valuesc, {}, phi, phisc, phitsc, tsc, theta, thetac,
#             cvp, phit, model_name, system, models
#         )
#     elif case == 'doe':
#         tv_ophi, ti_ophi, phit_interp = simula(
#             t_valuesc, swps, phi, phisc, phitsc, tsc, theta, thetac,
#             cvp, phit, model_name, system, models
#         )
#     else:
#         raise ValueError(f"Unsupported case: {case}")
#
#     # Identify measured variables from system
#     measured_vars = [
#         var for var in system['tvo'].keys()
#         if system['tvo'][var].get('measured', True)
#     ]
#
#     # Prepare lists of independent and dependent variables
#     # Assuming that there is a one-to-one mapping defined externally,
#     # we match the number of measured vars with the number of independent vars
#     indep_vars = list(range(len(measured_vars)))
#     dep_vars = measured_vars
#
#     if len(indep_vars) != len(dep_vars):
#         raise ValueError("The number of independent variables must match the number of dependent variables.")
#
#     indep_to_dep_mapping = dict(zip([f"X{i+1}" for i in indep_vars], dep_vars))
#
#     # Check that all dependent variables exist in tv_ophi
#     for indep_var, dep_var in indep_to_dep_mapping.items():
#         if dep_var not in tv_ophi:
#             raise KeyError(
#                 f"Dependent variable '{dep_var}' is missing in tv_ophi. Available keys: {list(tv_ophi.keys())}"
#             )
#
#     # 1. Per-variable DataFrames for tv_ophi
#     var_dfs = []
#     for indep_var, dep_var in indep_to_dep_mapping.items():
#         var_times = [t * tsc for t in t_values[dep_var]]  # Times for the dependent variable
#         var_results = []
#
#         for t in var_times:
#             # Find the closest index in the scaled time vector
#             closest_idx = np.argmin(np.abs(np.array(t_valuesc) * tsc - t))
#             original_val = tv_ophi[dep_var][closest_idx]
#
#             # Compute noise based on selected error type
#             if errortype == 'abs':
#                 sigma = std_dev[dep_var]  # constant std deviation
#             elif errortype == 'rel':
#                 sigma = std_dev[dep_var] * abs(original_val)  # signal-dependent
#             else:
#                 raise ValueError(f"Unsupported errortype: {errortype}")
#
#             noisy_val = np.random.normal(loc=original_val, scale=sigma)
#             noisy_val = max(noisy_val, 0.0)
#             var_results.append([t, noisy_val, sigma])
#
#         df_var = pd.DataFrame(var_results, columns=[f"MES_X:{dep_var}", f"MES_Y:{dep_var}", f"MES_E:{dep_var}"])
#         var_dfs.append(df_var)
#
#     # 2. Global DataFrame (t_global, tii, tvi/phit_interp, ti_ophi)
#     phi_order = list(phi.keys())
#     ti_ophi_order = list(ti_ophi.keys())
#     if case == 'classic':
#         phit_order = list(phit.keys())
#     else:
#         phit_order = list(phit_interp.keys())
#
#     # Noise added once for ti_ophi
#     ti_ophi_noisy_once = {
#         var: np.random.normal(val[0], abs(std_dev.get(var, 0) * val[0]))
#         for var, val in ti_ophi.items()
#     }
#
#     global_results = []
#     for i, t_global in enumerate(t_valuesc):
#         row = [t_global * tsc]
#
#         # tii
#         phi_vals = [phi[v] * phisc[v] for v in phi_order]
#         row.extend(phi_vals)
#
#         # tvi or phit_interp
#         if case == 'classic':
#             phit_vals = [phit[v] * phitsc[v] for v in phit_order]
#         else:
#             phit_vals = [phit_interp[v][i] for v in phit_order]
#         row.extend(phit_vals)
#
#         # ti_ophi noisy (constant)
#         ti_ophi_vals = [ti_ophi_noisy_once[v] for v in ti_ophi_order]
#         row.extend(ti_ophi_vals)
#
#         global_results.append(row)
#
#     global_columns = (['X:all']
#                       + phi_order
#                       + phit_order
#                       + ti_ophi_order)
#     df_global = pd.DataFrame(global_results, columns=global_columns)
#
#     # Combine var_dfs and df_global horizontally
#     all_dfs = var_dfs + [df_global]
#     df_main = pd.concat(all_dfs, axis=1)
#
#     # 3. piecewise_func DataFrame
#     df_piecewise = pd.DataFrame([{
#         f"CVP:{var}": method for var, method in cvp.items()
#     }])
#
#     # Combine all into one DataFrame
#     df_combined = pd.concat([df_main, df_piecewise], axis=1)
#
#     # 4. swpsu DataFrame
#     if case == 'doe':
#         swpsu_order = list(swpsu.keys())
#         swps_data = {key: swpsu[key] for key in swpsu_order}
#         df_swps = pd.DataFrame(swps_data)
#
#         # Insert blank columns to separate sections
#         max_length = max(len(df_main), len(df_piecewise), len(df_swps))
#         df_blank_1 = pd.DataFrame({' ': [None] * max_length})
#         df_blank_2 = pd.DataFrame({'  ': [None] * max_length})
#
#         # Combine all into one DataFrame
#         df_combined = pd.concat([df_main, df_blank_1, df_piecewise, df_blank_2, df_swps], axis=1)
#
#     return df_combined


# krnl_expera.py

from middoe.krnl_simula import simula
import numpy as np
import pandas as pd
from pathlib import Path
import os


def expera(system, models, insilicos, design_decisions, expr, swps=None):
    r"""
    Execute in-silico experiment and generate synthetic measurement data with noise.

    This function orchestrates the complete in-silico experimentation workflow: setting up
    simulation parameters, running the forward model, adding realistic measurement noise,
    and saving results to Excel. It supports both preliminary (classic) designs and optimized
    MBDoE designs. The function handles time-variant and time-invariant inputs/outputs,
    control variable parameterizations, and switching profiles.

    Parameters
    ----------
    system : dict
        System configuration defining model structure:
            - 't_s' : tuple[float, float]
                Time span (start, end).
            - 't_r' : float
                Time resolution for discretization.
            - 'tvi' : dict
                Time-variant input definitions:
                    * 'max' : float — upper bound
                    * 'cvp' : str — control variable parameterization method
            - 'tii' : dict
                Time-invariant input definitions.
            - 'tvo' : dict
                Time-variant output definitions:
                    * 'meas' : bool — measurement flag
                    * 'unc' : float — measurement uncertainty (std dev)
                    * 'sp' : int — number of sampling points
            - 'tio' : dict
                Time-invariant output definitions.
    models : dict
        Model definitions (currently unused but passed for extensibility).
    insilicos : dict
        In-silico experiment configuration:
            - 'tr_m' : str
                True model name (ground truth for simulation).
            - 'theta' : list[float]
                True parameter values (ground truth).
            - 'prels' : dict
                Preliminary/classic design values.
            - 'errt' : str, optional
                Error type: 'abs' (absolute) or 'rel' (relative, default: 'abs').
    design_decisions : dict or None
        MBDoE design decisions. If None, runs preliminary design. If provided:
            - 'St' : dict[str, np.ndarray]
                Sampling times for each output variable (unscaled).
            - 't_values' : np.ndarray
                Global time vector for simulation.
            - 'tii' : dict[str, float]
                Time-invariant input values.
            - 'tvi' : dict[str, np.ndarray]
                Time-variant input profiles.
    expr : int
        Experiment number (used for Excel sheet naming).
    swps : dict, optional
        Switching points for time-variant inputs (unscaled):
            - '{var}t' : np.ndarray — switching times
            - '{var}l' : np.ndarray — switching levels

    Returns
    -------
    excel_path : Path
        Path to the Excel file containing experiment data ('data.xlsx').
    df_combined : pd.DataFrame
        Combined DataFrame with all simulation results and metadata:
            - Per-variable sections: 'MES_X:{var}', 'MES_Y:{var}', 'MES_E:{var}'
            - Global time vector: 'X:all'
            - Input values: time-invariant and time-variant
            - Output values: time-invariant (noisy)
            - CVP metadata: 'CVP:{var}' (control parameterization methods)
            - Switching points: '{var}t', '{var}l' (if applicable)

    Notes
    -----
    **Experimental Workflow**:
        1. **Setup**: Extract scaling factors, construct variable/parameter dictionaries
        2. **Time Discretization**: Create global time vector based on sampling requirements
        3. **Simulation**: Call forward model (simula) with specified inputs
        4. **Noise Addition**: Add measurement noise according to error type
        5. **Data Organization**: Structure results in standardized DataFrame format
        6. **Export**: Save to Excel with experiment number as sheet name

    **Preliminary vs MBDoE Designs**:
        - **Preliminary** (design_decisions=None):
            * Uses classic_des values from insilicos['prels']
            * Uniform sampling in time for each output
            * Simple input profiles (constant or default CVP)

        - **MBDoE** (design_decisions provided):
            * Uses optimized sampling times from St
            * Optimized input profiles from tvi
            * Advanced CVP methods (step, ramp, etc.)

    **Measurement Noise**:
    Two error models supported:
        - **Absolute ('abs')**: \( y_{noisy} = y_{true} + \mathcal{N}(0, \sigma^2) \)
        - **Relative ('rel')**: \( y_{noisy} = y_{true} + \mathcal{N}(0, (\sigma \cdot y_{true})^2) \)

    Negative values are clipped to zero (physical constraint).

    **DataFrame Structure**:
    The output DataFrame contains multiple sections separated by blank columns:
        1. **Per-variable measurements**: Separate columns for each output variable
           with sampling times (MES_X), noisy values (MES_Y), and uncertainties (MES_E)
        2. **Global simulation**: Time vector (X:all), all inputs, time-invariant outputs
        3. **CVP metadata**: Control variable parameterization methods
        4. **Switching points**: Only for MBDoE designs with switching profiles

    **Excel Output**:
    Results are appended to 'data.xlsx' in the current directory. Each experiment
    creates/updates a sheet named by the experiment number. This allows accumulation
    of multiple experiments in a single file for batch processing.

    **Time Scaling**:
    All time values are normalized to [0, 1] internally, then scaled back to physical
    units (tsc = system['t_s'][1]) for output. This improves numerical conditioning.

    References
    ----------
    .. [1] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.

    .. [2] Bard, Y. (1974).
       *Nonlinear Parameter Estimation*. Academic Press.

    See Also
    --------
    simula : Forward model simulation kernel.
    _construct_var : Variable scaling and organization.
    _construct_par : Parameter vector construction.
    _inssimulator : Core simulation and noise addition routine.

    Examples
    --------
    >>> # Preliminary design (uniform sampling)
    >>> excel_path, df = expera(system, models, insilicos, None, expr=1)
    >>> print(f"Experiment saved to: {excel_path}")
    >>> print(f"Measurements: {df.filter(like='MES_Y').columns.tolist()}")

    >>> # MBDoE design (optimized sampling)
    >>> design_decisions = {
    ...     'St': {'y1': np.array([0, 50, 100]), 'y2': np.array([0, 75, 100])},
    ...     't_values': np.array([0, 0.5, 0.75, 1.0]),
    ...     'tii': {'P': 1.5, 'T0': 298},
    ...     'tvi': {'T': np.array([300, 320, 310, 300])}
    ... }
    >>> swps = {'Tt': np.array([0, 33, 67]), 'Tl': np.array([300, 320, 310])}
    >>> excel_path, df = expera(system, models, insilicos, design_decisions,
    ...                          expr=2, swps=swps)
    """
    # Extract scaling constant for independent variables
    tsc = system['t_s'][1]

    # Construct piecewise functions for initial and DOE cases
    cvp_initial = {var: 'no_CVP' for var in system['tvi'].keys()}
    cvp_doe = {var: system['tvi'][var]['cvp'] for var in system['tvi'].keys()}

    # Retrieve the model name and related simulation parameters
    model_name = insilicos.get('tr_m')
    errortype = insilicos.get('errt', 'abs')
    classic_des = insilicos['prels']
    theta_parameters = insilicos['theta']

    # Standard deviations for measured tv_ophi variables
    std_dev = {
        var: system['tvo'][var]['unc']
        for var in system['tvo'].keys()
        if system['tvo'][var].get('meas', True)
    }

    # If no design decisions are provided, run the "initial" case
    if not design_decisions:
        # Time values for each measured variable
        dt_real = system['t_r']
        nodes = int(round(tsc / dt_real)) + 1
        tlin = np.linspace(0, 1, nodes)

        # Generate t_values, snapping each to the closest value in tlin
        t_values = {
            var: [float(tlin[np.argmin(np.abs(tlin - t))]) for t in
                  np.linspace(0, 1, system['tvo'][var]['sp'])]
            for var in system['tvo']
            if system['tvo'][var].get('meas', True)
        }

        # Flatten all time points
        t_values_flat = [tp for times in t_values.values() for tp in times]

        # Create unique accumulated time points for simulation
        t_acc_unique = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

        # Construct variables and parameters for the "initial" case
        phi, phit, phisc, phitsc = _construct_var(system, classic_des, expr)
        theta, thetac = _construct_par(model_name, theta_parameters)

        if swps is None:
            swps, swpsu = {}, {}

        case = 'initial'
        df_combined = _inssimulator(
            t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac,
            cvp_initial, std_dev, t_acc_unique, case, model_name,
            system, models, errortype
        )

    else:
        # When design decisions are provided, run the "doe" case
        St = design_decisions['St']
        t_values = {var: np.array(St[var]) / tsc for var in St}

        # Accumulated time points for DOE
        t_acc_unique = np.array(design_decisions['t_values'])
        case = 'doe'

        # Construct scaling and parameters
        _, _, phisc, phitsc = _construct_var(system, classic_des, expr)
        theta, thetac = _construct_par(model_name, theta_parameters)

        phi = design_decisions['tii']
        phit = design_decisions['tvi']
        swpsu = swps if swps is not None else {}

        # Scale tii and tvi
        phi_scaled = {
            var: np.array(phi[var]) / phisc[var]
            for var in phi if var in system['tii'].keys()
        }
        phit_scaled = {
            var: np.array(phit[var])
            for var in phit if var in system['tvi'].keys()
        }

        # Scale swps
        swps_scaled = {}
        if swpsu:
            for var in swpsu:
                if var.endswith('l'):
                    # Length-based scaling
                    var_base = var.rstrip('l')
                    swps_scaled[var] = np.array(swpsu[var]) / phitsc[var_base]
                elif var.endswith('t'):
                    # Time-based scaling
                    swps_scaled[var] = np.array(swpsu[var]) / tsc
                else:
                    raise ValueError(f"Unrecognized variable format for {var} in swps.")

        df_combined = _inssimulator(
            t_values, swps_scaled, swpsu, phi_scaled, phisc, phit_scaled, phitsc, tsc, theta, thetac,
            cvp_doe, std_dev, t_acc_unique, case, model_name, system, models, errortype
        )

    # Define Excel path for in-silico data
    excel_path = Path.cwd() / 'data.xlsx'

    experiment_number = str(expr)
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

    # Return Excel path and DataFrame, with summary print
    print(f"[✓] In-silico data saved to: {excel_path}")
    print(f"[INFO] Model used         : {model_name}")
    print(f"[INFO] Design case        : {'classic/preliminary' if case == 'initial' else 'MBDoE'}")
    print(f"[INFO] Responses simulated:")

    for response in system.get('tvo', {}):
        meas = system['tvo'][response].get('meas', True)
        unc = system['tvo'][response].get('unc', 'N/A')
        status = "measurable" if meas else "non-measurable"
        print(f"   - {response:<10} | {status:<15} | std.dev = {unc}")

    return excel_path, df_combined


def _construct_var(system, classic_des, j):
    r"""
    Construct and normalize variable dictionaries from system configuration.

    This helper function extracts variable values from preliminary designs and normalizes
    them by their maximum bounds. It handles both time-invariant and time-variant inputs,
    preparing them for simulation in normalized coordinates.

    Parameters
    ----------
    system : dict
        System configuration with variable definitions:
            - 'tvi' : dict
                Time-variant inputs with 'max' bounds.
            - 'tii' : dict
                Time-invariant inputs with 'max' bounds.
    classic_des : dict
        Preliminary design values:
            - Keys: experiment indices (str)
            - Values: dicts mapping variable names to values
    j : int
        Experiment index for extracting values from classic_des.

    Returns
    -------
    phi : dict[str, float]
        Normalized time-invariant input values (phi / phisc).
    phit : dict[str, float]
        Normalized time-variant input values (phit / phitsc).
    phisc : dict[str, float]
        Scaling factors (max values) for time-invariant inputs.
    phitsc : dict[str, float]
        Scaling factors (max values) for time-variant inputs.

    Notes
    -----
    **Normalization**:
    All inputs are normalized to [0, 1] by dividing by their maximum values:
        \[
        \phi_{normalized} = \frac{\phi}{\phi_{max}}
        \]
    This improves numerical conditioning and allows bounds-free optimization.

    **Default Values**:
    If a variable is not present in classic_des[j], its maximum value is used as default.
    This ensures all variables have valid values even if partially specified.

    **Use Cases**:
        - Preliminary design: Uses classic_des values directly
        - MBDoE design: Returns only scaling factors (phisc, phitsc)

    See Also
    --------
    expera : Main function that calls this for variable setup.
    _construct_par : Analogous function for parameters.

    Examples
    --------
    >>> system = {
    ...     'tvi': {'T': {'max': 400}},
    ...     'tii': {'P': {'max': 5.0}}
    ... }
    >>> classic_des = {'1': {'T': 350, 'P': 3.0}}
    >>> phi, phit, phisc, phitsc = _construct_var(system, classic_des, 1)
    >>> print(phi)  # {'P': 0.6}  (3.0 / 5.0)
    >>> print(phit)  # {'T': 0.875}  (350 / 400)
    >>> print(phisc)  # {'P': 5.0}
    >>> print(phitsc)  # {'T': 400}
    """
    phit = {}
    phisc = {}
    phitsc = {}
    phi = {}

    # Time-variant input variables (tv_iphi)
    for var in system['tvi'].keys():
        max_val = system['tvi'][var]['max']
        classic_value = classic_des.get(str(j), {}).get(var, max_val)
        phitsc[var] = max_val
        phit[var] = classic_value / max_val

    # Time-invariant input variables (ti_iphi)
    for var in system['tii'].keys():
        max_val = system['tii'][var]['max']
        classic_value = classic_des.get(str(j), {}).get(var, max_val)
        phisc[var] = max_val
        phi[var] = classic_value / max_val

    return phi, phit, phisc, phitsc


def _construct_par(model_name, theta_parameters):
    r"""
    Construct normalized and scaled parameter vectors for simulation.

    This helper function prepares parameter vectors for forward model simulation.
    Parameters are normalized to [1, 1, ...] with separate scaling factors (thetac)
    following the same convention used in parameter estimation routines.

    Parameters
    ----------
    model_name : str
        Model name (currently unused but passed for extensibility).
    theta_parameters : list[float]
        True parameter values (ground truth for in-silico experiments).

    Returns
    -------
    theta : list[float]
        Normalized parameter vector (all ones: [1, 1, ..., 1]).
    thetac : list[float]
        Scaling factors (true parameter values).

    Raises
    ------
    ValueError
        If theta_parameters is invalid or cannot be accessed.

    Notes
    -----
    **Normalization Convention**:
    Following the pattern used throughout the package:
        \[
        \theta_{physical} = \theta_{normalized} \cdot \theta_c
        \]
    where \( \theta_{normalized} = [1, 1, ..., 1] \) for the true model.

    **True Model Simulation**:
    In-silico experiments use the "true" parameter values from insilicos['theta'].
    These represent the ground truth that parameter estimation aims to recover.

    **Consistency with Estimation**:
    This normalization matches the convention in parmest(), allowing estimated
    parameters (theta * thetac) to be directly compared to ground truth (thetac).

    See Also
    --------
    expera : Main function that calls this for parameter setup.
    _construct_var : Analogous function for variables.
    parmest : Parameter estimation routine using same normalization.

    Examples
    --------
    >>> theta_true = [1.5, 2.3, 0.8]  # True parameters
    >>> theta, thetac = _construct_par('M1', theta_true)
    >>> print(theta)  # [1, 1, 1]
    >>> print(thetac)  # [1.5, 2.3, 0.8]
    >>> # Physical parameters: theta * thetac = [1.5, 2.3, 0.8]
    """
    try:
        thetac = theta_parameters
        theta = [1] * len(thetac)
        return theta, thetac
    except Exception as e:
        raise ValueError(f"Unable to adjust theta or thetac for model {model_name}: {e}")


def _inssimulator(t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac, cvp, std_dev,
                  t_valuesc, case, model_name, system, models, errortype):
    r"""
    Execute forward simulation and add measurement noise to create synthetic experimental data.

    This is the core simulation routine that calls the forward model, generates noisy measurements,
    and organizes results into a structured DataFrame. It handles both time-variant and time-invariant
    outputs, interpolates input profiles, and applies realistic measurement noise according to
    specified error models.

    Parameters
    ----------
    t_values : dict[str, list[float]]
        Measurement times for each output variable (normalized [0,1]):
            - Keys: output variable names (e.g., 'y1', 'y2')
            - Values: lists of normalized sampling times
    swps : dict[str, np.ndarray]
        Scaled switching points:
            - '{var}t': switching times (normalized)
            - '{var}l': switching levels (normalized)
    swpsu : dict[str, np.ndarray]
        Unscaled switching points (for recording in output).
    phi : dict[str, float]
        Normalized time-invariant inputs.
    phisc : dict[str, float]
        Scaling factors for time-invariant inputs.
    phit : dict[str, float] or dict[str, np.ndarray]
        Time-variant inputs (constant or profile).
    phitsc : dict[str, float]
        Scaling factors for time-variant inputs.
    tsc : float
        Time scaling factor (physical time = normalized time * tsc).
    theta : list[float]
        Normalized parameter vector.
    thetac : list[float]
        Parameter scaling factors.
    cvp : dict[str, str]
        Control variable parameterization methods for each time-variant input.
    std_dev : dict[str, float]
        Measurement standard deviations for each output variable.
    t_valuesc : np.ndarray
        Global time vector for simulation (normalized).
    case : str
        Experiment type: 'initial' (preliminary) or 'doe' (MBDoE).
    model_name : str
        Name of the forward model to simulate.
    system : dict
        System configuration.
    models : dict
        Model definitions.
    errortype : str
        Error type: 'abs' (absolute noise) or 'rel' (relative noise).

    Returns
    -------
    df_combined : pd.DataFrame
        Comprehensive DataFrame with all experimental data organized in sections:
            **Section 1 - Per-variable measurements**:
                - 'MES_X:{var}': Sampling times (physical units)
                - 'MES_Y:{var}': Noisy measured values
                - 'MES_E:{var}': Measurement uncertainties

            **Section 2 - Global simulation**:
                - 'X:all': Full time vector
                - Time-invariant inputs (phi variables)
                - Time-variant inputs (phit variables, interpolated)
                - Time-invariant outputs (noisy, constant)

            **Section 3 - CVP metadata**:
                - 'CVP:{var}': Control parameterization method for each input

            **Section 4 - Switching points** (DOE only):
                - '{var}t': Switching times (unscaled)
                - '{var}l': Switching levels (unscaled)

    Raises
    ------
    ValueError
        If case is not 'initial' or 'doe', or if errortype is invalid.
    KeyError
        If a measured variable is missing from simulation results.

    Notes
    -----
    **Simulation Workflow**:
        1. Call forward model (simula) with normalized inputs/parameters
        2. Extract predictions at measurement times via interpolation
        3. Add Gaussian noise according to error type
        4. Clip negative values to zero (physical constraint)
        5. Organize into standardized DataFrame format

    **Noise Models**:
        - **Absolute** ('abs'):
          \[
          y_{noisy} \sim \mathcal{N}(y_{true}, \sigma^2)
          \]
          Constant variance across signal range.

        - **Relative** ('rel'):
          \[
          y_{noisy} \sim \mathcal{N}(y_{true}, (\sigma \cdot |y_{true}|)^2)
          \]
          Signal-dependent variance (heteroscedastic).

    **Time-Invariant Output Noise**:
    For time-invariant outputs, noise is added once and replicated across all time points.
    This reflects that these quantities are measured once per experiment, not continuously.

    **DataFrame Organization**:
    Multiple sections are combined horizontally with blank columns as separators.
    This mimics the structure expected by read_excel() during parameter estimation,
    ensuring consistency between data generation and analysis.

    **Negative Value Handling**:
    Physical quantities (concentrations, temperatures, etc.) cannot be negative.
    Noisy values are clipped: \( y_{noisy} = \max(y_{noisy}, 0) \).

    See Also
    --------
    expera : Main function that calls this routine.
    simula : Forward model simulation kernel.

    Examples
    --------
    >>> # Typically called internally by expera()
    >>> df = _inssimulator(
    ...     t_values={'y1': [0, 0.5, 1.0]},
    ...     swps={}, swpsu={},
    ...     phi={'P': 0.6}, phisc={'P': 5.0},
    ...     phit={'T': 0.875}, phitsc={'T': 400},
    ...     tsc=100, theta=[1, 1], thetac=[1.5, 2.3],
    ...     cvp={'T': 'no_CVP'}, std_dev={'y1': 0.05},
    ...     t_valuesc=np.array([0, 0.5, 1.0]),
    ...     case='initial', model_name='M1',
    ...     system=system, models=models, errortype='abs'
    ... )
    >>> print(df.filter(like='MES_Y').columns)  # ['MES_Y:y1']
    """
    # Run the model to get tv_ophi, ti_ophi, phit_interp
    if case == 'initial':
        tv_ophi, ti_ophi, phit_interp = simula(
            t_valuesc, {}, phi, phisc, phitsc, tsc, theta, thetac,
            cvp, phit, model_name, system, models
        )
    elif case == 'doe':
        tv_ophi, ti_ophi, phit_interp = simula(
            t_valuesc, swps, phi, phisc, phitsc, tsc, theta, thetac,
            cvp, phit, model_name, system, models
        )
    else:
        raise ValueError(f"Unsupported case: {case}")

    # Identify measured variables from system
    measured_vars = [
        var for var in system['tvo'].keys()
        if system['tvo'][var].get('measured', True)
    ]

    # Prepare lists of independent and dependent variables
    indep_vars = list(range(len(measured_vars)))
    dep_vars = measured_vars

    if len(indep_vars) != len(dep_vars):
        raise ValueError("The number of independent variables must match the number of dependent variables.")

    indep_to_dep_mapping = dict(zip([f"X{i+1}" for i in indep_vars], dep_vars))

    # Check that all dependent variables exist in tv_ophi
    for indep_var, dep_var in indep_to_dep_mapping.items():
        if dep_var not in tv_ophi:
            raise KeyError(
                f"Dependent variable '{dep_var}' is missing in tv_ophi. Available keys: {list(tv_ophi.keys())}"
            )

    # 1. Per-variable DataFrames for tv_ophi
    var_dfs = []
    for indep_var, dep_var in indep_to_dep_mapping.items():
        var_times = [t * tsc for t in t_values[dep_var]]
        var_results = []

        for t in var_times:
            # Find the closest index in the scaled time vector
            closest_idx = np.argmin(np.abs(np.array(t_valuesc) * tsc - t))
            original_val = tv_ophi[dep_var][closest_idx]

            # Compute noise based on selected error type
            if errortype == 'abs':
                sigma = std_dev[dep_var]
            elif errortype == 'rel':
                sigma = std_dev[dep_var] * abs(original_val)
            else:
                raise ValueError(f"Unsupported errortype: {errortype}")

            noisy_val = np.random.normal(loc=original_val, scale=sigma)
            noisy_val = max(noisy_val, 0.0)
            var_results.append([t, noisy_val, sigma])

        df_var = pd.DataFrame(var_results, columns=[f"MES_X:{dep_var}", f"MES_Y:{dep_var}", f"MES_E:{dep_var}"])
        var_dfs.append(df_var)

    # 2. Global DataFrame (t_global, tii, tvi/phit_interp, ti_ophi)
    phi_order = list(phi.keys())
    ti_ophi_order = list(ti_ophi.keys())
    if case == 'classic':
        phit_order = list(phit.keys())
    else:
        phit_order = list(phit_interp.keys())

    # Noise added once for ti_ophi
    ti_ophi_noisy_once = {
        var: np.random.normal(val[0], abs(std_dev.get(var, 0) * val[0]))
        for var, val in ti_ophi.items()
    }

    global_results = []
    for i, t_global in enumerate(t_valuesc):
        row = [t_global * tsc]

        # tii
        phi_vals = [phi[v] * phisc[v] for v in phi_order]
        row.extend(phi_vals)

        # tvi or phit_interp
        if case == 'classic':
            phit_vals = [phit[v] * phitsc[v] for v in phit_order]
        else:
            phit_vals = [phit_interp[v][i] for v in phit_order]
        row.extend(phit_vals)

        # ti_ophi noisy (constant)
        ti_ophi_vals = [ti_ophi_noisy_once[v] for v in ti_ophi_order]
        row.extend(ti_ophi_vals)

        global_results.append(row)

    global_columns = (['X:all'] + phi_order + phit_order + ti_ophi_order)
    df_global = pd.DataFrame(global_results, columns=global_columns)

    # Combine var_dfs and df_global horizontally
    all_dfs = var_dfs + [df_global]
    df_main = pd.concat(all_dfs, axis=1)

    # 3. piecewise_func DataFrame
    df_piecewise = pd.DataFrame([{f"CVP:{var}": method for var, method in cvp.items()}])

    # Combine all into one DataFrame
    df_combined = pd.concat([df_main, df_piecewise], axis=1)

    # 4. swpsu DataFrame
    if case == 'doe':
        swpsu_order = list(swpsu.keys())
        swps_data = {key: swpsu[key] for key in swpsu_order}
        df_swps = pd.DataFrame(swps_data)

        # Insert blank columns to separate sections
        max_length = max(len(df_main), len(df_piecewise), len(df_swps))
        df_blank_1 = pd.DataFrame({' ': [None] * max_length})
        df_blank_2 = pd.DataFrame({'  ': [None] * max_length})

        # Combine all into one DataFrame
        df_combined = pd.concat([df_main, df_blank_1, df_piecewise, df_blank_2, df_swps], axis=1)

    return df_combined