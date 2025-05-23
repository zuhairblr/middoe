from middoe.krnl_simula import simula
import numpy as np
import pandas as pd
from pathlib import Path
import os

def expera(system, models, insilicos, design_decisions, expr, swps=None):
    """
    Run an in-silico experiment, produce simulation results, and prepare them for reporting.

    This function:
    1. Sets up paths and simulation parameters.
    2. Constructs scaled and unscaled variables, parameters, and switching points as needed.
    3. Calls the in-silico simulator to generate results.
    4. Returns the path to an Excel file and the resulting simulation DataFrame.

    Parameters
    ----------
    system : dict
        Structure of the model, including variables and their properties.
        It contains keys like 'tv_iphi', 'tv_ophi', 'ti_iphi', 'ti_ophi',
        and 'independant_variables' each with 'var' and 'attributes' sub-dicts.

    models : dict
        User-defined settings related to the modelling process, including 'theta_parameters'.

    insilicos : dict
        User-defined settings for the simulator, including:
        - 'insilico_model': str, name of the model function to be simulated.
        - 'classic-des': dict, containing default or "classic" design values.
        - 'nd2': int, number of points for time discretization.

    design_decisions : dict
        Design decisions for the experiment. May contain keys like 'St', 'tii', 'tvi', 't_values'.

    expr : int
        Index for the current experiment.

    swps : dict, optional
        Switching points for the time-variant variables, unscaled. If None, empty dicts are used.

    Returns
    -------
    tuple
        (excel_path, df_combined)
        excel_path : str
            The path to the Excel file where results are to be stored.
        df_combined : pandas.DataFrame
            The combined results of the in-silico experiment.
    """
    # Extract scaling constant for independent variables
    tsc = system['t_s'][1]


    # Construct piecewise functions for initial and DOE cases
    cvp_initial = {
        var: 'no_CVP'
        for var in system['tvi'].keys()
    }
    cvp_doe = {
        var: system['tvi'][var]['cvp']
        for var in system['tvi'].keys()
    }

    # base_path = framework_settings['path']
    # modelling_folder = str(framework_settings['case'])
    # output_path = Path(base_path) / modelling_folder
    # output_path.mkdir(parents=True, exist_ok=True)

    # Retrieve the model name and related simulation parameters
    model_name = insilicos.get('tr_m')
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
        # Step 1: Create tlin
        dt_real = system['t_r']
        nodes = int(round(tsc / dt_real)) + 1
        tlin = np.linspace(0, 1, nodes)

        # Step 2: Generate t_values, snapping each to the closest value in tlin
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
            system, models
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
            cvp_doe, std_dev, t_acc_unique, case, model_name, system, models)

    # Define Excel path for in-silico data
    excel_path = Path.cwd() / 'indata.xlsx'

    experiment_number = str(expr)  # Ensure experiment_number is a string
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
    print(f"\n[✓] In-silico data saved to: {excel_path}")
    print(f"[INFO] Model used         : {model_name}")
    print(f"[INFO] Design case        : {'classic/preliminary' if case == 'initial' else 'MBDoE'}")
    print(f"[INFO] Responses simulated:")

    for response in system.get('tvo', {}):
        meas = system['tvo'][response].get('meas', True)
        unc = system['tvo'][response].get('unc', 'N/A')
        status = "measurable" if meas else "non-measurable"
        print(f"   - {response:<10} | {status:<15} | std.dev = {unc}")

    return excel_path, df_combined

    # return df_combined


def _construct_var(system, classic_des, j):
    """
    Construct and scale variable dictionaries for the simulation.

    Parameters
    ----------
    system : dict
        Dictionary describing the model structure.
    classic_des : dict
        Dictionary containing "classic" design values for variables.
    j : int
        Index of the current experiment.

    Returns
    -------
    tuple
        (tii, tvi, phisc, phitsc):
        - tii : dict
            Time-invariant variable values scaled by their maxima.
        - tvi : dict
            Time-variant variable values scaled by their maxima.
        - phisc : dict
            Scaling factors (max values) for time-invariant variables.
        - phitsc : dict
            Scaling factors (max values) for time-variant variables.
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
    """
    Construct parameter arrays for the simulation, given the model name and theta parameters.

    Parameters
    ----------
    model_name : str
        The name of the model.
    theta_parameters : dict
        Dictionary mapping model names to theta scaling parameters.

    Returns
    -------
    tuple
        (theta, thetac):
        - theta : list
            Normalized parameter values (e.g., [1, 1, ...]).
        - thetac : list
            Actual parameter scaling values for the given model.

    Raises
    ------
    ValueError
        If parameters cannot be constructed or accessed.
    """
    try:
        thetac = theta_parameters
        theta = [1] * len(thetac)
        return theta, thetac
    except Exception as e:
        raise ValueError(f"Unable to adjust theta or thetac for model {model_name}: {e}")


def _inssimulator(t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac, cvp, std_dev,
                  t_valuesc, case, model_name, system, models):
    """
    Conduct the in-silico experiments by setting up the simulation environment,
    running the simulations, adding noise, and saving the results.

    Returns one combined DataFrame with sections:
    - var_t, var_noisy for each tv_ophi variable (each with its own length).
    - t_global, tii(s), tvi(s)/phit_interp(s), ti_ophi (aligned with t_valuesc).
    - piecewise_func in a separate section.
    - swpsu in another separate section.

    All in one DataFrame, separated by blank columns.

    Parameters
    ----------
    t_values : dict
        Dictionary of time values keyed by dependent variable name,
        e.g. {'y1': [0, 0.25, 0.5, 1.0], 'y2': [0, 0.5, 1.0]}.
    swps : dict
        Scaled switching points for the time-variant variables.
    swpsu : dict
        Unscaled switching points for the time-variant variables.
    phi, phisc : dict
        Dictionaries of time-invariant variables and their scaling factors.
    phit, phitsc : dict
        Dictionaries of time-variant variables and their scaling factors.
    tsc : float
        Scaling factor for time.
    theta, thetac : list
        Parameter vectors.
    piecewise_func : dict
        Dictionary of piecewise functions defining variable trajectories.
    std_dev : dict
        Dictionary of standard deviations keyed by variable name.
    t_valuesc : list
        Global set of time points for simulation.
    case : str
        Indicates the type of run ('initial', 'doe', etc.).
    model_func : callable
        The simulation function used to generate tv_ophi, ti_ophi, phit_interp.
    model_name : str
        The name of the model being simulated.
    system : dict
        Defines the structure of the model, including variable attributes.
    models : dict
        Additional modelling settings.

    Returns
    -------
    pd.DataFrame
        A combined DataFrame with simulation results, noisy measurements, piecewise function info, and switching points.
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
    # Assuming that there is a one-to-one mapping defined externally,
    # we match the number of measured vars with the number of independent vars
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
        var_times = [t * tsc for t in t_values[dep_var]]  # Times for the dependent variable
        var_results = []

        # Add noise to measurements
        for t in var_times:
            closest_idx = np.argmin(np.abs(np.array(t_valuesc) * tsc - t))
            original_val = tv_ophi[dep_var][closest_idx]
            sigma = std_dev[dep_var]  # absolute standard deviation
            noisy_val = np.random.normal(loc=original_val, scale=sigma)
            var_results.append([t, noisy_val])

        df_var = pd.DataFrame(var_results, columns=[f"MES_X:{dep_var}", f"MES_Y:{dep_var}"])
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

    global_columns = (['X:all']
                      + phi_order
                      + phit_order
                      + ti_ophi_order)
    df_global = pd.DataFrame(global_results, columns=global_columns)

    # Combine var_dfs and df_global horizontally
    all_dfs = var_dfs + [df_global]
    df_main = pd.concat(all_dfs, axis=1)

    # 3. piecewise_func DataFrame
    df_piecewise = pd.DataFrame([{
        f"CVP:{var}": method for var, method in cvp.items()
    }])

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


