from middoe.krnl_simula import Simula
import numpy as np
import pandas as pd
from pathlib import Path

def Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j, swps=None):
    """
    Run an in-silico experiment, produce simulation results, and prepare them for reporting.

    This function:
    1. Sets up paths and simulation parameters.
    2. Constructs scaled and unscaled variables, parameters, and switching points as needed.
    3. Calls the in-silico simulator to generate results.
    4. Returns the path to an Excel file and the resulting simulation DataFrame.

    Parameters
    ----------
    framework_settings : dict
        Settings related to the framework, including paths and case information.
        Example structure:
        {
            'path': '/absolute/path/to/project',
            'case': 'experiment_01'
        }

    model_structure : dict
        Structure of the model, including variables and their properties.
        It contains keys like 'tv_iphi', 'tv_ophi', 'ti_iphi', 'ti_ophi',
        and 'independant_variables' each with 'var' and 'attributes' sub-dicts.

    modelling_settings : dict
        User-defined settings related to the modelling process, including 'theta_parameters'.

    simulator_settings : dict
        User-defined settings for the simulator, including:
        - 'insilico_model': str, name of the model function to be simulated.
        - 'classic-des': dict, containing default or "classic" design values.
        - 'nd2': int, number of points for time discretization.

    design_decisions : dict
        Design decisions for the experiment. May contain keys like 'St', 'phi', 'phit', 't_values'.

    j : int
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
    tsc = model_structure['t_s'][1]

    # Construct piecewise functions for initial and DOE cases
    cvp_initial = {
        var: model_structure['tv_iphi'][var]['initial_cvp']
        for var in model_structure['tv_iphi'].keys()
    }
    cvp_doe = {
        var: model_structure['tv_iphi'][var]['design_cvp']
        for var in model_structure['tv_iphi'].keys()
    }

    base_path = framework_settings['path']
    modelling_folder = str(framework_settings['case'])
    output_path = Path(base_path) / modelling_folder
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = str(output_path / "insilico_experiments.xlsx")

    # Retrieve the model name and related simulation parameters
    model_name = simulator_settings.get('insilico_model')
    classic_des = simulator_settings['classic-des']
    theta_parameters = modelling_settings['theta_parameters']
    nodes = simulator_settings.get('smoothness', 300)

    # Standard deviations for measured tv_ophi variables
    std_dev = {
        var: model_structure['tv_ophi'][var]['unc']
        for var in model_structure['tv_ophi'].keys()
        if model_structure['tv_ophi'][var].get('measured', True)
    }

    # If no design decisions are provided, run the "initial" case
    if not design_decisions:
        # Time values for each measured variable
        t_values = {
            var: np.linspace(0, 1, model_structure['tv_ophi'][var]['sp']).tolist()
            for var in model_structure['tv_ophi'].keys()
            if model_structure['tv_ophi'][var].get('measured', True)
        }

        # Flatten all time points
        t_values_flat = [tp for times in t_values.values() for tp in times]

        # Create unique accumulated time points for simulation
        t_acc_unique = np.unique(np.concatenate((np.linspace(0, 1, nodes), t_values_flat))).tolist()

        # Construct variables and parameters for the "initial" case
        phi, phit, phisc, phitsc = _construct_var(model_structure, classic_des, j)
        theta, thetac = _construct_par(model_name, theta_parameters)

        if swps is None:
            swps, swpsu = {}, {}

        case = 'initial'
        df_combined = _inssimulator(
            t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac,
            cvp_initial, std_dev, t_acc_unique, case, Simula, model_name,
            model_structure, modelling_settings
        )

    else:
        # When design decisions are provided, run the "doe" case
        St = design_decisions['St']
        t_values = {var: np.array(St[var]) / tsc for var in St}

        # Accumulated time points for DOE
        t_acc_unique = np.array(design_decisions['t_values'])
        case = 'doe'

        # Construct scaling and parameters
        _, _, phisc, phitsc = _construct_var(model_structure, classic_des, j)
        theta, thetac = _construct_par(model_name, theta_parameters)

        phi = design_decisions['phi']
        phit = design_decisions['phit']
        swpsu = swps if swps is not None else {}

        # Scale phi and phit
        phi_scaled = {
            var: np.array(phi[var]) / phisc[var]
            for var in phi if var in model_structure['ti_iphi'].keys()
        }
        phit_scaled = {
            var: np.array(phit[var])
            for var in phit if var in model_structure['tv_iphi'].keys()
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
            cvp_doe, std_dev, t_acc_unique, case, Simula, model_name,
            model_structure, modelling_settings
        )

    return excel_path, df_combined


def _construct_var(model_structure, classic_des, j):
    """
    Construct and scale variable dictionaries for the simulation.

    Parameters
    ----------
    model_structure : dict
        Dictionary describing the model structure.
    classic_des : dict
        Dictionary containing "classic" design values for variables.
    j : int
        Index of the current experiment.

    Returns
    -------
    tuple
        (phi, phit, phisc, phitsc):
        - phi : dict
            Time-invariant variable values scaled by their maxima.
        - phit : dict
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
    for var in model_structure['tv_iphi'].keys():
        max_val = model_structure['tv_iphi'][var]['max']
        classic_value = classic_des.get(str(j), {}).get(var, max_val)
        phitsc[var] = max_val
        phit[var] = classic_value / max_val

    # Time-invariant input variables (ti_iphi)
    for var in model_structure['ti_iphi'].keys():
        max_val = model_structure['ti_iphi'][var]['max']
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
        thetac = theta_parameters.get(model_name, [])
        theta = [1] * len(thetac)
        return theta, thetac
    except Exception as e:
        raise ValueError(f"Unable to adjust theta or thetac for model {model_name}: {e}")


def _inssimulator(t_values, swps, swpsu, phi, phisc, phit, phitsc, tsc, theta, thetac, cvp, std_dev,
                  t_valuesc, case, Simula, model_name, model_structure, modelling_settings):
    """
    Conduct the in-silico experiments by setting up the simulation environment,
    running the simulations, adding noise, and saving the results.

    Returns one combined DataFrame with sections:
    - var_t, var_noisy for each tv_ophi variable (each with its own length).
    - t_global, phi(s), phit(s)/phit_interp(s), ti_ophi (aligned with t_valuesc).
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
    model_structure : dict
        Defines the structure of the model, including variable attributes.
    modelling_settings : dict
        Additional modelling settings.

    Returns
    -------
    pd.DataFrame
        A combined DataFrame with simulation results, noisy measurements, piecewise function info, and switching points.
    """

    # Run the model to get tv_ophi, ti_ophi, phit_interp
    if case == 'initial':
        tv_ophi, ti_ophi, phit_interp = Simula(
            t_valuesc, {}, phi, phisc, phitsc, tsc, theta, thetac,
            cvp, phit, model_name, model_structure, modelling_settings
        )
    elif case == 'doe':
        tv_ophi, ti_ophi, phit_interp = Simula(
            t_valuesc, swps, phi, phisc, phitsc, tsc, theta, thetac,
            cvp, phit, model_name, model_structure, modelling_settings
        )
    else:
        raise ValueError(f"Unsupported case: {case}")

    # Identify measured variables from model_structure
    measured_vars = [
        var for var in model_structure['tv_ophi'].keys()
        if model_structure['tv_ophi'][var].get('measured', True)
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
            sigma = abs(std_dev[dep_var] * original_val)
            noisy_val = np.random.normal(original_val, sigma)
            var_results.append([t, noisy_val])

        df_var = pd.DataFrame(var_results, columns=[f"MES_X:{dep_var}", f"MES_Y:{dep_var}"])
        var_dfs.append(df_var)

    # 2. Global DataFrame (t_global, phi, phit/phit_interp, ti_ophi)
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

        # phi
        phi_vals = [phi[v] * phisc[v] for v in phi_order]
        row.extend(phi_vals)

        # phit or phit_interp
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


