import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import importlib

def Simula(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model_name, model_structure, modelling_settings):
    """
    Simulate the system dynamics using the provided parameters and model.

    Parameters
    ----------
    t : list or array of float
        Time points for the simulation.
    swps : dict of str: list
        Dictionary containing time and level vectors for piecewise interpolation.
        Keys are variable names ending with 't' or 'l' that define time and level vectors.
    uphi : dict of str: float
        Time-invariant variables scaled by their maxima.
    uphisc : dict of str: float
        Scaling factors for each time-invariant variable.
    uphitsc : dict of str: float
        Scaling factors for each time-variant variable.
    utsc : float
        Scaling factor for the time points.
    utheta : list of float
        Model parameters (unscaled).
    uthetac : list of float
        Scaling factors for each parameter in utheta.
    cvp : dict of str: str
        A dictionary mapping each time-variant input variable to an interpolation method:
        - 'CPF': Constant piecewise function (previous step interpolation)
        - 'LPF': Linear piecewise function
        - 'none': No interpolation, use a constant from uphit
    uphit : dict of str: float or list
        Initial or constant values for time-variant variables when using the 'none' method.
    model_name : str
        The name of the model function to be used for simulation.
    model_structure : dict
        Defines model structure, including variables, attributes, and initial conditions.
    modelling_settings : dict
        Settings for the modelling process, including external functions.

    Returns
    -------
    tuple
        tv_ophi (dict): Time-varying outputs {var_name: ndarray}
        ti_ophi (dict): Time-invariant outputs (currently empty)
        phit (dict): Scaled piecewise interpolated data {var_name: ndarray}
    """
    # Ensure 'ext_func' key is present in modelling_settings
    if 'ext_func' not in modelling_settings:
        raise KeyError("'ext_func' key is missing in modelling_settings")

    # Check if model_name is in modelling_settings['ext_func']
    if model_name in modelling_settings['ext_func']:
        model = modelling_settings['ext_func'][model_name]
    else:
        # Attempt to import the model from kernel_models if not found in ext_func
        try:
            krnl_models = importlib.import_module('middoe.krnl_models')
            model = getattr(krnl_models, model_name)
        except (ImportError, AttributeError):
            raise KeyError(f"Model function '{model_name}' not found in 'ext_func' or 'kernel_models'")

    tv_ophi, ti_ophi, phit = _ivpsolver(
        t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac,
        cvp, uphit, model, model_structure, modelling_settings
    )
    return tv_ophi, ti_ophi, phit

def _Piecewiser(t, swps, cvp, phit):
    """
    Perform piecewise interpolation for trajectories of time-variant input variables.

    Parameters
    ----------
    t : array-like
        Time points at which to evaluate the piecewise functions.
    swps : dict of str: list
        Dictionary containing time and level vectors for piecewise functions.
        For a variable 'X', 'Xt' defines times and 'Xl' defines levels.
    cvp : dict of str: str
        Dictionary of methods for each variable. Keys are variable names, values are:
        - 'CPF': Constant piecewise function (previous step interpolation)
        - 'LPF': Linear piecewise function
        - 'none': No interpolation, use phit values as constants.
    phit : dict of str: float or list
        Initial or constant values for each variable when using the 'none' method.

    Returns
    -------
    dict of str: ndarray
        Dictionary mapping variable names to arrays of interpolated values.
    """

    result = {}

    # Iterate over each variable defined in cvp
    for var, method in cvp.items():
        time_key = f'{var}t'
        level_key = f'{var}l'

        if method == 'CPF':
            # For CPF, we need both time and level data in swps
            if time_key in swps and level_key in swps:
                f_previous = interp1d(swps[time_key], swps[level_key], kind='previous', fill_value="extrapolate")
                result[var] = f_previous(t)
            else:
                raise KeyError(f"Missing keys for CPF method: '{time_key}' or '{level_key}'")

        elif method == 'LPF':
            # For LPF, we need both time and level data in swps
            if time_key in swps and level_key in swps:
                result[var] = np.interp(t, swps[time_key], swps[level_key])
            else:
                raise KeyError(f"Missing keys for LPF method: '{time_key}' or '{level_key}'")

        elif method == 'none':
            # For 'none', we do not rely on swps. Instead, we use the constant value from phit.
            if var not in phit:
                raise KeyError(f"No initial value provided in phit for variable '{var}' under 'none' method.")
            value = phit[var]
            if isinstance(value, (list, np.ndarray)):
                result[var] = np.full_like(t, value[0])
            else:
                result[var] = np.full_like(t, value)

        else:
            # Unrecognized method
            raise ValueError(f"Unrecognized piecewise method '{method}' for variable '{var}'.")

    return result


def _ivpsolver(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model, model_structure, modelling_settings):
    """
    Solve the ODE system defined by the model using solve_ivp.

    Parameters
    ----------
    t : list or array of float
        Time points for the simulation.
    swps : dict
        Switching points data for piecewise interpolation.
    uphi : dict
        Time-invariant variables (scaled to [0,1]).
    uphisc : dict
        Scaling factors for time-invariant variables.
    uphitsc : dict
        Scaling factors for time-variant variables.
    utsc : float
        Scaling factor for time.
    utheta : list of float
        Model parameters (unscaled).
    uthetac : list of float
        Scaling factors for parameters.
    cvp : dict of str: str
        Dictionary specifying interpolation methods per variable.
    uphit : dict
        Initial or constant values for time-variant variables when using 'none'.
    model : callable
        The model function to be integrated.
    model_structure : dict
        Defines the structure of the model, including variable attributes.
    modelling_settings : dict
        Additional modelling settings.

    Returns
    -------
    tuple
        tv_ophi (dict): {variable_name: ndarray of solution values over time}
        ti_ophi (dict): {time_invariant_output_name: value} (currently empty)
        phit (dict): {variable_name: ndarray of interpolated input values}
    """
    # Scale time-invariant variables by their max values
    phi = {k: v * uphisc[k] for k, v in uphi.items() if k in uphisc}

    # Scale parameters
    theta = [utheta[i] * uthetac[i] for i in range(len(utheta))]

    # Compute piecewise interpolations and scale them
    piecewise_data = _Piecewiser(t, swps, cvp, uphit)
    phit = {k: piecewise_data[k] * uphitsc[k] for k in uphitsc}

    # Construct initial conditions
    y0 = []
    for var in model_structure['tv_ophi'].keys():
        var_attrs = model_structure['tv_ophi'][var]
        initial_val = var_attrs['initials']
        if initial_val == 'variable':
            # Use time-invariant variable with '_0' suffix
            y0.append(phi[var + '_0'])
        else:
            # Use the specified initial value directly
            y0.append(initial_val)

    # Scale the time vector
    t_scaled = (np.array(t) * utsc).tolist()

    # Solve the ODE system
    result = solve_ivp(
        model,
        [min(t_scaled), max(t_scaled)],
        y0,
        method='LSODA',
        t_eval=t_scaled,
        args=(phi, phit, theta, t_scaled)
    )

    # Extract time-varying outputs
    tv_ophi = {f'y{i+1}': result.y[i] for i in range(len(result.y))}
    ti_ophi = {}  # Currently empty, add logic if needed

    return tv_ophi, ti_ophi, phit
