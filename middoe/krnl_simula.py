# # krnl_simula.py
#
# import numpy as np
# from scipy.interpolate import interp1d
# import importlib
# from subprocess import DEVNULL
# import time
# import random
# import importlib.util
#
#
# def simula(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model_name, system, models):
#     """
#     Simulate the system dynamics using the provided parameters and model.
#
#     Parameters
#     ----------
#     t : list or array of float
#         Time points for the simulation.
#     swps : dict of str: list
#         Dictionary containing time and level vectors for piecewise interpolation.
#     uphi : dict of str: float
#         Time-invariant variables scaled by their maxima.
#     uphisc : dict of str: float
#         Scaling factors for each time-invariant variable.
#     uphitsc : dict of str: float
#         Scaling factors for each time-variant variable.
#     utsc : float
#         Scaling factor for the time points.
#     utheta : list of float
#         Model parameters (unscaled).
#     uthetac : list of float
#         Scaling factors for each parameter in utheta.
#     cvp : dict of str: str
#         A dictionary mapping each time-variant input variable to an interpolation method:
#         - 'CPF': Constant piecewise function (previous step interpolation)
#         - 'LPF': Linear piecewise function
#         - 'none': No interpolation, use a constant from uphit
#     uphit : dict of str: float or list
#         Initial or constant values for time-variant variables when using the 'none' method.
#     model_name : str
#         The name of the model function to be used for simulation.
#     system : dict
#         Defines model structure, including variables, attributes, and initial conditions.
#     models : dict
#         Settings for the modelling process, including external functions.
#
#     Returns
#     -------
#     tuple
#         tv_ophi (dict): Time-varying outputs {var_name: ndarray}
#         ti_ophi (dict): Time-invariant outputs (currently empty)
#         tvi (dict): Scaled piecewise interpolated data {var_name: ndarray}
#     """
#
#     model_type = models['krt'].get(model_name)
#
#     if model_type == 'pym':
#         # Load from kernel models
#         try:
#             krnl_models = importlib.import_module('middoe.krnl_models')
#             model = getattr(krnl_models, model_name)
#         except (ImportError, AttributeError) as e:
#             raise KeyError(f"Model '{model_name}' not found in middoe.krnl_models. Error: {e}")
#
#     elif model_type in ['pys', 'gpr']:
#         # Load from external script or gPROMS
#         try:
#             script_path = models['src'][model_name]
#             spec = importlib.util.spec_from_file_location("external_model", script_path)
#             module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(module)
#             model = getattr(module, 'solve_model')
#         except Exception as e:
#             raise KeyError(f"External model script for '{model_name}' could not be loaded. Error: {e}")
#
#     elif callable(model_type):
#         # ✅ This is your case: model is directly passed as a function
#         model = model_type
#
#     else:
#         raise KeyError(f"Model type for '{model_name}' not recognized or not callable.")
#
#     tv_ophi, ti_ophi, phit = _backscal(
#         t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac,
#         cvp, uphit, model, system, models, model_name
#     )
#     return tv_ophi, ti_ophi, phit
#
# def _Piecewiser(t, swps, cvp, phit):
#     """
#     Perform piecewise interpolation for trajectories of time-variant input variables.
#     """
#
#     result = {}
#     for var, method in cvp.items():
#         time_key = f'{var}t'
#         level_key = f'{var}l'
#
#         if method == 'CPF':
#             if time_key in swps and level_key in swps:
#                 if len(swps[time_key]) != len(swps[level_key]):
#                     raise ValueError(
#                         f"[Piecewiser-CPF] Length mismatch for '{var}': "
#                         f"len({time_key})={len(swps[time_key])}, len({level_key})={len(swps[level_key])}"
#                     )
#                 f_previous = interp1d(swps[time_key], swps[level_key], kind='previous', fill_value="extrapolate")
#                 result[var] = f_previous(t)
#             else:
#                 raise KeyError(f"Missing keys for CPF method: '{time_key}' or '{level_key}'")
#
#         elif method == 'LPF':
#             if time_key in swps and level_key in swps:
#                 if len(swps[time_key]) != len(swps[level_key]):
#                     raise ValueError(
#                         f"[Piecewiser-LPF] Length mismatch for '{var}': "
#                         f"len({time_key})={len(swps[time_key])}, len({level_key})={len(swps[level_key])}"
#                     )
#                 result[var] = np.interp(t, swps[time_key], swps[level_key])
#             else:
#                 raise KeyError(f"Missing keys for LPF method: '{time_key}' or '{level_key}'")
#
#         elif method == 'no_CVP':
#             if var not in phit:
#                 raise KeyError(f"No initial value provided in phit for variable '{var}' under 'none' method.")
#             value = phit[var]
#             if isinstance(value, (list, np.ndarray)):
#                 result[var] = np.full_like(t, value[0])
#             else:
#                 result[var] = np.full_like(t, value)
#
#         else:
#             raise ValueError(f"Unrecognized piecewise method '{method}' for variable '{var}'.")
#
#     return result
#
# def _backscal(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model, system, models, model_name):
#     """
#     Solve the ODE system defined by the model using solve_ivp.
#
#     Parameters
#     ----------
#     t : list or array of float
#         Time points for the simulation.
#     swps : dict
#         Switching points data for piecewise interpolation.
#     uphi : dict
#         Time-invariant variables (scaled to [0,1]).
#     uphisc : dict
#         Scaling factors for time-invariant variables.
#     uphitsc : dict
#         Scaling factors for time-variant variables.
#     utsc : float
#         Scaling factor for time.
#     utheta : list of float
#         Model parameters (unscaled).
#     uthetac : list of float
#         Scaling factors for parameters.
#     cvp : dict of str: str
#         Dictionary specifying interpolation methods per variable.
#     uphit : dict
#         Initial or constant values for time-variant variables when using 'none'.
#     model : callable
#         The model function to be integrated.
#     system : dict
#         Defines the structure of the model, including variable attributes.
#     models : dict
#         Additional modelling settings.
#
#     Returns
#     -------
#     tuple
#         tv_ophi (dict): {variable_name: ndarray of solution values over time}
#         ti_ophi (dict): {time_invariant_output_name: value} (currently empty)
#         tvi (dict): {variable_name: ndarray of interpolated input values}
#     """
#     # Scale time-invariant variables by their max values
#     phi = {k: v * uphisc[k] for k, v in uphi.items() if k in uphisc}
#
#     # Scale parameters
#     theta = [utheta[i] * uthetac[i] for i in range(len(utheta))]
#
#     # Compute piecewise interpolations and scale them
#     piecewise_data = _Piecewiser(t, swps, cvp, uphit)
#     phit = {k: piecewise_data[k] * uphitsc[k] for k in uphitsc}
#
#     # Construct initial conditions
#     y0 = []
#     for var in system['tvo'].keys():
#         var_attrs = system['tvo'][var]
#         initial_val = var_attrs['init']
#         if initial_val == 'variable':
#             # Use time-invariant variable with '_0' suffix
#             y0.append(phi[var + '_0'])
#         else:
#             # Use the specified initial value directly
#             y0.append(initial_val)
#
#     # Scale the time vector
#     t_scaled = (np.array(t) * utsc).tolist()
#
#     # Select the model and perform the integration
#     tv_ophi, ti_ophi, phit = solver_selector(model, t_scaled, y0, phi, phit, theta, models, model_name, system)
#
#
#
#     return tv_ophi, ti_ophi, phit
#
# def solver_selector(model, t, y0, phi, phit, theta, models, model_name, system):
#     """
#     General ODE simulator handler supporting:
#     - 'sci'      : Python-defined model and solve_ivp used directly
#     - 'sci_file' : External Python file containing solve_model(t, y0, tii, tvi, theta)
#     - 'gp'       : gPROMS/gPAS model via StartedConnected client
#
#     Returns
#     -------
#     tv_ophi : dict
#     ti_ophi : dict
#     tvi    : dict (unchanged)
#     """
#
#
#     max_retries = 1
#     retry_count = 0
#     success = False
#     tv_ophi, ti_ophi = {}, {}
#     sim_mode = models['krt'][model_name]
#
#     if sim_mode == 'pys':
#         while not success and retry_count < max_retries:
#             try:
#                 # Load external model dynamically
#                 script_path = models['src'][model_name]  # path to external .py file
#
#                 spec = importlib.util.spec_from_file_location("external_model", script_path)
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)
#
#                 external_solver_func = getattr(module, 'solve_model')
#                 external_result = external_solver_func(t, y0, phi, phit, theta)
#
#                 tv_ophi = external_result.get('tvo', {})
#                 ti_ophi = external_result.get('tio', {})
#                 success = True
#             except Exception as e:
#                 retry_count += 1
#                 print(f"SciPy file-based simulation attempt {retry_count} for model '{model_name}' failed: {e}")
#                 time.sleep(0.5)
#
#         if not success:
#             print(f"All {max_retries} SciPy file-based simulation attempts failed for model '{model_name}'")
#
#     elif sim_mode == 'gpr':
#         try:
#             from pygpas.evaluate import evaluate
#             from pygpas.server import StartedConnected
#         except ModuleNotFoundError:
#             raise ImportError(
#                 "The 'pygpas' module is required for gPROMS-based simulations. "
#                 "Please install it manually or ensure it's available in your environment."
#             )
#         while not success and retry_count < max_retries:
#             try:
#                 ports = random.randint(10000, 90000)
#                 with StartedConnected(port=str(ports), stdout=DEVNULL, stderr=DEVNULL) as client:
#                     client.open(str(models['src'][model_name]),
#                                 models['creds'][model_name])
#                     for key, value in phi.items():
#                         client.set_input_value(key, value.tolist() if hasattr(value, 'tolist') else value)
#                     for key, value in phit.items():
#                         client.set_input_value(key, value.tolist() if hasattr(value, 'tolist') else value)
#                     client.set_input_value('theta', theta.tolist() if hasattr(theta, 'tolist') else theta)
#                     client.set_input_value('y0', y0.tolist() if hasattr(y0, 'tolist') else y0)
#                     evaluate(client)
#                     tv_ophi = {
#                         key: client.get_value(key)
#                         for key in system.get('tvo', {})
#                     }
#                     ti_ophi = {
#                         key: client.get_value(key)
#                         for key in system.get('tio', {})
#                     }
#                     success = True
#             except Exception as e:
#                 retry_count += 1
#                 print(f"GP simulation attempt {retry_count} for model '{model_name}' failed: {e}")
#                 time.sleep(1)
#
#         if not success:
#             print(f"All {max_retries} GP simulation attempts failed for model '{model_name}'")
#
#     else:
#         while not success and retry_count < max_retries:
#             try:
#                 from scipy.integrate import solve_ivp
#                 result = solve_ivp(
#                     model,
#                     [min(t), max(t)],
#                     y0,
#                     method='LSODA',
#                     t_eval=t,
#                     args=(phi, phit, theta, t),
#                     rtol=1e-8,
#                     atol=1e-10,
#                 )
#                 if not result.success:
#                     raise RuntimeError(f"SciPy solver failed: {result.message}")
#                 tv_ophi = {f'y{i+1}': result.y[i] for i in range(result.y.shape[0])}
#                 ti_ophi = {}
#                 success = True
#             except Exception as e:
#                 retry_count += 1
#                 print(f"SciPy simulation attempt {retry_count} for model '{model_name}' failed: {e}")
#                 time.sleep(0.5)
#
#     return tv_ophi, ti_ophi, phit

# krnl_simula.py

import numpy as np
from scipy.interpolate import interp1d
import importlib
from subprocess import DEVNULL
import time
import random
import inspect
import importlib.util


class SimulationError(RuntimeError):
    """
    Raised when a model simulation fails or returns an invalid result.

    This covers an exception raised inside the user's model (the original
    exception is chained as ``__cause__`` and its message is included), a
    model file whose ``solve_model`` has the wrong signature, and a result
    with the wrong structure (missing ``'tvo'``, a missing measured output,
    wrong length or non-finite values).
    """


def _check_model_output(res, system, n, model_name=None):
    """
    Validate the result returned by an external ``solve_model`` ('pys').

    Requires a dict with a ``'tvo'`` dict that contains every *measured*
    time-variant output (``system['tvo'][k].get('meas', True)``), each of
    length ``n`` and with finite values. Unmeasured outputs may be omitted;
    if present they are checked as well. Returns ``(tv_ophi, ti_ophi)``.
    """
    who = f" for model '{model_name}'" if model_name is not None else ''
    if not isinstance(res, dict) or not isinstance(res.get('tvo'), dict):
        got = type(res).__name__
        if isinstance(res, dict):
            got += f" with keys {list(res)}"
        raise SimulationError(
            f"solve_model{who} must return a dict with a 'tvo' dict "
            f"({{'tvo': {{...}}, 'tio': {{...}}}}); got {got}")
    tvo = res['tvo']
    for k, cfg in system.get('tvo', {}).items():
        measured = cfg.get('meas', True) if isinstance(cfg, dict) else True
        if k not in tvo:
            if measured:
                raise SimulationError(
                    f"solve_model{who}: result['tvo'] is missing the measured output '{k}' "
                    f"(returned keys: {list(tvo)})")
            continue
        try:
            v = np.asarray(tvo[k], dtype=float)
        except (TypeError, ValueError) as e:
            raise SimulationError(f"solve_model{who}: result['tvo']['{k}'] is not numeric: {e}") from e
        if v.ndim != 1 or v.shape[0] != n:
            raise SimulationError(
                f"solve_model{who}: result['tvo']['{k}'] has shape {v.shape}, expected ({n},) "
                f"(one value per time point)")
        if not np.all(np.isfinite(v)):
            raise SimulationError(f"solve_model{who}: result['tvo']['{k}'] contains non-finite values")
    tio = res.get('tio', {})
    if tio is None:
        tio = {}
    if not isinstance(tio, dict):
        raise SimulationError(f"solve_model{who}: result['tio'] must be a dict, got {type(tio).__name__}")
    for k, cfg in system.get('tio', {}).items():
        measured = cfg.get('meas', True) if isinstance(cfg, dict) else True
        if k in tio:
            try:
                v = np.asarray(tio[k], dtype=float)
            except (TypeError, ValueError) as e:
                raise SimulationError(f"solve_model{who}: result['tio']['{k}'] is not numeric: {e}") from e
            if not np.all(np.isfinite(v)):
                raise SimulationError(f"solve_model{who}: result['tio']['{k}'] contains non-finite values")
    return tvo, tio


def _check_solve_model_signature(func, model_name, src=None):
    """Raise SimulationError unless ``func`` accepts ``(t, y0, tii, tvi, theta)``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return  # builtins / C functions: cannot inspect, let the call decide
    try:
        sig.bind(None, None, None, None, None)
    except TypeError as e:
        where = f" in {src}" if src else ''
        raise SimulationError(
            f"solve_model{where} (model '{model_name}') must accept 5 positional arguments "
            f"(t, y0, tii, tvi, theta); its signature is {sig}: {e}") from None


def simula(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model_name, system, models):
    r"""
    Execute forward model simulation with flexible solver and interpolation support.

    This is the main simulation kernel that orchestrates the complete forward modeling
    workflow. It handles model loading (internal, external Python, or gPROMS), variable
    scaling and interpolation, initial condition setup, and ODE integration. The function
    supports multiple control variable parameterization (CVP) methods and integrates
    seamlessly with parameter estimation and MBDoE routines.

    Parameters
    ----------
    t : list or np.ndarray
        Normalized time points for simulation [0, 1]. Will be scaled to physical time
        internally using utsc.
    swps : dict[str, np.ndarray]
        Switching points for piecewise interpolation of time-variant inputs:
            - '{var}t' : np.ndarray
                Switching times (normalized [0, 1]).
            - '{var}l' : np.ndarray
                Switching levels (normalized [0, 1]).
    uphi : dict[str, float]
        Time-invariant input variables (normalized [0, 1]).
    uphisc : dict[str, float]
        Scaling factors (maximum values) for time-invariant inputs.
    uphitsc : dict[str, float]
        Scaling factors (maximum values) for time-variant inputs.
    utsc : float
        Time scaling factor (physical time = normalized time * utsc).
    utheta : list[float]
        Normalized model parameters (typically [1, 1, ..., 1]).
    uthetac : list[float]
        Parameter scaling factors (true parameter values).
    cvp : dict[str, str]
        Control variable parameterization methods for each time-variant input:
            - 'CPF' : Constant piecewise function (zero-order hold).
            - 'LPF' : Linear piecewise function (linear interpolation).
            - 'no_CVP' : No control parameterization (constant value from uphit).
    uphit : dict[str, float or np.ndarray]
        Initial or constant values for time-variant inputs when CVP='no_CVP'.
    model_name : str
        Name of the model to simulate. Must exist in models['krt'].
    system : dict
        System configuration:
            - 'tvo' : dict
                Time-variant output definitions with:
                    * 'init' : float or 'variable'
                        Initial condition (value or reference to uphi key).
            - 'tio' : dict
                Time-invariant output definitions.
    models : dict
        Model registry and settings:
            - 'krt' : dict[str, str or callable]
                Model type/function mapping:
                    * 'pym' : Internal Python model (from krnl_models module)
                    * 'pys' : External Python script with solve_model()
                    * 'gpr' : gPROMS model via pygpas interface (nothing is
                      imported in Python; the .gPAS file is opened by pygpas)
                    * callable : Direct function reference
            - 'src' : dict[str, str], optional
                File paths for 'pys' (.py script) and 'gpr' (.gPAS file) models.
            - 'creds' : dict[str, str], optional
                Credentials for gPROMS models.

    Returns
    -------
    tv_ophi : dict[str, np.ndarray]
        Time-variant output trajectories:
            - Keys: output variable names (e.g., 'y1', 'y2', 'CA', 'T')
            - Values: arrays of shape (len(t),) with simulation results
    ti_ophi : dict[str, float]
        Time-invariant output values: the model's 'tio' for 'pys' and 'gpr'
        models; an empty dict for 'pym' models and callables.
    phit : dict[str, np.ndarray]
        Scaled and interpolated time-variant input profiles:
            - Keys: input variable names
            - Values: arrays of shape (len(t),) with input trajectories

    Raises
    ------
    KeyError
        If model_name is not found in models['krt'] or its kind is not recognised,
        if the model cannot be loaded ('pym' function missing from krnl_models,
        'pys' script not loadable), or if required keys are missing from swps or
        uphit for specified CVP methods.
    ImportError
        If pygpas is not installed for a 'gpr' model.
    ValueError
        If CVP method is unrecognized or if switching point arrays have mismatched lengths.
    SimulationError
        If the model raises, the integration fails, ``solve_model`` has the wrong
        signature, or the model output is invalid (see solver_selector).

    Notes
    -----
    **Simulation Workflow**:
        1. **Model Loading**: Determine model type and load model function
        2. **Variable Scaling**: Convert normalized inputs to physical units
        3. **Interpolation**: Construct time-variant input profiles via CVP
        4. **Initial Conditions**: Extract from system['tvo'] definitions
        5. **Integration**: Call solver_selector() with appropriate backend
        6. **Return**: Scaled outputs and interpolated inputs

    **Model Types**:
        - **'pym' (Python internal)**: Model function defined in middoe.krnl_models.
          Loaded via importlib and called directly.

        - **'pys' (Python external)**: Model defined in external .py file.
          Must contain solve_model(t, y0, tii, tvi, theta) function.
          Loaded dynamically via importlib.util.

        - **'gpr' (gPROMS)**: Model exported as a gPROMS .gPAS file
          (models['src'][model_name], password in models['creds'][model_name]).
          Nothing is imported in Python; solver_selector opens the file through
          pygpas and evaluates it via the gPAS client-server architecture.

        - **callable**: Model function passed directly in models['krt'][model_name].
          Most flexible option for custom models.

    **Control Variable Parameterization (CVP)**:
        - **CPF (Constant Piecewise)**: Zero-order hold interpolation.
          Input held constant between switching points (step changes).
          \[
          u(t) = u_k \quad \text{for } t \in [t_k, t_{k+1})
          \]

        - **LPF (Linear Piecewise)**: Linear interpolation between switching points.
          Smooth transitions create linear ramps.
          \[
          u(t) = u_k + (u_{k+1} - u_k) \frac{t - t_k}{t_{k+1} - t_k}
          \]

        - **no_CVP**: No control optimization. Input remains constant at uphit value.

    **Normalization Convention**:
    All variables, parameters, and time are normalized internally:

        - Variables: \( \phi_{physical} = \phi_{norm} \cdot \phi_{max} \)
        - Parameters: \( \theta_{physical} = \theta_{norm} \cdot \theta_c \)
        - Time: \( t_{physical} = t_{norm} \cdot t_{scale} \)

    This improves numerical conditioning and allows unified handling across different
    scales. Scaling is reversed internally before calling the physical model.

    **Initial Conditions**:
    Each time-variant output can specify:

        - Numeric value: Used directly as \( y_0 \)
        - 'variable': References uphi['{var}_0'] (e.g., 'CA_0' for initial concentration)

    **Integration Backend**:
    solver_selector() handles ODE integration using:

        - SciPy solve_ivp (LSODA) for Python models
        - Custom solve_model() for external scripts
        - gPAS evaluate() for gPROMS models

    **Error Handling**:
    A failing or invalid model raises :class:`SimulationError` with the model's own
    error message (see solver_selector); simulation failures are not silenced.

    References
    ----------
    1. Hairer, E., Nørsett, S. P., & Wanner, G. (1993).
       *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.

    2. gPROMS documentation: www.psenterprise.com/products/gproms

    See Also
    --------
    _backscal : Internal function that performs scaling and calls solver_selector.
    _Piecewiser : Constructs piecewise interpolated input profiles.
    solver_selector : Backend-specific integration routine.
    expera : Uses this function for in-silico experiment generation.
    parmest : Uses this function during parameter estimation.

    Examples
    --------
    >>> # Basic simulation with internal Python model
    >>> t = np.linspace(0, 1, 101)
    >>> swps = {'Tt': np.array([0, 0.5, 1]), 'Tl': np.array([300, 350, 300])}
    >>> uphi = {'P': 0.8, 'CA_0': 0.5}
    >>> uphisc = {'P': 5.0, 'CA_0': 2.0}
    >>> uphitsc = {'T': 400}
    >>> utheta, uthetac = [1, 1, 1], [1.5e5, 2.3, 0.8]
    >>> cvp = {'T': 'LPF'}
    >>> tv_out, ti_out, tv_in = simula(
    ...     t, swps, uphi, uphisc, uphitsc, 100, utheta, uthetac,
    ...     cvp, {}, 'CSTR_model', system, models
    ... )
    >>> print(tv_out.keys())  # ['CA', 'CB', 'T']
    >>> print(tv_in['T'])  # Temperature profile [300, ..., 350, ..., 300]

    >>> # External Python model
    >>> models = {
    ...     'krt': {'my_model': 'pys'},
    ...     'src': {'my_model': '/path/to/model.py'}
    ... }
    >>> tv_out, ti_out, tv_in = simula(t, {}, uphi, uphisc, uphitsc,
    ...                                  100, utheta, uthetac, {}, uphit,
    ...                                  'my_model', system, models)

    >>> # gPROMS model
    >>> models = {
    ...     'krt': {'gproms_model': 'gpr'},
    ...     'src': {'gproms_model': '/path/to/model.gPAS'},
    ...     'creds': {'gproms_model': 'password'}
    ... }
    >>> tv_out, ti_out, tv_in = simula(t, swps, uphi, uphisc, uphitsc,
    ...                                  100, utheta, uthetac, cvp, {},
    ...                                  'gproms_model', system, models)
    """
    model_type = models['krt'].get(model_name)

    if model_type == 'pym':
        # Load from kernel models
        try:
            krnl_models = importlib.import_module('middoe.krnl_models')
            model = getattr(krnl_models, model_name)
        except (ImportError, AttributeError) as e:
            raise KeyError(f"Model '{model_name}' not found in middoe.krnl_models. Error: {e}")

    elif model_type == 'pys':
        # Load the external Python script
        try:
            script_path = models['src'][model_name]
            spec = importlib.util.spec_from_file_location("external_model", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module, 'solve_model')
        except Exception as e:
            raise KeyError(f"External model script for '{model_name}' could not be loaded. Error: {e}")
        _check_solve_model_signature(model, model_name, models['src'][model_name])

    elif model_type == 'gpr':
        # gPROMS: solver_selector opens models['src'][model_name] (the .gPAS file) with
        # models['creds'][model_name] through pygpas; there is no Python module to load here.
        model = None

    elif callable(model_type):
        # Direct function reference
        model = model_type

    else:
        raise KeyError(f"Model type for '{model_name}' not recognized or not callable.")

    tv_ophi, ti_ophi, phit = _backscal(
        t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac,
        cvp, uphit, model, system, models, model_name
    )
    return tv_ophi, ti_ophi, phit


def _Piecewiser(t, swps, cvp, phit):
    r"""
    Construct piecewise interpolated trajectories for time-variant input variables.

    This function implements control variable parameterization (CVP) by creating
    continuous-time input profiles from discrete switching points. It supports
    zero-order hold (constant piecewise), linear interpolation, and constant
    (no control) trajectories.

    Parameters
    ----------
    t : np.ndarray
        Normalized time points [0, 1] at which to evaluate interpolations.
    swps : dict[str, np.ndarray]
        Switching points for piecewise functions:
            - '{var}t' : np.ndarray
                Switching times (must be sorted, values in [0, 1]).
            - '{var}l' : np.ndarray
                Switching levels (same length as '{var}t').
    cvp : dict[str, str]
        Interpolation method for each variable:
            - 'CPF' : Constant piecewise function (zero-order hold)
            - 'LPF' : Linear piecewise function
            - 'no_CVP' : No interpolation (constant from phit)
    phit : dict[str, float or np.ndarray]
        Constant values for variables with cvp='no_CVP'.

    Returns
    -------
    result : dict[str, np.ndarray]
        Interpolated trajectories for each variable:
            - Keys: variable names
            - Values: arrays of shape (len(t),) with interpolated values

    Raises
    ------
    KeyError
        If required switching point keys are missing for CPF/LPF methods, or if
        constant value is missing for no_CVP method.
    ValueError
        If switching time and level arrays have mismatched lengths, or if CVP
        method is unrecognized.

    Notes
    -----
    **Constant Piecewise Function (CPF)**:
    Zero-order hold interpolation. Input held constant between switching points:
        \[
        u(t) = u_k \quad \text{for } t \in [t_k, t_{k+1})
        \]
    Implemented with ``np.searchsorted(..., side='right')`` on the sorted switching
    times. A time point within a relative tolerance of 1e-9 of a switching time
    takes the new level, so a switch time that differs from a grid node only by
    floating-point round-off (e.g. ``(t_k * tf) / tf``) does not lag one step.

    **Linear Piecewise Function (LPF)**:
    Linear interpolation between switching points:
        \[
        u(t) = u_k + (u_{k+1} - u_k) \frac{t - t_k}{t_{k+1} - t_k}
        \]
    Implemented using np.interp (linear interpolation with extrapolation).

    **No Control (no_CVP)**:
    Input remains constant at initial value from phit:
        \[
        u(t) = u_0 \quad \forall t
        \]

    **Extrapolation**:
    Beyond the range of switching points:
        - After the last switching point: both CPF and LPF use the last level.
        - Before the first switching point: LPF uses the first level; CPF returns
          NaN (as ``interp1d(kind='previous')`` did), so the first switching time
          should be at t = 0.

    **Use in MBDoE**:
    Switching points are optimization variables in MBDoE for parameter precision.
    The optimizer adjusts '{var}t' and '{var}l' to maximize information content
    while respecting constraints on rate of change and feasibility.

    **Switching Point Convention**:
    - Times must be sorted in ascending order
    - Levels can be arbitrary (within physical bounds)
    - First switching point typically at t=0 (initial condition)
    - Last switching point typically at t=1 (final condition)

    See Also
    --------
    simula : Main function that calls this for input interpolation.
    numpy.searchsorted : Used for CPF interpolation.
    numpy.interp : Used for LPF interpolation.

    Examples
    --------
    >>> t = np.linspace(0, 1, 11)
    >>> swps = {
    ...     'Tt': np.array([0, 0.3, 0.7, 1.0]),
    ...     'Tl': np.array([300, 350, 350, 300])
    ... }
    >>> cvp = {'T': 'LPF'}
    >>> result = _Piecewiser(t, swps, cvp, {})
    >>> print(result['T'])
    # Linear ramp: [300, 315, 330, 345, 350, 350, 350, 340, 320, 310, 300]

    >>> # Constant piecewise (step changes)
    >>> cvp = {'T': 'CPF'}
    >>> result = _Piecewiser(t, swps, cvp, {})
    >>> print(result['T'])
    # Steps: [300, 300, 300, 350, 350, 350, 350, 300, 300, 300, 300]

    >>> # No control (constant)
    >>> cvp = {'T': 'no_CVP'}
    >>> result = _Piecewiser(t, swps, cvp, {'T': 325})
    >>> print(result['T'])
    # [325, 325, 325, ..., 325]
    """
    result = {}
    for var, method in cvp.items():
        time_key = f'{var}t'
        level_key = f'{var}l'

        if method == 'CPF':
            if time_key in swps and level_key in swps:
                if len(swps[time_key]) != len(swps[level_key]):
                    raise ValueError(
                        f"[Piecewiser-CPF] Length mismatch for '{var}': "
                        f"len({time_key})={len(swps[time_key])}, len({level_key})={len(swps[level_key])}"
                    )
                # Zero-order hold with a switch-time tolerance. interp1d(kind='previous') gave the
                # OLD level at a grid node when the switch time was the node + 1 ulp (e.g. after a
                # physical -> normalised round trip t*tf/tf), so the generated data and the fit
                # used profiles shifted by one grid step. Times are sorted (as interp1d did) and a
                # node within 1e-9 (relative) of a switch time takes the NEW level.
                ts = np.asarray(swps[time_key], dtype=float).ravel()
                ls = np.asarray(swps[level_key], dtype=float).ravel()
                order = np.argsort(ts, kind='stable')
                ts, ls = ts[order], ls[order]
                tt = np.asarray(t, dtype=float)
                tol = 1e-9 * max(1.0, float(np.max(np.abs(tt))) if tt.size else 1.0,
                                 float(np.max(np.abs(ts))) if ts.size else 1.0)
                k = np.searchsorted(ts, tt + tol, side='right') - 1
                vals = ls[np.clip(k, 0, len(ls) - 1)]
                # Before the first switch time interp1d(kind='previous') returned NaN; keep that.
                result[var] = np.where(k < 0, np.nan, vals)
            else:
                raise KeyError(f"Missing keys for CPF method: '{time_key}' or '{level_key}'")

        elif method == 'LPF':
            if time_key in swps and level_key in swps:
                if len(swps[time_key]) != len(swps[level_key]):
                    raise ValueError(
                        f"[Piecewiser-LPF] Length mismatch for '{var}': "
                        f"len({time_key})={len(swps[time_key])}, len({level_key})={len(swps[level_key])}"
                    )
                result[var] = np.interp(t, swps[time_key], swps[level_key])
            else:
                raise KeyError(f"Missing keys for LPF method: '{time_key}' or '{level_key}'")

        elif method == 'no_CVP':
            if var not in phit:
                raise KeyError(f"No initial value provided in phit for variable '{var}' under 'none' method.")
            value = phit[var]
            if isinstance(value, (list, np.ndarray)):
                result[var] = np.full_like(t, value[0])
            else:
                result[var] = np.full_like(t, value)

        else:
            raise ValueError(f"Unrecognized piecewise method '{method}' for variable '{var}'.")

    return result


def _backscal(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model, system, models, model_name):
    r"""
    Scale normalized inputs to physical units and prepare for ODE integration.

    This internal function reverses the normalization applied for optimization,
    converting all normalized variables, parameters, and time back to physical units.
    It also constructs initial conditions from system definitions and calls the
    appropriate solver backend.

    Parameters
    ----------
    t : np.ndarray
        Normalized time points [0, 1].
    swps : dict
        Normalized switching points for interpolation.
    uphi : dict[str, float]
        Normalized time-invariant inputs [0, 1].
    uphisc : dict[str, float]
        Scaling factors (max values) for time-invariant inputs.
    uphitsc : dict[str, float]
        Scaling factors (max values) for time-variant inputs.
    utsc : float
        Time scaling factor.
    utheta : list[float]
        Normalized parameters.
    uthetac : list[float]
        Parameter scaling factors.
    cvp : dict[str, str]
        Control variable parameterization methods.
    uphit : dict[str, float]
        Constant values for no_CVP variables.
    model : callable
        Model function to integrate.
    system : dict
        System configuration with initial condition specifications.
    models : dict
        Model registry and settings.
    model_name : str
        Name of the model being simulated.

    Returns
    -------
    tv_ophi : dict[str, np.ndarray]
        Time-variant output trajectories (physical units).
    ti_ophi : dict[str, float]
        Time-invariant output values (physical units).
    tvi : dict[str, np.ndarray]
        Interpolated time-variant input profiles (physical units).

    Notes
    -----
    **Scaling Operations**:
        1. **Time-invariant inputs**: \( \phi = \phi_{norm} \cdot \phi_{max} \)
        2. **Parameters**: \( \theta = \theta_{norm} \cdot \theta_c \)
        3. **Time**: \( t = t_{norm} \cdot t_{scale} \)
        4. **Time-variant inputs**: \( \phi_t = \phi_{t,norm} \cdot \phi_{t,max} \)

    **Initial Condition Construction**:
    For each output variable in system['tvo']:
        - If init='variable': Use uphi['{var}_0'] (e.g., 'CA_0' for initial CA)
        - Otherwise: Use numeric value directly

    **Workflow**:
        1. Scale all inputs to physical units
        2. Construct piecewise input profiles via _Piecewiser()
        3. Extract initial conditions from system definitions
        4. Scale time vector to physical units
        5. Call solver_selector() with physical quantities
        6. Return results in physical units

    **Why Normalization?**:
    Working in normalized coordinates [0, 1] during optimization:
        - Improves numerical conditioning
        - Allows bounds-free optimization
        - Simplifies constraint handling
        - Makes optimizers more robust

    Physical units are only restored immediately before model evaluation.

    See Also
    --------
    simula : Main function that calls this.
    _Piecewiser : Constructs interpolated input profiles.
    solver_selector : Performs actual ODE integration.

    Examples
    --------
    >>> # Typically called internally by simula()
    >>> # Scaling example:
    >>> uphi_norm = {'P': 0.6}  # Normalized pressure
    >>> uphisc = {'P': 5.0}  # Max pressure = 5 bar
    >>> phi_physical = uphi_norm['P'] * uphisc['P']  # 3.0 bar
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
    for var in system['tvo'].keys():
        var_attrs = system['tvo'][var]
        initial_val = var_attrs['init']
        if initial_val == 'variable':
            # Use time-invariant variable with '_0' suffix
            y0.append(phi[var + '_0'])
        else:
            # Use the specified initial value directly
            y0.append(initial_val)

    # Scale the time vector
    t_scaled = (np.array(t) * utsc).tolist()

    # Select the model and perform the integration
    tv_ophi, ti_ophi, phit = solver_selector(model, t_scaled, y0, phi, phit, theta, models, model_name, system)

    return tv_ophi, ti_ophi, phit


def solver_selector(model, t, y0, phi, phit, theta, models, model_name, system):
    r"""
    Select and execute the appropriate ODE solver backend.

    This function dispatches to the correct integration backend based on model type,
    handling SciPy solve_ivp, external Python scripts, and gPROMS models. Model
    failures and invalid model outputs raise :class:`SimulationError`.

    Parameters
    ----------
    model : callable
        Model function to integrate. Signature depends on backend:
            - SciPy: dydt = model(t, y, phi, phit, theta, t_vec)
            - External: result = model(t, y0, phi, phit, theta)
    t : list[float]
        Physical time points for evaluation.
    y0 : list[float]
        Initial conditions for ODE states.
    phi : dict[str, float]
        Time-invariant inputs (physical units).
    phit : dict[str, np.ndarray]
        Time-variant input profiles (physical units).
    theta : list[float]
        Model parameters (physical units).
    models : dict
        Model registry:
            - 'krt' : dict[str, str]
                Model types ('pys', 'gpr', or callable).
            - 'src' : dict[str, str]
                File paths for external models.
            - 'creds' : dict[str, str]
                Credentials for gPROMS models.
    model_name : str
        Name of model being simulated.
    system : dict
        System configuration with output variable definitions.

    Returns
    -------
    tv_ophi : dict[str, np.ndarray]
        Time-variant output trajectories.
    ti_ophi : dict[str, float]
        Time-invariant output values.
    tvi : dict[str, np.ndarray]
        Time-variant input profiles (unchanged, passed through).

    Notes
    -----
    **Solver Backends**:
        1. **'pys' (External Python Script)**:

           - Dynamically loads .py file from models['src'][model_name]
           - Calls solve_model(t, y0, tii, tvi, theta)
           - Returns {'tvo': {...}, 'tio': {...}}

        2. **'gpr' (gPROMS)**:

           - Requires pygpas package
           - Opens .gPJ file via StartedConnected client
           - Sets inputs via client.set_input_value()
           - Evaluates model via evaluate(client)
           - Retrieves outputs via client.get_value()

        3. **Default (SciPy solve_ivp)**:

           - Uses LSODA method (automatic stiffness detection)
           - Integrates from min(t) to max(t)
           - Evaluates at specified time points (t_eval=t)
           - Returns {'y1': sol[0], 'y2': sol[1], ...}

    **Output Checks** ('pys'):
    The result must be a dict with a ``'tvo'`` dict holding every *measured*
    output of ``system['tvo']`` (``meas`` not False), each with ``len(t)``
    finite values. Unmeasured outputs may be omitted. ``solve_model`` must
    accept ``(t, y0, tii, tvi, theta)``. For SciPy-integrated models the
    outputs must have ``len(t)`` finite values.

    **SciPy Integration Settings**:
        - Method: LSODA (stiff/non-stiff automatic switching)
        - rtol: 1e-8 (relative tolerance)
        - atol: 1e-10 (absolute tolerance)
        - t_span: [min(t), max(t)]
        - t_eval: t (evaluation points)

    **gPROMS Connection**:
        - Random port selection (10000-90000) to avoid conflicts
        - Silent mode (stdout/stderr suppressed)
        - Automatic connection teardown via context manager
        - Requires valid credentials in models['creds']

    **Error Handling**:
    Any failure raises :class:`SimulationError` (a ``RuntimeError``): an
    exception raised inside the user's model (chained as ``__cause__``, with its
    message included), a failed SciPy integration, a failed gPROMS evaluation, a
    wrong ``solve_model`` signature or an invalid result. Nothing is printed and
    no empty result is returned. A model script that cannot be loaded raises
    ``KeyError`` as before.

    **Output Naming Convention**:
    For SciPy models, outputs are automatically named 'y1', 'y2', etc.
    based on position in state vector. For external/gPROMS models,
    output names must match keys in system['tvo'] and system['tio'].

    References
    ----------
    1. Hindmarsh, A. C. (1983).
       ODEPACK, A Systematized Collection of ODE Solvers.
       *Scientific Computing*, 55-64. North-Holland.

    2. gPROMS documentation: www.psenterprise.com/products/gproms

    See Also
    --------
    simula : Main entry point.
    _backscal : Prepares inputs and calls this function.
    scipy.integrate.solve_ivp : SciPy ODE solver.

    Examples
    --------
    >>> # Typically called internally by _backscal()
    >>> # SciPy backend (default)
    >>> def my_model(t_curr, y, tii, tvi, theta, t_vec):
    ...     dydt = [-theta[0] * y[0] * tii['CA'], theta[0] * y[0] * tii['CA']]
    ...     return dydt
    >>> tv_out, ti_out, tv_in = solver_selector(
    ...     my_model, [0, 50, 100], [1.0, 0.0],
    ...     {'CA': 2.0}, {'T': T_profile}, [1.5e5, 2.3],
    ...     models, 'my_model', system
    ... )
    >>> print(tv_out.keys())  # ['y1', 'y2']
    """
    sim_mode = models['krt'][model_name]
    n_t = len(t)

    if sim_mode == 'pys':
        # Load external model dynamically. Loading problems are reported as KeyError
        # (as in simula); a wrong signature or an invalid result as SimulationError;
        # an exception raised by the user's model is re-raised as SimulationError with
        # the original exception chained (no print, no sleep, no silent {}).
        script_path = models['src'][model_name]
        try:
            spec = importlib.util.spec_from_file_location("external_model", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            external_solver_func = getattr(module, 'solve_model')
        except Exception as e:
            raise KeyError(f"External model script for '{model_name}' could not be loaded. Error: {e}")
        _check_solve_model_signature(external_solver_func, model_name, script_path)

        try:
            external_result = external_solver_func(t, y0, phi, phit, theta)
        except SimulationError:
            raise
        except Exception as e:
            raise SimulationError(
                f"Model '{model_name}' raised {type(e).__name__}: {e}  (model file: {script_path})") from e
        tv_ophi, ti_ophi = _check_model_output(external_result, system, n_t, model_name)

    elif sim_mode == 'gpr':
        try:
            from pygpas.evaluate import evaluate
            from pygpas.server import StartedConnected
        except ModuleNotFoundError:
            raise ImportError(
                "The 'pygpas' module is required for gPROMS-based simulations. "
                "Please install it manually or ensure it's available in your environment."
            )
        max_retries = 1
        retry_count = 0
        success = False
        last_exc = None
        tv_ophi, ti_ophi = {}, {}
        while not success and retry_count < max_retries:
            try:
                ports = random.randint(10000, 90000)
                with StartedConnected(port=str(ports), stdout=DEVNULL, stderr=DEVNULL) as client:
                    client.open(str(models['src'][model_name]),
                                models['creds'][model_name])
                    for key, value in phi.items():
                        client.set_input_value(key, value.tolist() if hasattr(value, 'tolist') else value)
                    for key, value in phit.items():
                        client.set_input_value(key, value.tolist() if hasattr(value, 'tolist') else value)
                    client.set_input_value('theta', theta.tolist() if hasattr(theta, 'tolist') else theta)
                    client.set_input_value('y0', y0.tolist() if hasattr(y0, 'tolist') else y0)
                    evaluate(client)
                    tv_ophi = {
                        key: client.get_value(key)
                        for key in system.get('tvo', {})
                    }
                    ti_ophi = {
                        key: client.get_value(key)
                        for key in system.get('tio', {})
                    }
                    success = True
            except Exception as e:
                retry_count += 1
                last_exc = e
                if retry_count < max_retries:
                    time.sleep(1)

        if not success:
            raise SimulationError(
                f"gPROMS simulation of model '{model_name}' failed after {max_retries} attempt(s): "
                f"{type(last_exc).__name__}: {last_exc}") from last_exc

    else:
        # 'pym' (middoe.krnl_models) or a callable: model is the ODE right-hand side.
        from scipy.integrate import solve_ivp
        try:
            result = solve_ivp(
                model,
                [min(t), max(t)],
                y0,
                method='LSODA',
                t_eval=t,
                args=(phi, phit, theta, t),
                rtol=1e-8,
                atol=1e-10,
            )
        except Exception as e:
            raise SimulationError(
                f"Model '{model_name}' raised {type(e).__name__}: {e}") from e
        if not result.success:
            raise SimulationError(f"SciPy solver failed for model '{model_name}': {result.message}")
        tv_ophi = {f'y{i+1}': result.y[i] for i in range(result.y.shape[0])}
        ti_ophi = {}
        # Output check: states are named y1..yn by position, so only length/finiteness
        # can be checked here (missing names are reported by the caller).
        for k, v in tv_ophi.items():
            if len(v) != n_t:
                raise SimulationError(
                    f"Model '{model_name}': output '{k}' has {len(v)} values, expected {n_t}")
            if not np.all(np.isfinite(v)):
                raise SimulationError(f"Model '{model_name}': output '{k}' contains non-finite values")

    return tv_ophi, ti_ophi, phit
