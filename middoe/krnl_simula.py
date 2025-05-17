import numpy as np
from scipy.interpolate import interp1d
import importlib
from subprocess import DEVNULL
import time
import random
import importlib.util


def simula(t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac, cvp, uphit, model_name, system, models):
    """
    Simulate the system dynamics using the provided parameters and model.

    Parameters
    ----------
    t : list or array of float
        Time points for the simulation.
    swps : dict of str: list
        Dictionary containing time and level vectors for piecewise interpolation.
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
    system : dict
        Defines model structure, including variables, attributes, and initial conditions.
    models : dict
        Settings for the modelling process, including external functions.

    Returns
    -------
    tuple
        tv_ophi (dict): Time-varying outputs {var_name: ndarray}
        ti_ophi (dict): Time-invariant outputs (currently empty)
        tvi (dict): Scaled piecewise interpolated data {var_name: ndarray}
    """

    # # Determine model source
    # if model_name in models.get('ext_func', {}):
    #     model = models['ext_func'][model_name]
    # else:
    #     try:
    #         krnl_models = importlib.import_module('middoe.krnl_models')
    #         model = getattr(krnl_models, model_name)
    #     except (ImportError, AttributeError):
    #         try:
    #             # Try to load from external file path (sci_file mode)
    #             script_path = models['exfiles']['connector'][model_name]
    #             spec = importlib.util.spec_from_file_location("external_model", script_path)
    #             module = importlib.util.module_from_spec(spec)
    #             spec.loader.exec_module(module)
    #             model = getattr(module, 'solve_model')
    #         except Exception as e:
    #             raise KeyError(f"Model function '{model_name}' not found in 'ext_func', 'kernel_models', or external file. Error: {e}")

    model_type = models['krt'].get(model_name)

    if model_type == 'pym':
        # Load from kernel models
        try:
            krnl_models = importlib.import_module('middoe.krnl_models')
            model = getattr(krnl_models, model_name)
        except (ImportError, AttributeError) as e:
            raise KeyError(f"Model '{model_name}' not found in middoe.krnl_models. Error: {e}")

    elif model_type in ['pys', 'gpr']:
        # Load from external script or gPROMS
        try:
            script_path = models['src'][model_name]
            spec = importlib.util.spec_from_file_location("external_model", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module, 'solve_model')
        except Exception as e:
            raise KeyError(f"External model script for '{model_name}' could not be loaded. Error: {e}")

    elif callable(model_type):
        # ✅ This is your case: model is directly passed as a function
        model = model_type

    else:
        raise KeyError(f"Model type for '{model_name}' not recognized or not callable.")

    # # Determine model source based on model type
    # model_type = models['krt'].get(model_name)
    #
    # if model_type == 'pym':
    #     # Load from MIDDoE internal kernel models
    #     try:
    #         krnl_models = importlib.import_module('middoe.krnl_models')
    #         model = getattr(krnl_models, model_name)
    #     except (ImportError, AttributeError) as e:
    #         raise KeyError(f"Model '{model_name}' not found in middoe.krnl_models. Error: {e}")
    #
    # elif model_type in ['pys', 'gpr']:
    #     # Load from external file path (script-based or GPR models)
    #     try:
    #         script_path = models['src'][model_name]
    #         spec = importlib.util.spec_from_file_location("external_model", script_path)
    #         module = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(module)
    #         model = getattr(module, 'solve_model')
    #     except Exception as e:
    #         raise KeyError(f"External model script for '{model_name}' could not be loaded. Error: {e}")
    #
    # else:
    #     # Assume the model is already provided as a callable (e.g., function handle)
    #     try:
    #         model = models[model_name]
    #     except KeyError:
    #         raise KeyError(f"Model '{model_name}' not found or not callable. Check models dictionary.")

    tv_ophi, ti_ophi, phit = _backscal(
        t, swps, uphi, uphisc, uphitsc, utsc, utheta, uthetac,
        cvp, uphit, model, system, models, model_name
    )
    return tv_ophi, ti_ophi, phit

def _Piecewiser(t, swps, cvp, phit):
    """
    Perform piecewise interpolation for trajectories of time-variant input variables.
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
                f_previous = interp1d(swps[time_key], swps[level_key], kind='previous', fill_value="extrapolate")
                result[var] = f_previous(t)
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
    system : dict
        Defines the structure of the model, including variable attributes.
    models : dict
        Additional modelling settings.

    Returns
    -------
    tuple
        tv_ophi (dict): {variable_name: ndarray of solution values over time}
        ti_ophi (dict): {time_invariant_output_name: value} (currently empty)
        tvi (dict): {variable_name: ndarray of interpolated input values}
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
    """
    General ODE simulator handler supporting:
    - 'sci'      : Python-defined model and solve_ivp used directly
    - 'sci_file' : External Python file containing solve_model(t, y0, tii, tvi, theta)
    - 'gp'       : gPROMS/gPAS model via StartedConnected client

    Returns
    -------
    tv_ophi : dict
    ti_ophi : dict
    tvi    : dict (unchanged)
    """


    max_retries = 10
    retry_count = 0
    success = False
    tv_ophi, ti_ophi = {}, {}
    sim_mode = models['krt'][model_name]

    if sim_mode == 'pys':
        while not success and retry_count < max_retries:
            try:
                # Load external model dynamically
                script_path = models['src'][model_name]  # path to external .py file

                spec = importlib.util.spec_from_file_location("external_model", script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                external_solver_func = getattr(module, 'solve_model')
                external_result = external_solver_func(t, y0, phi, phit, theta)

                tv_ophi = external_result.get('tvo', {})
                ti_ophi = external_result.get('tio', {})
                success = True
            except Exception as e:
                retry_count += 1
                print(f"SciPy file-based simulation attempt {retry_count} for model '{model_name}' failed: {e}")
                time.sleep(0.5)

        if not success:
            print(f"All {max_retries} SciPy file-based simulation attempts failed for model '{model_name}'")

    elif sim_mode == 'gpr':
        try:
            from pygpas.evaluate import evaluate
            from pygpas.server import StartedConnected
        except ModuleNotFoundError:
            raise ImportError(
                "The 'pygpas' module is required for gPROMS-based simulations. "
                "Please install it manually or ensure it's available in your environment."
            )
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
                print(f"GP simulation attempt {retry_count} for model '{model_name}' failed: {e}")
                time.sleep(1)

        if not success:
            print(f"All {max_retries} GP simulation attempts failed for model '{model_name}'")

    else:
        while not success and retry_count < max_retries:
            try:
                from scipy.integrate import solve_ivp
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
                if not result.success:
                    raise RuntimeError(f"SciPy solver failed: {result.message}")
                tv_ophi = {f'y{i+1}': result.y[i] for i in range(result.y.shape[0])}
                ti_ophi = {}
                success = True
            except Exception as e:
                retry_count += 1
                print(f"SciPy simulation attempt {retry_count} for model '{model_name}' failed: {e}")
                time.sleep(0.5)

    return tv_ophi, ti_ophi, phit
