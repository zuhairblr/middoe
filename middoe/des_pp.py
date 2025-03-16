import os
from middoe.des_utils import _segmenter, _slicer, _reporter, _par_update, build_var_groups, build_linear_constraints, penalized_objective, constraint_violation
from middoe.krnl_simula import Simula
from collections import defaultdict
from functools import partial
import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds


def PP(design_settings, model_structure, modelling_settings, core_number, framework_settings, round):
    """
    Perform MBDOE for Parameter Precision (MBDOE_PP) with a structure parallel
    to des-md's MD(...) function.
    """

    # --------------------------- EXTRACT DATA --------------------------- #
    tf = model_structure['t_s'][1]
    ti = model_structure['t_s'][0]

    # Time-variant inputs
    tv_iphi_vars = list(model_structure['tv_iphi'].keys())
    tv_iphi_max = [model_structure['tv_iphi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [model_structure['tv_iphi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [model_structure['tv_iphi'][var]['swp'] for var in tv_iphi_vars]
    tv_iphi_const = [model_structure['tv_iphi'][var]['constraints'] for var in tv_iphi_vars]
    tv_iphi_offsett = [model_structure['tv_iphi'][var]['offsett'] for var in tv_iphi_vars]
    tv_iphi_offsetl = [model_structure['tv_iphi'][var]['offsetl'] for var in tv_iphi_vars]
    tv_iphi_cvp = {var: model_structure['tv_iphi'][var]['design_cvp'] for var in tv_iphi_vars}

    # Time-invariant inputs
    ti_iphi_vars = list(model_structure['ti_iphi'].keys())
    ti_iphi_max = [model_structure['ti_iphi'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [model_structure['ti_iphi'][var]['min'] for var in ti_iphi_vars]

    # Time-variant outputs
    tv_ophi_vars = [
        var for var in model_structure['tv_ophi'].keys()
        if model_structure['tv_ophi'][var].get('measured', True)
    ]
    tv_ophi_seg = [model_structure['tv_ophi'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett_ophi = [model_structure['tv_ophi'][var]['offsett'] for var in tv_ophi_vars]
    tv_ophi_matching = [model_structure['tv_ophi'][var]['matching'] for var in tv_ophi_vars]

    # Time-invariant outputs
    ti_ophi_vars = [
        var for var in model_structure['ti_ophi'].keys()
        if model_structure['ti_ophi'][var].get('measured', True)
    ]

    # Solver and optimization settings
    active_solvers = modelling_settings['active_solvers']
    estimations = modelling_settings['normalized_parameters']
    ref_thetas = modelling_settings['theta_parameters']
    theta_parameters = _par_update(ref_thetas, estimations)

    # Criterion, iteration, and penalty settings
    design_criteria = design_settings['criteria']['MBDOE_PP_criterion']
    nd = design_settings['iteration_settings']['nd']
    maxpp = design_settings['iteration_settings']['maxpp']
    tolpp = design_settings['iteration_settings']['tolpp']
    eps = design_settings['eps']
    mutation = modelling_settings['mutation']
    V_matrix = modelling_settings['V_matrix']

    # ------------------------ CHOOSE METHOD (Local vs Global) ------------------------ #
    method_key = design_settings['optimization_methods']['mdopt_method']
    if method_key == 'Local':
        result, index_dict = _optimizel(
            tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
            tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
            ti_iphi_vars, ti_iphi_max, ti_iphi_min,
            tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
            ti_ophi_vars,
            tf, ti,
            active_solvers, theta_parameters,
            nd, eps, maxpp, tolpp,
            mutation, V_matrix, design_criteria,
            model_structure, modelling_settings
        )
    elif method_key == 'Global':
        result, index_dict = _optimizeg(
            tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
            tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
            ti_iphi_vars, ti_iphi_max, ti_iphi_min,
            tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
            ti_ophi_vars,
            tf, ti,
            active_solvers, theta_parameters,
            nd, eps, maxpp, tolpp,
            mutation, V_matrix, design_criteria,
            model_structure, modelling_settings
        )
    else:
        raise ValueError(f"Unknown method '{method_key}' for mdopt_method in PP().")

    # ------------------------ RE-EVALUATE AT ND2 ------------------------ #
    nd = design_settings['iteration_settings']['nd2']

    # ------------------- USE FINAL SOLUTION IN _runnerpp ---------------- #
    phi, swps, St, pp_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _runner(
        result.x,
        nd,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tv_ophi_vars, ti_ophi_vars,
        active_solvers, theta_parameters,
        tv_iphi_cvp,
        tf, eps, mutation, V_matrix, design_criteria,
        index_dict,
        model_structure,
        modelling_settings
    )

    # --------------------------- REPORT & SAVE --------------------------- #
    base_path = framework_settings['path']
    modelling_folder = str(framework_settings['case'])
    filename = os.path.join(base_path, modelling_folder)
    os.makedirs(filename, exist_ok=True)

    # You may have a specialized reporter or reuse the same `_reporter`
    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        pp_obj,                  # pass the objective for logging
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf,
        filename,
        design_criteria,
        round,
        core_number
    )

    # ------------------------- RETURN FINAL DATA ------------------------- #
    design_decisions = {
        'phi': phi,
        'phit': phit,
        'swps': swps,
        'St': St,
        'pp_obj': pp_obj,
        't_values': t_values
    }

    return design_decisions, pp_obj, swps


def _optimizel(tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
                tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
                ti_iphi_vars, ti_iphi_max, ti_iphi_min,
                tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
                ti_ophi_vars,
                tf, ti,
                active_solvers, theta_parameters,
                nd, eps, maxpp, tolpp,
                mutation, V_matrix, design_criteria,
                model_structure, modelling_settings):
    """
    Local optimization for MBDOE_PP using trust-constr (analogous to _optimizel in des-md).
    Returns (result, index_dict).
    """

    # 1) Possibly build a var_groups dict like in des-md
    var_groups = build_var_groups(
        tv_iphi_vars, tv_iphi_offsetl, tv_iphi_offsett, tv_iphi_const,
        tv_ophi_vars, tv_ophi_offsett_ophi
    )

    # 2) Segmenter => bounds, x0, index_pairs, index_dict
    bounds_list, x0, index_pairs_levels, index_pairs_times, index_dict = _segmenter(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
        tf, ti
    )
    lower_bounds = [b[0] for b in bounds_list]
    upper_bounds = [b[1] for b in bounds_list]
    x0 = np.clip(x0, lower_bounds, upper_bounds)
    bounds = Bounds(lower_bounds, upper_bounds)

    # 3) Build linear constraints (similarly to des-md)
    A_level, lb_level, A_time, lb_time, constraints_list = build_linear_constraints(
        x_len=len(x0),
        index_pairs_levels=index_pairs_levels,
        index_pairs_times=index_pairs_times,
        var_groups=var_groups
    )

    # 4) Objective function that calls _runnerpp
    def local_objective(x):
        return pp_objective_function(
            x, nd,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            model_structure,
            modelling_settings
        )

    # 5) trust-constr solve
    result = minimize(
        fun=local_objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints_list,
        method='trust-constr',
        options={
            'maxiter': maxpp,
            'verbose': 3,
            'xtol': tolpp,
            'gtol': tolpp,
            'barrier_tol': 1e-8,
        }
    )

    if not result.success:
        print("Parameter-Precision (Local) Optimization failed!")
    else:
        print("Parameter-Precision (Local) Optimization succeeded!")

    return result, index_dict


def _optimizeg(tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
               tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
               ti_iphi_vars, ti_iphi_max, ti_iphi_min,
               tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
               ti_ophi_vars,
               tf, ti,
               active_solvers, theta_parameters,
               nd, eps, maxpp, tolpp,
               mutation, V_matrix, design_criteria,
               model_structure, modelling_settings):

    # 1) Build var_groups (parallel to des-md)
    var_groups = build_var_groups(
        tv_iphi_vars, tv_iphi_offsetl, tv_iphi_offsett, tv_iphi_const,
        tv_ophi_vars, tv_ophi_offsett_ophi
    )

    # 2) Segmenter => bounds, x0, index_pairs, index_dict
    bounds_list, x0, index_pairs_levels, index_pairs_times, index_dict = _segmenter(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
        tf, ti
    )
    lower_bounds = [b[0] for b in bounds_list]
    upper_bounds = [b[1] for b in bounds_list]
    trust_bounds = Bounds(lower_bounds, upper_bounds)

    # 3) Build linear constraints
    (
        A_level, lb_level,
        A_time, lb_time,
        constraints_list
    ) = build_linear_constraints(
        x_len=len(x0),
        index_pairs_levels=index_pairs_levels,
        index_pairs_times=index_pairs_times,
        var_groups=var_groups
    )

    # 4) Build partial function for local objective (pp_objective_function)
    local_obj_fun = partial(
        pp_objective_function,
        nd=nd,
        tv_iphi_vars=tv_iphi_vars,
        tv_iphi_max=tv_iphi_max,
        ti_iphi_vars=ti_iphi_vars,
        ti_iphi_max=ti_iphi_max,
        tv_ophi_vars=tv_ophi_vars,
        ti_ophi_vars=ti_ophi_vars,
        active_solvers=active_solvers,
        theta_parameters=theta_parameters,
        tv_iphi_cvp=tv_iphi_cvp,
        tf=tf, eps=eps, mutation=mutation,
        V_matrix=V_matrix, design_criteria=design_criteria,
        index_dict=index_dict,
        model_structure=model_structure,
        modelling_settings=modelling_settings
    )

    # 5) Build penalized objective for DE (just like des-md):
    #
    #    First, define arguments for the objective and the constraint violation:
    obj_args = {
        'objective_fun': local_obj_fun
    }
    constraint_args = {
        # If your violation function is named differently, adjust accordingly
        'violation_fun': partial(
            constraint_violation,  # or constraint_violation
            A_level=A_level,
            lb_level=lb_level,
            A_time=A_time,
            lb_time=lb_time
        )
    }

    #    Then define penalized function via partial:
    penalized_de_fun = partial(
        penalized_objective,  # or penalized_objective_de
        obj_args=obj_args,
        constraint_args=constraint_args,
        penalty_weight=1e4  # tune your penalty factor
    )

    # 6) Run Differential Evolution with the penalized objective
    de_bounds = list(zip(lower_bounds, upper_bounds))
    de_result = differential_evolution(
        penalized_de_fun,
        bounds=de_bounds,
        strategy='best1bin',
        maxiter=50,          # or any other suitable number
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=None,
        polish=False,        # we do our own local refinement
        workers=-1,          # parallel
        updating='deferred',
        disp=True
    )
    x0_de = de_result.x

    # 7) Local refinement with trust-constr
    result = minimize(
        fun=local_obj_fun,
        x0=x0_de,
        bounds=trust_bounds,
        constraints=constraints_list,
        method='trust-constr',
        options={
            'maxiter': maxpp,
            'verbose': 3,
            'xtol': tolpp,
            'gtol': tolpp,
            'barrier_tol': 1e-8,
        }
    )

    if not result.success:
        print("Global+Local Parameter-Precision Optimization failed or did not converge.")
    else:
        print("Global+Local Parameter-Precision Optimization succeeded!")

    return result, index_dict


def pp_objective_function(
    x,
    nd,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp,
    tf, eps, mutation, V_matrix, design_criteria,
    index_dict,
    model_structure,
    modelling_settings
):
    """
    Similar to md_objective_function but for PP. We want to maximize the PP-criterion
    => so return -pp_obj from _runnerpp (unless it's a min-type objective).
    """
    try:
        _, _, _, pp_obj, _, _, _, _ = _runner(
            x,
            nd,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            model_structure,
            modelling_settings
        )
        if np.isnan(pp_obj):
            return 1e6
        return -pp_obj  # negative => maximize pp_obj
    except Exception as e:
        print(f"Exception in pp_objective_function: {e}")
        return 1e6


def _runner(
    x,
    nd,
    tv_iphi_vars,
    tv_iphi_max,
    ti_iphi_vars,
    ti_iphi_max,
    tv_ophi_vars,
    ti_ophi_vars,
    active_solvers,
    theta_parameters,
    tv_iphi_cvp,
    tf,
    eps,
    mutation,
    V_matrix,
    MBDOE_PP_criterion,
    index_dict,
    model_structure,
    modelling_settings
):
    """
    Run the simulation and compute the parameter precision criterion for the MBDOE_PP problem.

    Parameters
    ----------
    x : list or array
        Design variables to be optimized.
    nd : int
        Number of time points in the first stage.
    tv_iphi_vars : list
        Time-variant input variables.
    tv_iphi_max : list
        Maximum values for time-variant input variables.
    ti_iphi_vars : list
        Time-invariant input variables.
    ti_iphi_max : list
        Maximum values for time-invariant input variables.
    tv_ophi_vars : list
        Time-variant output variables.
    ti_ophi_vars : list
        Time-invariant output variables.
    active_solvers : list
        List of active solvers.
    theta_parameters : dict
        Dictionary of theta parameters for solvers (keyed by solver_name).
    tv_iphi_cvp : list or dict
        Piecewise interpolation or other design data for time-variant inputs.
    tf : float
        Total experiment time.
    eps : float
        Small epsilon value for numerical sensitivity.
    mutation : dict
        Mutation settings for solvers (keyed by solver_name).
    V_matrix : dict
        V-matrix settings (keyed by solver_name).
    MBDOE_PP_criterion : str
        Criterion for the design (e.g., 'D', 'A', 'E', 'ME').
    index_dict : dict
        Dictionary of index mappings for design variables.
    model_structure : dict
        Model structure dictionary.
    modelling_settings : dict
        Additional solver settings.

    Returns
    -------
    tuple
        (ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp)
        where:
        - ti (dict): Time-invariant input data.
        - swps (dict): Switching points data.
        - St (dict): Sampling time points (dict of var -> array).
        - pp_obj (float): Performance criterion value.
        - t_values (list): Time values for the simulation.
        - tv_ophi (dict): Time-variant output results (solver_name -> {var -> array}).
        - ti_ophi (dict): Time-invariant output results (solver_name -> {var -> scalar}).
        - phit_interp (dict): Interpolated piecewise data (last solver run).
    """
    # Convert x to list if not already
    x = x.tolist() if not isinstance(x, list) else x

    # ---------------------------------------------------------------------
    # 1) Extract design parameters via _slicer
    # ---------------------------------------------------------------------
    ti, swps, St = _slicer(x, index_dict)
    # Convert St into dict of var -> np.array
    St = {var: np.array(St[var]) for var in St}

    # Combine nd equally spaced points in [0,1] with the sampling points
    t_values_flat = [tp for times in St.values() for tp in times]
    t_values = np.unique(np.concatenate((np.linspace(0, 1, nd), t_values_flat))).tolist()

    # ---------------------------------------------------------------------
    # 2) Prepare data structures
    #    LSA: dict -> dict -> dict, for partial derivatives
    #    J_dot_matrix: dict -> 2D array
    # ---------------------------------------------------------------------
    LSA = defaultdict(lambda: defaultdict(dict))
    J_dot_matrix = defaultdict(
        lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                          len(theta_parameters[active_solvers[0]])))
    )
    y_values_dict = {}       # y_values_dict[solver_name][var] -> array
    y_i_values_dict = {}     # y_i_values_dict[solver_name][var] -> array
    indices = {}             # indices[var] -> boolean mask for times

    # Check that each tv_ophi_var actually appears in St
    for var in tv_ophi_vars:
        if var not in St:
            raise ValueError(f"No time values found for variable '{var}' in St.")
        # For each var, find the boolean mask that matches St[var] in t_values
        indices[var] = np.isin(t_values, St[var])

    # Matrices to accumulate from partial derivatives
    M0 = defaultdict(
        lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                          len(theta_parameters[active_solvers[0]])))
    )
    M = {}

    # For returning solver results
    tv_ophi = {}
    ti_ophi = {}
    phit_interp = {}

    # ---------------------------------------------------------------------
    # 3) Main loop over solvers
    # ---------------------------------------------------------------------
    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        # Example: start from a "1.0" vector as your baseline parameter
        theta = np.array([1.0] * len(thetac))

        # Time-invariant and switching points data directly from slicer
        ti_iphi_data = ti
        swps_data = swps

        # Scaling factors for time-invariant (phisc) and time-variant (phitsc) inputs
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf  # treat tf as a time-scaling factor

        # -----------------------------------------------------------------
        # 3a) Run the solver for unperturbed (baseline) theta
        # -----------------------------------------------------------------
        tv_out, ti_out, interp_data = Simula(
            t_values, swps_data, ti_iphi_data,
            phisc, phitsc, tsc,
            theta, thetac,
            tv_iphi_cvp, {},  # pass empty dict or relevant data
            solver_name,
            model_structure,
            modelling_settings
        )
        tv_ophi[solver_name] = tv_out      # e.g. {var -> array} for time-variant
        ti_ophi[solver_name] = ti_out      # e.g. {var -> scalar} for time-invariant
        phit_interp = interp_data

        # Store unperturbed outputs in y_values_dict
        y_values_dict[solver_name] = {}
        for var in tv_ophi_vars:
            # Make sure it's a NumPy array
            y_values_dict[solver_name][var] = np.array(tv_out[var])
        for var in ti_ophi_vars:
            # Time-invariant output as a scalar or array(1,)
            y_values_dict[solver_name][var] = np.array([ti_out[var]])

        # Create a dictionary for "combined" unperturbed values (by var)
        y_combined = {}
        for var in tv_ophi_vars:
            y_combined[var] = y_values_dict[solver_name][var]
        for var in ti_ophi_vars:
            y_combined[var] = y_values_dict[solver_name][var]

        # -----------------------------------------------------------------
        # 3b) Loop over free parameters (the ones set to True in mutation)
        # -----------------------------------------------------------------
        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]

        # Initialize dictionary for storing partial-perturbation outputs
        y_i_values_dict[solver_name] = {}

        for para_idx in free_params_indices:
            # Perturb a single parameter
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            # -------------------------------------------------------------
            # 3b-i) Run solver with modified theta
            # -------------------------------------------------------------
            tv_out_mod, ti_out_mod, _ = Simula(
                t_values, swps_data, ti_iphi_data,
                phisc, phitsc, tsc,
                modified_theta, thetac,
                tv_iphi_cvp, {},  # pass empty dict or relevant data
                solver_name,
                model_structure,
                modelling_settings
            )

            # Store these "modified" results
            y_i_values_dict[solver_name][var] = {}
            for var in tv_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array(tv_out_mod[var])
            for var in ti_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array([ti_out_mod[var]])

            # Build a dictionary for the perturbed outputs
            y_modified_combined = {}
            for var in tv_ophi_vars:
                y_modified_combined[var] = y_i_values_dict[solver_name][var]
            for var in ti_ophi_vars:
                y_modified_combined[var] = y_i_values_dict[solver_name][var]

            # -------------------------------------------------------------
            # 3b-ii) Compute partial derivatives (LSA)
            #        LSA[var][solver_name][para_idx] = (perturbed - base)/eps
            # -------------------------------------------------------------
            for var in indices:
                LSA[var][solver_name][para_idx] = (
                    y_modified_combined[var] - y_combined[var]
                ) / eps

        # -----------------------------------------------------------------
        # 3c) Construct solver_LSA matrix from LSA
        #     Rows: each var in indices
        #     Cols: each free parameter
        # -----------------------------------------------------------------
        # If "indices" is {var1: mask, var2: mask, ...}, we can iterate over var in sorted(indices).
        # Or keep them in the order they appear. Each row in solver_LSA will be a "stack" of partials
        # for that var. (Alternatively, you might sum them or some other approach.)
        solver_LSA_list = []
        for var in indices:
            row_for_var = [LSA[var][solver_name][i] for i in free_params_indices]
            # row_for_var might be a 1D array or scalar for each i. If you want to combine them into a single float,
            # or if each is an array, you might need to flatten or handle shapes.
            # For now, let's assume each LSA[var][solver_name][i] is a 1D array => we must handle that carefully.
            # A common approach: sum or average those partial derivatives. Or stack them vertically.
            # If you want each var to produce multiple rows, you do something else.
            #
            # For a single row in solver_LSA, we might need to average or something. Let's assume we do a simple sum:
            row_summed = np.array([arr.mean() for arr in row_for_var])  # Example: each arr is a 1D difference
            solver_LSA_list.append(row_summed)
        solver_LSA = np.array(solver_LSA_list)  # shape (#vars, #free_params)

        # Transpose for J_dot = J^T J
        J_transpose = solver_LSA.T  # shape (#free_params, #vars)
        J_dot = np.dot(J_transpose, solver_LSA)  # shape (#free_params, #free_params)

        # Make sure shapes align with J_dot_matrix[solver_name]
        if J_dot_matrix[solver_name].shape != (solver_LSA.shape[1], solver_LSA.shape[1]):
            J_dot_matrix[solver_name] = np.zeros((solver_LSA.shape[1], solver_LSA.shape[1]))

        # Accumulate
        J_dot_matrix[solver_name] += J_dot
        M0[solver_name] = J_dot_matrix[solver_name]

        # Build final M = M0 + inv(V_matrix_for_free_params)
        reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
        M[solver_name] = M0[solver_name] + np.linalg.inv(reduced_V_matrix)

    # ---------------------------------------------------------------------
    # 4) Evaluate design criterion
    # ---------------------------------------------------------------------
    pp_obj = None
    solver_name = active_solvers[0]  # or whichever solver you want to measure
    if MBDOE_PP_criterion == 'D':
        pp_obj = np.linalg.det(M[solver_name])
    elif MBDOE_PP_criterion == 'A':
        pp_obj = np.trace(M[solver_name])
    elif MBDOE_PP_criterion == 'E':
        eigenvalues = np.linalg.eigvalsh(M[solver_name])
        pp_obj = np.min(eigenvalues)
    elif MBDOE_PP_criterion == 'ME':
        condition_number = np.linalg.cond(M[solver_name])
        pp_obj = -condition_number

    print(f"\rmbdoe-PP:{MBDOE_PP_criterion} is running with {pp_obj}", end='', flush=True)

    # ---------------------------------------------------------------------
    # 5) Return the relevant pieces
    # ---------------------------------------------------------------------
    # phit_interp might hold data only for the last solver or each solver in a dict
    return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
