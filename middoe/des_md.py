from functools import partial
from scipy.optimize import differential_evolution, Bounds, minimize
import os
from middoe.des_utils import _slicer, _segmenter, _reporter, _par_update, build_var_groups, build_linear_constraints, penalized_objective, constraint_violation
from collections import defaultdict
import numpy as np
from middoe.krnl_simula import Simula

def MD(design_settings, model_structure, modelling_settings, core_number, framework_settings, round):
    """
    Perform MBDOE_MD design optimization (either local or global).
    """

    # 1) Extract relevant data
    tf = model_structure['t_s'][1]
    ti = model_structure['t_s'][0]

    tv_iphi_vars = list(model_structure['tv_iphi'].keys())
    tv_iphi_max = [model_structure['tv_iphi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [model_structure['tv_iphi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [model_structure['tv_iphi'][var]['swp'] for var in tv_iphi_vars]
    tv_iphi_const = [model_structure['tv_iphi'][var]['constraints'] for var in tv_iphi_vars]
    tv_iphi_offsett = [model_structure['tv_iphi'][var]['offsett'] for var in tv_iphi_vars]
    tv_iphi_offsetl = [model_structure['tv_iphi'][var]['offsetl'] for var in tv_iphi_vars]
    tv_iphi_cvp = {var: model_structure['tv_iphi'][var]['design_cvp'] for var in tv_iphi_vars}

    ti_iphi_vars = list(model_structure['ti_iphi'].keys())
    ti_iphi_max = [model_structure['ti_iphi'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [model_structure['ti_iphi'][var]['min'] for var in ti_iphi_vars]

    tv_ophi_vars = [
        var for var in model_structure['tv_ophi'].keys()
        if model_structure['tv_ophi'][var].get('measured', True)
    ]
    tv_ophi_seg = [model_structure['tv_ophi'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett = [model_structure['tv_ophi'][var]['offsett'] for var in tv_ophi_vars]
    tv_ophi_matching = [model_structure['tv_ophi'][var]['matching'] for var in tv_ophi_vars]

    ti_ophi_vars = [
        var for var in model_structure['ti_ophi'].keys()
        if model_structure['ti_ophi'][var].get('measured', True)
    ]

    active_solvers = modelling_settings['active_solvers']
    estimations = modelling_settings['normalized_parameters']
    ref_thetas = modelling_settings['theta_parameters']
    theta_parameters = _par_update(ref_thetas, estimations)

    design_criteria = design_settings['criteria']['MBDOE_MD_criterion']
    nd = design_settings['iteration_settings']['nd']
    eps = design_settings['eps']
    maxmd = design_settings['iteration_settings']['maxmd']
    tolmd = design_settings['iteration_settings']['tolmd']
    mutation = modelling_settings['mutation']

    # 2) Decide local vs global
    method_key = design_settings['optimization_methods']['mdopt_method']
    if method_key == 'Local':
        result, index_dict = _optimizel(
            tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
            tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
            ti_iphi_vars, ti_iphi_max, ti_iphi_min,
            tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett, tv_ophi_matching,
            ti_ophi_vars,
            tf, ti,
            active_solvers, theta_parameters, nd, eps, maxmd, tolmd,
            mutation, design_criteria, model_structure, modelling_settings
        )
    elif method_key == 'Global':
        result, index_dict = _optimizeg(
            tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
            tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
            ti_iphi_vars, ti_iphi_max, ti_iphi_min,
            tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett, tv_ophi_matching,
            ti_ophi_vars,
            tf, ti,
            active_solvers, theta_parameters, nd, eps, maxmd, tolmd,
            mutation, design_criteria, model_structure, modelling_settings
        )
    else:
        raise ValueError(f"Unknown method '{method_key}' for mdopt_method.")

    # 3) Evaluate again at nd2
    nd = design_settings['iteration_settings']['nd2']

    # 4) Use the final solution in _runner
    phi, swps, St, sum_sq_diffs, t_vals, tv_ophi_dict, ti_ophi_dict, phit = _runner(
        result.x, nd, tv_iphi_vars, tv_iphi_max, ti_iphi_vars, ti_iphi_max,
        tv_ophi_vars, ti_ophi_vars, active_solvers,
        theta_parameters, tv_iphi_cvp, design_criteria, tf, eps, mutation,
        index_dict, model_structure, modelling_settings
    )

    # 5) Prepare output
    # You can create plots, save to file, etc.:
    base_path = framework_settings['path']
    modelling_folder = str(framework_settings['case'])
    filename = os.path.join(base_path, modelling_folder)
    os.makedirs(filename, exist_ok=True)

    # Suppose you call _reporter(...) to finalize
    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        sum_sq_diffs, t_vals,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf, filename,
        design_criteria,
        round,
        core_number
    )

    design_decisions = {
        'phi': phi,
        'phit': phit,
        'swps': swps,
        'St': St,
        't_values': t_vals
    }

    return design_decisions, sum_sq_diffs, swps

def _optimizeg(tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
               tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
               ti_iphi_vars, ti_iphi_max, ti_iphi_min,
               tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
               ti_ophi_vars,
               tf, ti,
               active_solvers, theta_parameters, nd, eps, maxmd, tolmd,
               mutation, design_criteria, model_structure, modelling_settings):
    """
    Global optimization (Differential Evolution) for MBDOE_MD,
    then local refinement (trust-constr).
    Returns (result, index_dict).
    """

    # 1) var_groups
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

    trust_bounds = Bounds(lower_bounds, upper_bounds)

    # 3) Build linear constraints
    (A_level, lb_level,
     A_time, lb_time,
     constraints_list) = build_linear_constraints(
        x_len=len(x0),
        index_pairs_levels=index_pairs_levels,
        index_pairs_times=index_pairs_times,
        var_groups=var_groups
    )

    # 4) Build a partial function for the objective
    local_obj_fun = partial(
        md_objective_function,
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
        design_criteria=design_criteria,
        tf=tf,
        eps=eps,
        mutation=mutation,
        index_dict=index_dict,
        model_structure=model_structure,
        modelling_settings=modelling_settings,
        runner_function=_runner
    )

    # 5) For DE, define a penalty-based objective
    #    pass references to local_obj_fun plus A_level etc.
    obj_args = {
        'objective_fun': local_obj_fun
    }
    constraint_args = {
        'violation_fun': partial(
            constraint_violation,
            A_level=A_level,
            lb_level=lb_level,
            A_time=A_time,
            lb_time=lb_time
        )
    }

    penalized_de_fun = partial(
        penalized_objective,
        obj_args=obj_args,
        constraint_args=constraint_args,
        penalty_weight=1e4
    )

    # 6) Differential Evolution
    de_bounds = list(zip(lower_bounds, upper_bounds))
    de_result = differential_evolution(
        penalized_de_fun,
        bounds=de_bounds,
        strategy='best1bin',
        maxiter=50,       # or whatever you want
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=None,
        polish=False,     # local refinement ourselves
        workers=-1,       # parallel
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
            'maxiter': maxmd,
            'verbose': 3,
            'xtol': tolmd,
            'gtol': tolmd,
            'barrier_tol': 1e-8,
        }
    )

    if not result.success:
        print("Global+Local Optimization (MBDOE_MD) failed or did not converge.")
    else:
        print("Global+Local Optimization (MBDOE_MD) succeeded!")

    return result, index_dict

def _optimizel(tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
               tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
               ti_iphi_vars, ti_iphi_max, ti_iphi_min,
               tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_matching,
               ti_ophi_vars,
               tf, ti,
               active_solvers, theta_parameters, nd, eps, maxmd, tolmd,
               mutation, design_criteria, model_structure, modelling_settings):
    """
    Local optimization for MBDOE_MD (trust-constr).
    Returns (result, index_dict).
    """

    # 1) var_groups
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

    # 3) Build linear constraints
    A_level, lb_level, A_time, lb_time, constraints_list = build_linear_constraints(
        x_len=len(x0),
        index_pairs_levels=index_pairs_levels,
        index_pairs_times=index_pairs_times,
        var_groups=var_groups
    )

    # 4) Objective function for trust-constr
    def local_objective(x):
        return md_objective_function(
            x,
            nd,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp,
            design_criteria,
            tf, eps, mutation,
            index_dict,
            model_structure,
            modelling_settings,
            _runner  # pass your actual runner function
        )

    # 5) Solve with trust-constr
    result = minimize(
        fun=local_objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints_list,
        method='trust-constr',
        options={
            'maxiter': maxmd,
            'verbose': 3,
            'xtol': tolmd,
            'gtol': tolmd,
            'barrier_tol': 1e-8,
        }
    )

    if not result.success:
        print("Optimization (MBDOE_MD local) failed!")
    else:
        print("Optimization (MBDOE_MD local) succeeded!")

    return result, index_dict

def _runner(
    x, nd,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp,
    design_criteria,
    tf, eps, mutation,
    index_dict,
    model_structure,
    modelling_settings
):
    """
    Run the optimization process and return the results.

    Parameters
    ----------
    x : list
        Current values of the decision variables.
    nd : int
        Number of time points in the first stage.
    tv_iphi_vars : list
        Time-variant input variables.
    tv_iphi_max : list
        Maximum scaling values for time-variant inputs.
    ti_iphi_vars : list
        Time-invariant input variables.
    ti_iphi_max : list
        Maximum scaling values for time-invariant inputs.
    tv_ophi_vars : list
        Time-variant output variables.
    ti_ophi_vars : list
        Time-invariant output variables.
    active_solvers : list
        List of solver names.
    theta_parameters : dict
        Dictionary of parameter vectors keyed by solver_name.
    tv_iphi_cvp : dict
        Piecewise interpolation or design functions for time-variant inputs.
    design_criteria : str
        The criterion for the design (e.g., 'HR', 'BFF').
    tf : float
        Final time for the experiment.
    eps : float
        Small epsilon for numerical sensitivity.
    mutation : dict
        Dictionary keyed by solver_name → list of booleans (which parameters are free).
    index_dict : dict
        Dictionary of index segments from `_segmenter` / `_slicer`.
    model_structure : dict
        Model structure used by the solver.
    modelling_settings : dict
        Additional settings for the solver or experiment.

    Returns
    -------
    ti : dict or list
        Time-invariant input data from `_slicer`.
    swps : dict or list
        Switching points from `_slicer`.
    St : dict
        Sampling times, keyed by output variable name → array of times.
    sum_squared_differences : float
        The objective function value.
    t_values : list
        Final time grid used in the simulation (dimensionless from 0..1 plus any St points).
    tv_ophi : dict
        Time-variant outputs from each solver.
    ti_ophi : dict
        Time-invariant outputs from each solver.
    phit_interp : any
        Additional data from the solver (if returned).
    """

    # 1) Extract design parameters via _slicer
    x = x.tolist() if not isinstance(x, list) else x
    ti, swps, St = _slicer(x, index_dict)

    # Convert St to a dict of {var: np.array([...])}
    for var in St:
        if not isinstance(St[var], np.ndarray):
            St[var] = np.array(St[var])

    # 2) Build the combined time vector: dimensionless 0..1 plus any points from St
    t_values_flat = []
    for var in St:
        t_values_flat.extend(St[var])  # gather all sampling times for all variables
    t_lin = np.linspace(0, 1, nd)
    t_values = np.unique(np.concatenate([t_lin, t_values_flat])).tolist()

    # 3) Prepare data structures
    LSA = defaultdict(lambda: defaultdict(dict))
    y_values_dict = defaultdict(dict)      # For each solver, store {var: array_of_outputs}
    y_i_values_dict = defaultdict(dict)    # For partial-derivative modifications
    indices = {}                           # For each var, a boolean mask in t_values
    for var in tv_ophi_vars:
        if var not in St:
            raise ValueError(f"No time values found for variable '{var}' in St.")
        mask = np.isin(t_values, St[var])
        indices[var] = mask

    # J_dot: dictionary of solver_name → array of shape (#params, #params)
    J_dot = defaultdict(
        lambda: np.zeros(
            (len(theta_parameters[active_solvers[0]]),
             len(theta_parameters[active_solvers[0]])),
            dtype=float
        )
    )

    # For storing final solver outputs
    tv_ophi = {}
    ti_ophi = {}

    # 4) Solve each model + compute partial derivatives
    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        theta = np.array([1.0] * len(thetac), dtype=float)  # initial guess or scale factor?

        # Gather data from slicer
        ti_iphi_data = ti
        swps_data = swps

        # Build scaling dicts
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf  # pass to solver if the solver does time-scaling internally

        # 4a) Run solver unperturbed
        tv_ophi[solver_name], ti_ophi[solver_name], phit_interp = Simula(
            t_values,
            swps_data,
            ti_iphi_data,
            phisc,
            phitsc,
            tsc,
            theta,
            thetac,
            tv_iphi_cvp,
            {},
            solver_name,
            model_structure,
            modelling_settings
        )

        # 4b) Store results in y_values_dict[solver_name][var]
        #     So we can easily do y_values_dict[solver_name][var][mask]
        for var in tv_ophi_vars:
            y_values_dict[solver_name][var] = np.array(tv_ophi[solver_name][var])
        for var in ti_ophi_vars:
            y_values_dict[solver_name][var] = np.array(ti_ophi[solver_name][var])

        # 4c) Prepare for partial derivatives
        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]

        # For each free parameter, perturb and re-run
        for para_idx in free_params_indices:
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            # Re-run solver with perturbed parameter
            tv_mod, ti_mod, _ = Simula(
                t_values,
                swps_data,
                ti_iphi_data,
                phisc,
                phitsc,
                tsc,
                modified_theta,
                thetac,
                tv_iphi_cvp,
                {},
                solver_name,
                model_structure,
                modelling_settings
            )

            # Store these modified solver outputs
            for var in tv_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array(tv_mod[var])
            for var in ti_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array(ti_mod[var])

            # For each var in indices, compute partial derivative
            for var, mask in indices.items():
                # base_vals = unperturbed solver result
                base_vals = y_values_dict[solver_name][var]
                # pert_vals = perturbed solver result
                pert_vals = y_i_values_dict[solver_name][var]
                # difference
                diff = pert_vals - base_vals
                # partial derivative
                LSA[var][solver_name][para_idx] = diff / eps

        # 4d) Build solver_LSA by stacking across all `var in indices`
        solver_LSA_list = []
        for var in indices:
            # For each param i in free_params_indices, grab LSA[var][solver_name][i]
            row_for_var = [LSA[var][solver_name][i] for i in free_params_indices]
            # Suppose we take the mean across the time dimension
            row_summed = np.array([arr.mean() for arr in row_for_var])
            solver_LSA_list.append(row_summed)

        # Convert to array => shape (#vars, #free_params)
        solver_LSA = np.array(solver_LSA_list, dtype=float)

        # 4e) Compute J^T J
        J_transpose = solver_LSA.T
        J_dot_matrix = np.dot(J_transpose, solver_LSA)

        # Ensure correct shape
        if J_dot[solver_name].shape != (solver_LSA.shape[1], solver_LSA.shape[1]):
            J_dot[solver_name] = np.zeros((solver_LSA.shape[1], solver_LSA.shape[1]), dtype=float)

        J_dot[solver_name] += J_dot_matrix

    # 5) Compute the objective
    sum_squared_differences = 0.0

    if design_criteria == 'HR':
        # Compare pairwise differences between all solvers
        for i, solver1 in enumerate(active_solvers):
            for solver2 in active_solvers[i+1:]:
                # For each var in tv_ophi_vars (since indices only has those)
                for var, mask in indices.items():
                    y1 = y_values_dict[solver1][var]
                    y2 = y_values_dict[solver2][var]
                    y1_subset = y1[mask]
                    y2_subset = y2[mask]
                    sum_squared_differences += np.sum((y1_subset - y2_subset)**2)

    elif design_criteria == 'BFF':
        # For each solver, do quadratic form (diff)^T (J + regI) (diff)
        for solver_name in active_solvers:
            J_reg = J_dot[solver_name] + 1e-10 * np.eye(J_dot[solver_name].shape[0])
            # For each variable in indices, build difference_vector
            for var, mask in indices.items():
                diff_vec = y_values_dict[solver_name][var][mask]
                # Check shape
                if J_reg.shape[0] != diff_vec.shape[0]:
                    raise ValueError(
                        f"Shape mismatch: J_dot has shape {J_reg.shape}, "
                        f"but difference_vector for var '{var}' has shape {diff_vec.shape}."
                    )
                sum_squared_differences += diff_vec @ (J_reg @ diff_vec)

    print(f"\rmbdoe-MD:{design_criteria} is running with {sum_squared_differences}", end='', flush=True)

    # Return everything needed
    return ti, swps, St, sum_squared_differences, t_values, tv_ophi, ti_ophi, phit_interp

def md_objective_function(
    x,
    nd,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp,
    design_criteria,
    tf, eps, mutation,
    index_dict,
    model_structure,
    modelling_settings,
    runner_function
):
    """
    A top-level objective function that calls your _runner function
    and returns -md_obj (assuming you want to maximize md_obj).
    """
    try:
        # _runner signature:
        # def _runner(x, nd, tv_iphi_vars, tv_iphi_max, ti_iphi_vars, ti_iphi_max,
        #             tv_ophi_vars, ti_ophi_vars, active_solvers, theta_parameters,
        #             tv_iphi_cvp, design_criteria, tf, eps, mutation,
        #             index_dict, model_structure, modelling_settings)
        _, _, _, md_obj, _, _, _, _ = runner_function(
            x,
            nd,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp,
            design_criteria,
            tf, eps, mutation,
            index_dict,
            model_structure,
            modelling_settings
        )
        if np.isnan(md_obj):
            return 1e6  # penalty
        return -md_obj  # negative => "maximize" md_obj
    except Exception as e:
        print(f"Exception in md_objective_function: {e}")
        return 1e6  # big penalty on exception

