from middoe.des_utils import _slicer, _reporter, _par_update, configure_logger
from functools import partial
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from multiprocessing import Pool
import traceback
from middoe.krnl_simula import simula
from collections import defaultdict
import numpy as np
import logging



from typing import Dict, Any, Union



def mbdoe_md(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    round: int,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    """
    Execute the Model-Based Design of Experiments for Model Discrimination (MBDoE-MD).

    Orchestrates parallel or single-core optimisation runs, filters out failures,
    and returns a dictionary containing the best design and associated metrics.
    """
    if num_parallel_runs > 1:
        with Pool(num_parallel_runs) as pool:
            results_list = pool.map(
                _safe_run_md,
                [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
            )

        successful = [res for res in results_list if res is not None]
        if not successful:
            raise RuntimeError(
                "All MBDoE-MD optimisation runs failed. "
                "Try increasing 'maxmd', adjusting bounds, or relaxing constraints."
            )

        best_design_decisions, best_md_obj, best_swps = max(successful, key=lambda x: x[1])

    else:
        try:
            best_design_decisions, best_md_obj, best_swps = _run_single_md(
                des_opt, system, models, core_number=0, round=round
            )
        except Exception as e:
            raise RuntimeError(f"Single-core optimisation failed: {e}")

        print("Design your experiment based on:")
        print("  tii   :", best_design_decisions['tii'])
        print("  tvi   :", best_design_decisions['tvi'])
        print("  swps  :", best_design_decisions['swps'])
        print("  St    :", best_design_decisions['St'])
        print("  md_obj:", best_design_decisions['md_obj'])
    return best_design_decisions

def _safe_run_md(args):
    des_opt, system, models, core_num, round = args

    try:
        return _run_single_md(des_opt, system, models, core_num, round)
    except Exception as e:
        print(f"[Core {core_num}] FAILED: {e}")
        traceback.print_exc()
        return None

def _run_single_md(des_opt, system, models, core_number=0, round=round):
    tf = system['t_s'][1]
    ti = system['t_s'][0]

    tv_iphi_vars = list(system['tvi'].keys())
    tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
    tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
    tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
    tv_iphi_offsetl = [system['tvi'][var]['offl'] / system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}

    ti_iphi_vars = list(system['tii'].keys())
    ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]

    tv_ophi_vars = [var for var in system['tvo'].keys() if system['tvo'][var].get('meas', True)]
    tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
    tv_ophi_sampling = {var: system['tvo'][var].get('samp_s', 'default_group') for var in tv_ophi_vars}
    tv_ophi_forcedsamples = {
        var: [v / tf for v in system['tvo'][var].get('samp_f', [])]
        for var in tv_ophi_vars
    }

    ti_ophi_vars = [var for var in system['tio'].keys() if system['tio'][var].get('meas', True)]

    active_solvers = models['can_m']
    estimations = models['normalized_parameters']
    ref_thetas = models['theta']
    theta_parameters = _par_update(ref_thetas, estimations)

    design_criteria = des_opt['md_ob']
    maxmd = des_opt['itr']['maxmd']
    tolmd = des_opt['itr']['tolmd']
    eps = des_opt['eps']
    mutation = models['mutation']
    population_size = des_opt['itr']['pps']
    pltshow = des_opt['plt']

    result, index_dict = _optimiser_md(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
        ti_ophi_vars,
        tf, ti,
        active_solvers, theta_parameters,
        eps, maxmd, tolmd,population_size,
        mutation, design_criteria,
        system, models
    )

    x_final = getattr(result, 'X', getattr(result, 'x', None))
    if x_final is None:
        raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")

    try:
        phi, swps, St, md_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _runner_md(
            x_final,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            design_criteria, tf, eps, mutation,
            index_dict,
            system,
            models
        )
    except ValueError as e:
        if "MBDoE optimiser kernel was unsuccessful" in str(e):
            print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
            return None
        else:
            raise

    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        md_obj,
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf,
        design_criteria,
        round, pltshow,
        core_number
    )

    design_decisions = {
        'tii': phi,
        'tvi': phit,
        'swps': swps,
        'St': St,
        'md_obj': md_obj,
        't_values': t_values
    }


    return design_decisions, md_obj, swps

def _optimiser_md(
    tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
    tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
    ti_iphi_vars, ti_iphi_max, ti_iphi_min,
    tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
    ti_ophi_vars,
    tf, ti,
    active_solvers, theta_parameters,
    eps, maxmd, tolmd,population_size,
    mutation, design_criteria,
    system, models
):
    bounds = []
    x0 = []
    index_dict = {
        'ti': {},
        'swps': {},
        'st': {}
    }

    for i, (name, mn, mx) in enumerate(zip(ti_iphi_vars, ti_iphi_min, ti_iphi_max)):
        lo, hi = mn / mx, 1.0
        bounds.append((lo, hi))
        x0.append((lo + hi) / 2)
        index_dict['ti'][name] = [i]

    start = len(x0)
    for i, name in enumerate(tv_iphi_vars):
        seg = tv_iphi_seg[i]
        mn, mx = tv_iphi_min[i], tv_iphi_max[i]
        lo = mn / mx
        level_idxs = list(range(start, start + seg - 1))
        index_dict['swps'][name + 'l'] = level_idxs
        for _ in range(seg - 1):
            bounds.append((lo, 1.0))
            x0.append((lo + 1.0) / 2)
        start += seg - 1

    for i, name in enumerate(tv_iphi_vars):
        seg = tv_iphi_seg[i]
        lo, hi = ti / tf, 1 - ti / tf
        time_idxs = list(range(start, start + seg - 2))
        index_dict['swps'][name + 't'] = time_idxs
        for _ in range(seg - 2):
            bounds.append((lo, hi))
            x0.append((lo + hi) / 2)
        start += seg - 2

    sampling_groups = defaultdict(list)
    for var in tv_ophi_vars:
        group_id = tv_ophi_sampling[var]
        sampling_groups[group_id].append(var)

    for group_id, group_vars in sampling_groups.items():
        var = group_vars[0]
        i = tv_ophi_vars.index(var)
        seg = tv_ophi_seg[i]
        num_forced = len(tv_ophi_forcedsamples[var])
        num_free = seg - num_forced
        lo, hi = ti / tf, 1 - ti / tf

        idxs = list(range(start, start + num_free))
        for var_in_group in group_vars:
            index_dict['st'][var_in_group] = idxs

        for _ in range(num_free):
            bounds.append((lo, hi))
            x0.append((lo + hi) / 2)
        start += num_free

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    x0 = np.array(x0)

    constraint_index_list = []
    for i, name in enumerate(tv_iphi_vars):
        const = tv_iphi_const[i]
        if const != 'rel':
            idxs = index_dict['swps'][name + 'l']
            for j in range(len(idxs) - 1):
                constraint_index_list.append(('lvl', i, idxs[j], idxs[j + 1]))

    for i, name in enumerate(tv_iphi_vars):
        idxs = index_dict['swps'][name + 't']
        for j in range(len(idxs) - 1):
            constraint_index_list.append(('t', i, idxs[j], idxs[j + 1]))

    for i, name in enumerate(tv_ophi_vars):
        idxs = index_dict['st'][name]
        for j in range(len(idxs) - 1):
            constraint_index_list.append(('st', i, idxs[j], idxs[j + 1]))

    total_constraints = len(constraint_index_list)

    local_obj = partial(
        _md_of,
        tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
        ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
        tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
        tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
        tv_ophi_sampling=tv_ophi_sampling,
        active_solvers=active_solvers, theta_parameters=theta_parameters,
        tf=tf, eps=eps, mutation=mutation,
        design_criteria=design_criteria,
        index_dict=index_dict, system=system, models=models
    )

    class DEProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(bounds),
                n_obj=1,
                n_constr=total_constraints,
                xl=lower,
                xu=upper
            )

        def _evaluate(self, x, out, *args, **kwargs):
            f_val = local_obj(x)
            g = []
            for kind, i, i1, i2 in constraint_index_list:
                if kind == 'lvl':
                    offs = tv_iphi_offsetl[i]
                    const = tv_iphi_const[i]
                    diff = x[i2] - x[i1] if const == 'inc' else x[i1] - x[i2]
                    g.append(offs - diff)
                elif kind == 't':
                    offs = tv_iphi_offsett[i]
                    g.append(offs - (x[i2] - x[i1]))
                elif kind == 'st':
                    offs = tv_ophi_offsett_ophi[i]
                    g.append(offs - (x[i2] - x[i1]))
            out['F'] = f_val
            out['G'] = np.array(g, dtype=np.float64)

    problem = DEProblem()
    algo = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)

    res_de = pymoo_minimize(
        problem,
        algo,
        termination=('n_gen', maxmd),
        seed=None,
        verbose=True,
        constraint_tolerance=tolmd
    )

    return res_de, index_dict


def _md_of(
    x,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
    tf, eps, mutation, design_criteria,
    index_dict, system, models
):
    """
    Objective function wrapper for MBDOE-MD. Returns the negative MD objective
    for maximization purposes, with penalty fallback in case of exceptions.
    """
    try:
        _, _, _, md_obj, _, _, _, _ = _runner_md(
            x,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            design_criteria,
            tf, eps, mutation,
            index_dict,
            system, models
        )
        if np.isnan(md_obj):
            return 1e6
        return -md_obj  # negative for maximization
    except Exception as e:
        print(f"Exception in md_objective_function: {e}")
        return 1e6

def _runner_md(
    x,
    tv_iphi_vars,
    tv_iphi_max,
    ti_iphi_vars,
    ti_iphi_max,
    tv_ophi_vars,
    ti_ophi_vars,
    active_solvers,
    theta_parameters,
    tv_iphi_cvp,
    tv_ophi_forcedsamples,
    tv_ophi_sampling,
    design_criteria,
    tf,
    eps,
    mutation,
    index_dict,
    system,
    models
):
    """
    Simulate models and evaluate the MD objective.
    Returns the sliced inputs, objective value, time vector, and outputs.
    """

    x = x.tolist() if not isinstance(x, list) else x

    # Time slicing
    dt_real = system['t_r']
    nodes = int(round(tf / dt_real)) + 1
    tlin = np.linspace(0, 1, nodes)
    ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
    St = {var: np.array(sorted(St[var])) for var in St}
    t_values_flat = [tp for times in St.values() for tp in times]
    t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

    LSA = defaultdict(lambda: defaultdict(dict))
    y_values_dict = defaultdict(dict)
    y_i_values_dict = defaultdict(dict)
    indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars}

    J_dot = defaultdict(
        lambda: np.zeros(
            (len(theta_parameters[active_solvers[0]]),
             len(theta_parameters[active_solvers[0]]))
        )
    )
    tv_ophi = {}
    ti_ophi = {}
    phit_interp = {}

    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        theta = np.array([1.0] * len(thetac))
        ti_iphi_data = ti
        swps_data = swps
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf

        tv_out, ti_out, interp_data = simula(
            t_values, swps_data, ti_iphi_data,
            phisc, phitsc, tsc,
            theta, thetac,
            tv_iphi_cvp, {},
            solver_name,
            system,
            models
        )

        tv_ophi[solver_name] = tv_out
        ti_ophi[solver_name] = ti_out
        phit_interp = interp_data

        for var in tv_ophi_vars:
            y_values_dict[solver_name][var] = np.array(tv_out[var])
        for var in ti_ophi_vars:
            y_values_dict[solver_name][var] = np.array([ti_out[var]])

        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]

        for para_idx in free_params_indices:
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            tv_out_mod, ti_out_mod, _ = simula(
                t_values, swps_data, ti_iphi_data,
                phisc, phitsc, tsc,
                modified_theta, thetac,
                tv_iphi_cvp, {},
                solver_name,
                system,
                models
            )

            for var in tv_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array(tv_out_mod[var])
            for var in ti_ophi_vars:
                y_i_values_dict[solver_name][var] = np.array([ti_out_mod[var]])

            for var in indices:
                LSA[var][solver_name][para_idx] = (
                    y_i_values_dict[solver_name][var] - y_values_dict[solver_name][var]
                ) / eps

    md_obj = 0.0
    if design_criteria == 'HR':
        for i, s1 in enumerate(active_solvers):
            for s2 in active_solvers[i+1:]:
                for var, mask in indices.items():
                    y1 = y_values_dict[s1][var]
                    y2 = y_values_dict[s2][var]
                    md_obj += np.sum((y1[mask] - y2[mask]) ** 2)

    elif design_criteria == 'BFF':
        std_dev = {var: system['tvo'][var]['unc'] for var in tv_ophi_vars}
        Sigma_y = np.diag([std_dev[var] ** 2 for var in tv_ophi_vars])
        for i, s1 in enumerate(active_solvers):
            for s2 in active_solvers[i+1:]:
                for t_idx, t in enumerate(t_values):
                    y1 = np.array([y_values_dict[s1][var][t_idx] for var in tv_ophi_vars])
                    y2 = np.array([y_values_dict[s2][var][t_idx] for var in tv_ophi_vars])
                    diff = y1 - y2

                    free_s1 = [i for i, v in enumerate(mutation[s1]) if v]
                    thetac_s1 = np.array(theta_parameters[s1])[free_s1]
                    V1 = np.array([[LSA[var][s1][p][t_idx] for p in free_s1] for var in tv_ophi_vars])
                    Sigma_theta_s1_inv = np.diag(1 / (thetac_s1 ** 2 + 1e-50))
                    W1 = V1 @ Sigma_theta_s1_inv @ V1.T

                    free_s2 = [i for i, v in enumerate(mutation[s2]) if v]
                    thetac_s2 = np.array(theta_parameters[s2])[free_s2]
                    V2 = np.array([[LSA[var][s2][p][t_idx] for p in free_s2] for var in tv_ophi_vars])
                    Sigma_theta_s2_inv = np.diag(1 / (thetac_s2 ** 2 + 1e-50))
                    W2 = V2 @ Sigma_theta_s2_inv @ V2.T

                    try:
                        S = Sigma_y + W1 + W2
                        md_obj += diff.T @ np.linalg.inv(S) @ diff
                    except np.linalg.LinAlgError:
                        md_obj += 1e6  # Penalise ill-conditioned cases

    #
    #     solver_LSA_list = []
    #     for var in indices:
    #         row_for_var = [LSA[var][solver_name][i] for i in free_params_indices]
    #         row_summed = np.array([arr.mean() for arr in row_for_var])
    #         solver_LSA_list.append(row_summed)
    #     solver_LSA = np.array(solver_LSA_list)
    #     J_transpose = solver_LSA.T
    #     J_dot_matrix = np.dot(J_transpose, solver_LSA)
    #
    #     if J_dot[solver_name].shape != (solver_LSA.shape[1], solver_LSA.shape[1]):
    #         J_dot[solver_name] = np.zeros((solver_LSA.shape[1], solver_LSA.shape[1]))
    #
    #     J_dot[solver_name] += J_dot_matrix
    #
    # md_obj = 0.0
    # if design_criteria == 'HR':
    #     for i, s1 in enumerate(active_solvers):
    #         for s2 in active_solvers[i+1:]:
    #             for var, mask in indices.items():
    #                 y1 = y_values_dict[s1][var]
    #                 y2 = y_values_dict[s2][var]
    #                 md_obj += np.sum((y1[mask] - y2[mask]) ** 2)
    # elif design_criteria == 'BFF':
    #     for solver_name in active_solvers:
    #         J_reg = J_dot[solver_name] + 1e-10 * np.eye(J_dot[solver_name].shape[0])
    #         for var, mask in indices.items():
    #             diff_vec = y_values_dict[solver_name][var][mask]
    #             md_obj += diff_vec @ (J_reg @ diff_vec)

    logger = configure_logger()
    logger.info(f"mbdoe-MD:{design_criteria} is running with {md_obj:.4f}")


    return ti, swps, St, md_obj, t_values, tv_ophi, ti_ophi, phit_interp