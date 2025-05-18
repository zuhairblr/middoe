import logging

from middoe.des_utils import _slicer, _reporter, _par_update, configure_logger
from middoe.krnl_simula import simula
from collections import defaultdict
from functools import partial
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from multiprocessing import Pool
import traceback

from typing import Dict, Any, Union

def mbdoe_pp(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    round: int,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    """
    Execute the Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).

    This function orchestrates one or more optimisation runs (in parallel or single-core)
    to identify the most informative experimental design for improving parameter precision.
    It filters failed runs and returns the design that maximises the specified precision criterion.

    Parameters
    ----------
    des_opt : dict
        Dictionary of optimisation settings:
            - 'criteria' : dict[str, str]
                Specifies the objective function for precision (e.g., {'MBDOE_PP_criterion': 'D'})
            - 'iteration_settings' : dict
                Iteration controls:
                    * 'maxpp': int — maximum iterations
                    * 'tolpp': float — convergence threshold
            - 'eps' : float
                Perturbation step size for numerical derivatives.

    system : dict
        Dictionary defining the system model and constraints:
            - 'tf', 'ti' : float
                Final and initial time for simulation.
            - 'tv_iphi', 'ti_iphi' : dict
                Time-variant and time-invariant input variable definitions.
            - 'tv_ophi', 'ti_ophi' : dict
                Time-variant and time-invariant output definitions.
            - 'tv_iphi_seg' : dict[str, int]
                Segmentation for control profiles (CVP).
            - 'tv_iphi_offsett', 'tv_iphi_offsetl' : dict
                Time/level offsets to enforce switching logic.
            - 'tv_ophi_matching' : dict[str, bool]
                Enforces synchronised sampling between outputs.
            - 'sampling_constraints' : dict
                Sampling limits, including:
                    * 'min_interval': float — minimum time between samples
                    * 'max_samples_per_response': int
                    * 'total_samples': int
                    * 'forbidden_times': dict[str, List[float]]
                    * 'forced_times': dict[str, List[float]]
                    * 'sync_sampling': bool

    models : dict
        Model and model settings:
            - 'active_models' : List[str]
                Names of solvers to evaluate.
            - 'theta_parameters' : dict[str, List[float]]
                Nominal parameter values for each model.
            - 'mutation' : dict[str, List[bool]]
                Boolean masks for which parameters are to be optimised.
            - 'V_matrix' : dict[str, ndarray]
                Prior covariance matrices for parameters.
            - 'ext_models' : dict[str, Callable]
                Mapping of model names to simulation functions.

    round : int
        Current experimental design round. Used for conditional logic/logging.

    num_parallel_runs : int, optional
        Number of parallel runs to execute. Default is 1 (single-core mode).

    Returns
    -------
    results : dict
        Dictionary containing the best optimisation result:
            - 'x' : dict[str, Any]
                Optimal design decisions including sampling times, control levels, etc.
            - 'fun' : float
                Objective function value (e.g., log det(FIM)) of the best design.
            - 'swps' : dict[str, List[float]]
                Switching times for time-variant control profiles.

    Raises
    ------
    RuntimeError
        Raised when all parallel runs fail, or the single-core run encounters an exception.

    Notes
    -----
    MBDoE-PP focuses on reducing the uncertainty in estimated parameters
    by designing informative experiments. This is achieved by maximising
    a criterion derived from the Fisher Information Matrix (FIM), such as:
        - D-optimality (log det(FIM))
        - A-optimality (trace of FIM⁻¹)
        - E-optimality (smallest eigenvalue of FIM)

    Constraints supported:
        - Level bounds for control variables
        - Minimum sampling intervals
        - Total number of samples or max per response
        - Forced or forbidden sampling times
        - Control switching limits and synchronisation
        - Synchronized sampling across outputs

    The best design is selected as the one achieving the highest value of the objective
    among all valid runs.

    References
    ----------
    Franceschini, G., & Macchietto, S. (2008).
    Model-based design of experiments for parameter precision: State of the art.
    Chemical Engineering Science, 63(19), 4846–4872.
    https://doi.org/10.1016/j.ces.2008.07.006

    See Also
    --------
    _safe_run : Wrapper to safely run optimisation in parallel.
    _run_single_pp : Direct call for single-core optimisation logic.

    Examples
    --------
    >>> result = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=4)
    >>> print("Best objective value:", result['fun'])
    >>> print("Design variables:", result['x'])

    """

    if num_parallel_runs > 1:
        with Pool(num_parallel_runs) as pool:
            results_list = pool.map(
                _safe_run,
                [(des_opt, system, models, core_number, round) for core_number in range(num_parallel_runs)]
            )

        successful = [res for res in results_list if res is not None]
        if not successful:
            raise RuntimeError(
                "All MBDOE-PP optimisation runs failed. "
                "Try increasing 'maxpp', adjusting bounds, or relaxing constraints."
            )

        best_design_decisions, best_pp_obj, best_swps = max(successful, key=lambda x: x[1])

    else:
        try:
            best_design_decisions, best_pp_obj, best_swps = _run_single_pp(
                des_opt, system, models, core_number=0, round=round
            )
        except Exception as e:
            raise RuntimeError(f"Single-core optimisation failed: {e}")
        print("Design your experiment based on:")
        print("  tii   :", best_design_decisions['tii'])
        print("  tvi   :", best_design_decisions['tvi'])
        print("  swps  :", best_design_decisions['swps'])
        print("  St    :", best_design_decisions['St'])
        print("  pp_obj:", best_design_decisions['pp_obj'])
    return best_design_decisions



def _safe_run(args):
    """
    Executes a function safely and captures any exceptions that occur during execution.

    This function is intended to execute a single function call with specified
    arguments within a safe environment. If an exception is raised during
    execution, the exception is handled by printing an error message with the
    failed core number, followed by the traceback. In such cases, the function
    returns `None`.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        - des_opt : Any
            Descriptor optimization information.
        - system : Any
            The system to be processed.
        - models : Any
            Models or configurations associated with the process.
        - core_num : int
            An identifier for the core running this execution.
        - round : Any
            The round or iteration related to the execution.

    Returns
    -------
    Any or None
        Returns the result of the `_run_single_pp` call if successful, or `None`
        if an exception is encountered during execution.
    """
    des_opt, system, models, core_num, round = args
    configure_logger()
    try:
        return _run_single_pp(des_opt, system, models, core_num, round)
    except Exception as e:
        print(f"[Core {core_num}] FAILED: {e}")
        traceback.print_exc()
        return None


def _run_single_pp(des_opt, system, models, core_number=0, round=round):
    """
    Perform Model-Based Design of Experiments for Parameter Precision (MBDOE-PP).

    This function implements a strategy for designing experiments that maximise
    the information content of data with respect to parameter estimation.
    It supports multiple optimisation strategies—local, global, and hybrid—implemented
    in Python or C++, depending on the user's configuration.

    The algorithm structure follows the principles of parameter-precision-oriented
    design described in Franceschini and Macchietto [1]_, where the goal is to select
    experimental conditions that maximise a precision criterion—typically defined
    based on the Fisher Information Matrix (FIM). The most common criteria include:

    - **D-optimality**: Maximises the determinant of the FIM. This minimises the volume
      of the confidence ellipsoid in parameter space, improving the overall identifiability
      of parameters.

    - **A-optimality**: Minimises the trace of the inverse FIM, reducing the average variance
      across all parameter estimates.

    - **E-optimality**: Maximises the smallest eigenvalue of the FIM, ensuring the least identifiable
      parameter is still well-estimated.

    - **Modified E-optimality (ME)**: Minimises the condition number of the FIM, improving
      numerical stability and distributing sensitivity more evenly across parameters.

    Various optimisation strategies are supported to improve exploration of the design space,
    depending on the computational budget:

    - **Local optimisation** (`L`): A local search method using `trust-constr` to refine
      an initial design. Efficient, but may converge to local optima.

    - **Global optimisation** (`G_P`, `G_C`): Global search strategies such as
      differential evolution with penalised constraints (`G_P`, Python-based),
      or CasADi+IPOPT (`G_C`, C++-based) are used to explore complex, non-convex
      regions of the design space.

    - **Joint optimisation** (`GL`): Combines global and local strategies—
      using a penalised differential evolution phase for exploration and `trust-constr`
      for local refinement. This hybrid approach often yields higher-quality designs
      with better convergence properties.

    Parameters
    ----------
    des_opt : dict
        Dictionary specifying design criteria, optimisation settings, and iteration limits.
    system : dict
        Model structure including time domain, input/output definitions, and constraints.
    models : dict
        Estimation settings including active solvers, parameter values, and mutation masks.
    core_number : int
        Core index used to differentiate parallel optimisation jobs.
    round : int
        Current round number in the MBDOE-PE iterative design framework.

    Returns
    -------
    design_decisions : dict
        Dictionary containing:
        - 'tii': Time-invariant input levels.
        - 'tvi': Time-variant input profiles.
        - 'swps': Switching points (levels and times) for CVP inputs.
        - 'St': Sampling time schedule.
        - 'pp_obj': Precision criterion value.
        - 't_values': Full simulation time grid.
    pp_obj : float
        The final value of the selected precision criterion (e.g., determinant of FIM).
    swps : dict
        Switching point values for CVP input profiles.

    Notes
    -----
    This function serves as the main entry point for MBDOE-PP. It manages the entire
    workflow including optimisation strategy selection, simulation execution, and
    design result processing.

    References
    ----------
    .. [1] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.
       https://doi.org/10.1016/j.ces.2007.11.034
    """

    # --------------------------- EXTRACT DATA --------------------------- #
    tf = system['t_s'][1]
    ti = system['t_d']

    # Time-variant inputs
    tv_iphi_vars = list(system['tvi'].keys())
    tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [system['tvi'][var]['stps']+1 for var in tv_iphi_vars]
    tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
    tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
    tv_iphi_offsetl = [
        system['tvi'][var]['offl'] / system['tvi'][var]['max']
        for var in tv_iphi_vars
    ]
    tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}

    # Time-invariant inputs
    ti_iphi_vars = list(system['tii'].keys())
    ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]

    # Time-variant outputs
    tv_ophi_vars = [
        var for var in system['tvo'].keys()
        if system['tvo'][var].get('meas', True)
    ]
    tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett_ophi = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
    tv_ophi_sampling = {var: system['tvo'][var]['samp_s'] for var in tv_ophi_vars}
    tv_ophi_forcedsamples = {var: [v / tf for v in system['tvo'][var]['samp_f']] for var in tv_ophi_vars}

    # Time-invariant outputs
    ti_ophi_vars = [
        var for var in system['tio'].keys()
        if system['tio'][var].get('meas', True)
    ]

    # Solver and optimization settings
    active_solvers = models['can_m']
    estimations = models['normalized_parameters']
    ref_thetas = models['theta']
    theta_parameters = _par_update(ref_thetas, estimations)

    # Criterion, iteration, and penalty settings
    design_criteria = des_opt['pp_ob']
    maxpp = des_opt['itr']['maxpp']
    tolpp = des_opt['itr']['tolpp']
    population_size = des_opt['itr']['pps']
    eps = des_opt['eps']
    mutation = models['mutation']
    V_matrix = models['V_matrix']
    pltshow= des_opt['plt']

    # ------------------------ CHOOSE METHOD (Local vs Global) ------------------------ #

    result, index_dict = _optimiser(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
        ti_ophi_vars,
        tf, ti,
        active_solvers, theta_parameters,
        eps, maxpp, tolpp,population_size,
        mutation, V_matrix, design_criteria,
        system, models
    )

    if hasattr(result, 'X'):
        x_final = result.X
    elif hasattr(result, 'x'):
        x_final = result.x
    else:
        raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")
    # ------------------- USE FINAL SOLUTION IN _runnerpp ---------------- #
    try:
        phi, swps, St, pp_obj, t_values, tv_ophi_dict, ti_ophi_dict, phit = _pp_runner(
            x_final,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            system,
            models
        )
    except ValueError as e:
        if "MBDoE optimiser kernel was unsuccessful" in str(e):
            print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
            return None
        else:
            raise  # propagate other ValueErrors if unrelated

    # --------------------------- REPORT & SAVE --------------------------- #

    # You may have a specialized reporter or reuse the same `_reporter`
    phi, phit, swps, St = _reporter(
        phi, phit, swps, St,
        pp_obj,                  # pass the objective for logging
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf,
        design_criteria,
        round, pltshow,
        core_number
    )

    # ------------------------- RETURN FINAL DATA ------------------------- #
    design_decisions = {
        'tii': phi,
        'tvi': phit,
        'swps': swps,
        'St': St,
        'pp_obj': pp_obj,
        't_values': t_values
    }

    return design_decisions, pp_obj, swps


def _optimiser(
    tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
    tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
    ti_iphi_vars, ti_iphi_max, ti_iphi_min,
    tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
    ti_ophi_vars,
    tf, ti,
    active_models, theta_parameters,
    eps, maxpp, tolpp,population_size,
    mutation, V_matrix, design_criteria,
    system, models
):
    """
    Optimizes given variables using a differential evolution (DE) optimization algorithm, constrained by specified bounds
    and properties. This function dynamically computes bounds and constraints for multiple variable groups, segments,
    and constraints. It initializes the DE problem, sets constraints, and solves for the optimal variable configuration
    via the DE algorithm.

    Parameters
    ----------
    tv_iphi_vars : list[str]
        List of variable names for temporal variables in phi-space.
    tv_iphi_seg : list[int]
        Number of segments per temporal variable in phi-space.
    tv_iphi_max : list[float]
        List of maximum allowable values for temporal variables in phi-space.
    tv_iphi_min : list[float]
        List of minimum allowable values for temporal variables in phi-space.
    tv_iphi_const : list[str]
        Constraints applied to temporal variables in phi-space.
    tv_iphi_offsett : list[float]
        Offset values for time-based constraints in phi-space.
    tv_iphi_offsetl : list[float]
        Offset values for level-based constraints in phi-space.
    tv_iphi_cvp : dict[str, float]
        Constant values pertaining to temporal variables in phi-space.
    ti_iphi_vars : list[str]
        Names of initial condition variables in phi-space.
    ti_iphi_max : list[float]
        Maximum allowable values for initial condition variables in phi-space.
    ti_iphi_min : list[float]
        Minimum allowable values for initial condition variables in phi-space.
    tv_ophi_vars : list[str]
        Names of output variables in phi-space.
    tv_ophi_seg : list[int]
        Number of segments for each output variable in phi-space.
    tv_ophi_offsett_ophi : list[float]
        Offset values for time-based constraints in output variables.
    tv_ophi_sampling : dict[str, str]
        Sampling group mapping for output variables in phi-space.
    tv_ophi_forcedsamples : dict[str, list]
        Predefined values for forced samples of output variables.
    ti_ophi_vars : list[str]
        Names of initial condition variables for output variables.
    tf : float
        Final time for optimization.
    ti : float
        Initial time for optimization.
    active_models : list[str]
        List of active solvers used in model evaluations.
    theta_parameters : dict
        Dictionary of theta parameters used in calculations.
    eps : float
        Epsilon, a small convergence factor for optimization.
    maxpp : int
        Maximum number of generations for the DE optimizer.
    tolpp : float
        Tolerance for constraint satisfaction in the DE optimizer.
    mutation : float
        Mutation factor for the DE algorithm.
    V_matrix : numpy.ndarray
        Matrix used for design evaluation and optimization.
    design_criteria : dict
        Criteria for evaluating the design performance.
    system : object
        Representation of the system being optimized, including its properties and state.
    models : list
        Collection of models associated with system evaluation.

    Returns
    -------
    res_de : pymoo.optimize.Result
        Result of the differential evolution optimization containing the optimal solution, objective values, and
        performance metrics.
    index_dict : dict
        Dictionary mapping variable names to their corresponding indices or groups in the optimization process.
    """
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
        _pp_of,
        tv_iphi_vars=tv_iphi_vars, tv_iphi_max=tv_iphi_max,
        ti_iphi_vars=ti_iphi_vars, ti_iphi_max=ti_iphi_max,
        tv_ophi_vars=tv_ophi_vars, ti_ophi_vars=ti_ophi_vars,
        tv_iphi_cvp=tv_iphi_cvp, tv_ophi_forcedsamples=tv_ophi_forcedsamples,
        tv_ophi_sampling=tv_ophi_sampling,
        active_solvers=active_models, theta_parameters=theta_parameters,
        tf=tf, eps=eps, mutation=mutation,
        V_matrix=V_matrix, design_criteria=design_criteria,
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
        termination=('n_gen', maxpp),
        seed=None,
        verbose=True,
        constraint_tolerance=tolpp
    )

    return res_de, index_dict


def _pp_of(
    x,
    tv_iphi_vars, tv_iphi_max,
    ti_iphi_vars, ti_iphi_max,
    tv_ophi_vars, ti_ophi_vars,
    active_solvers, theta_parameters,
    tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
    tf, eps, mutation, V_matrix, design_criteria,
    index_dict,
    system,
    models
):
    """
    Similar to md_objective_function but for PP. We want to maximize the PP-criterion
    => so return -pp_obj from _runnerpp (unless it's a min-type objective).
    """
    try:
        _, _, _, pp_obj, _, _, _, _ = _pp_runner(
            x,
            tv_iphi_vars, tv_iphi_max,
            ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars,
            active_solvers, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict,
            system,
            models
        )
        if np.isnan(pp_obj):
            return 1e6
        return -pp_obj  # negative => maximize pp_obj
    except Exception as e:
        print(f"Exception in pp_objective_function: {e}")
        return 1e6

def _pp_runner(
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
    tf,
    eps,
    mutation,
    V_matrix,
    MBDOE_PP_criterion,
    index_dict,
    system,
    models
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
    system : dict
        Model structure dictionary.
    models : dict
        Additional model settings.

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
        - phit_interp (dict): Interpolated piecewise data (last model run).
    """
    # Convert x to list if not already
    if x is None:
        raise ValueError("MBDoE optimiser kernel was unsuccessful, increase iteration to avoid constraint violations.")
    x = x.tolist() if not isinstance(x, list) else x

    # ---------------------------------------------------------------------
    # 1) Extract design parameters via _slicer
    # ---------------------------------------------------------------------
    dt_real = system['t_r']
    nodes = int(round(tf / dt_real)) + 1
    tlin = np.linspace(0, 1, nodes)
    ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
    # Convert St into dict of var -> np.array
    St = {var: np.array(sorted(St[var])) for var in St}

    # Combine nd equally spaced points in [0,1] with the sampling points
    t_values_flat = [tp for times in St.values() for tp in times]

    t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

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

    # For returning model results
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
        # 3a) Run the model for unperturbed (baseline) theta
        # -----------------------------------------------------------------
        tv_out, ti_out, interp_data = simula(
            t_values, swps_data, ti_iphi_data,
            phisc, phitsc, tsc,
            theta, thetac,
            tv_iphi_cvp, {},  # pass empty dict or relevant data
            solver_name,
            system,
            models
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
            # 3b-i) Run model with modified theta
            # -------------------------------------------------------------
            tv_out_mod, ti_out_mod, _ = simula(
                t_values, swps_data, ti_iphi_data,
                phisc, phitsc, tsc,
                modified_theta, thetac,
                tv_iphi_cvp, {},  # pass empty dict or relevant data
                solver_name,
                system,
                models
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
    solver_name = active_solvers[0]  # or whichever model you want to measure
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


    logger = configure_logger()
    logger.info(f"mbdoe-MPP:{MBDOE_PP_criterion} is running with {pp_obj:.4f}")
    # ---------------------------------------------------------------------
    # 5) Return the relevant pieces
    # ---------------------------------------------------------------------
    # phit_interp might hold data only for the last model or each model in a dict
    return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp
