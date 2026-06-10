# des_pp.py
import scipy.linalg
from middoe.des_utils import _slicer, _reporter, configure_logger, _write_final_results_txt
from middoe.krnl_simula import simula
from multiprocessing import Pool
import traceback
from typing import Dict, Any, Union
from middoe.sc_estima import parameter_ranking
import logging
import scipy.linalg
import pandas as pd
from pathlib import Path
import datetime
from pymoo.core.callback import Callback
from threading import Lock
import scipy.linalg
import numpy as np
from collections import defaultdict
from functools import partial
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.operators.sampling.lhs import LHS

logger = logging.getLogger(__name__)


def mbdoe_pp(
    des_opt: Dict[str, Any],
    system: Dict[str, Any],
    models: Dict[str, Any],
    insilicos: Dict[str, Any] = None,
    round: int = 0,
    num_parallel_runs: int = 1
) -> Dict[str, Union[Dict[str, Any], float]]:
    r"""
    Perform Model-Based Design of Experiments for Parameter Precision (MBDoE-PP).

    This function searches for the most informative experimental design to
    improve parameter estimation accuracy by maximising a precision criterion
    based on the Fisher Information Matrix (FIM). Optimisation can be run
    in single-core or parallel mode, and the best valid design is returned.

    Parameters
    ----------
    des_opt : dict
        Optimisation settings, including:

        - 'ppob' : str
            Precision criterion ('D', 'A', 'E', 'ME', 'ORB', 'SEM'):

            * 'D': D-optimality - Scale-invariant criterion that minimises the volume of the confidence ellipsoid by maximising det(FIM). Advantages: Strong overall precision. Disadvantages: May excessively prioritise sensitive parameters.

            * 'A': A-optimality - Minimises the average parameter variance (trace of inverse FIM). Advantages: Balances overall precision. Disadvantages: Not scale-invariant, ignores parameter correlations.

            * 'E': E-optimality - Maximises the minimum eigenvalue of FIM (minimises worst-case parameter variance). Advantages: Targets most uncertain parameter. Disadvantages: May be overly conservative, sensitive to ill-conditioning.

            * 'ME': ME-optimality (Modified E-optimality) - Reduces the disparity between the most and least identifiable directions in parameter space. Promotes uniform precision distribution and avoids elongated confidence ellipsoids. Advantages: Enhanced robustness, uniform precision across parameters. Disadvantages: Sensitive to ill-conditioning for poorly observable parameters.

            * 'ORB': Orthogonal Residual Balancing - Focuses on increasing the infromation content on poor directions in a harmonic way. Advantages: Directly targets parameter estimability. Disadvantages: May neglect overall precision, can be computationally demanding due to non-convexity.

            * 'SEM': Spectral Entropy Maximization - Uses a Shannon Entropy-inspired approach to maximise the entropy of the normalised FIM, promoting a uniform distribution of information across parameters. Advantages: Encourages balanced parameter precision, robust to scaling issues. Disadvantages: Computationally intensive for each iteration.

        - 'itr' : dict
            Iteration controls with keys 'maxpp' (int, maximum iterations), 'toldpp' (float, convergence tolerance), 'pps' (int, population size).

        - 'eps' : float
            Perturbation step size for numerical derivatives.

        - 'plt' : bool
            Enable plotting of results.

        - 'meth' : str, optional
            Optimisation method ('DE', 'PS', 'DEPS').

        - 'initial_ranking' : list[int], optional
            Pre-computed parameter ranking (set automatically if LSA available).

    system : dict
        System model and constraints:

        - 't_s' : tuple[float, float]
            Start and end times (ti, tf).

        - 't_d' : tuple[float, float]
            Restricted initial/final intervals (dead time).

        - 't_r' : float
            Time resolution for simulation.

        - 'tvi' : dict
            Time-variant input definitions with keys 'max', 'min' (float bounds), 'stps' (int, number of switching segments), 'const' (str, constraint type 'inc'/'dec'/'rel'), 'offt' (float, minimum time offset between switches), 'offl' (float, minimum level offset), 'cvp' (str, control variable profile 'CPF' piecewise-constant or 'LPF' piecewise-linear).

        - 'tii' : dict
            Time-invariant input definitions ('max', 'min').

        - 'tvo' : dict
            Time-variant output definitions with keys 'meas' (bool, include in measurement), 'sp' (int, number of sampling points), 'offt' (float, minimum time between samples), 'samp_s' (str, sampling synchronisation group), 'samp_f' (list[float], forced sampling times), 'unc' (float, measurement uncertainty standard deviation).

        - 'tio' : dict
            Time-invariant output definitions.

    models : dict
        Model definitions and settings:

        - 'can_m' : list[str]
            Active model names (typically single model for PP).

        - 'theta' : dict[str, list[float]]
            Nominal parameter values for each model.

        - 'mutation' : dict[str, list[bool]]
            Parameter masks indicating which parameters are free (True) vs fixed (False).

        - 'V_matrix' : dict[str, np.ndarray]
            Prior parameter covariance matrices (typically diagonal).

        - 'LSA' : dict[str, np.ndarray], optional
            Pre-computed local sensitivity analysis matrices for initial ranking.

    insilicos : dict, optional
        In-silico experiment settings. When ``insilicos['theta']`` is supplied,
        it is used as the true/reference parameter vector for FIM eigen-direction
        alignment reporting only.

    round : int
        Current experimental design round (used for logging/reporting).

    num_parallel_runs : int, optional
        Number of optimisation runs in parallel (default: 1, single-core mode).

    Returns
    -------
    results : dict
        Best design found with keys:

        - 'tii' : dict[str, float] - Optimal time-invariant input values.
        - 'tvi' : dict[str, Any] - Optimal time-variant input profiles.
        - 'swps' : dict[str, list[float]] - Switching times for time-variant controls.
        - 'St' : dict[str, np.ndarray] - Sampling times for each output variable.
        - 'pp_obj' : float - Parameter precision objective value (maximised).
        - 't_values' : list[float] - Full simulation time vector.

    Raises
    ------
    RuntimeError
        If all parallel runs fail or if the single-core run encounters an exception.

    Notes
    -----
    MBDoE-PP designs experiments that maximise information for parameter estimation,
    using precision-based criteria derived from the Fisher Information Matrix.

    The Fisher Information Matrix is computed as:

    .. math::
        \\mathbf{M} = \\mathbf{M}_0 + \\mathbf{M}_p = \\mathbf{Q}^T \\mathbf{\\Sigma}_y^{-1} \\mathbf{Q} + \\mathbf{V}^{-1}

    where Q is the sensitivity matrix, Sigma_y is the measurement covariance,
    and V is the prior parameter covariance.

    If pre-computed LSA matrices are available, initial parameter estimability
    ranking is computed and logged using orthogonalisation procedure.

    Supported constraints include control level bounds and switching rules
    (increasing, decreasing, relaxed), minimum sampling intervals and total
    sample limits, forced or forbidden sampling times, synchronised sampling
    across outputs, and piecewise-constant (CPF) or piecewise-linear (LPF)
    control profiles.

    References
    ----------
    .. [1] Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
       MIDDoE: An MBDoE Python package for model identification, discrimination,
       and calibration.
       *Digital Chemical Engineering*, 17, 100276.
       https://doi.org/10.1016/j.dche.2025.100276

    .. [2] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846–4872.
       https://doi.org/10.1016/j.ces.2007.11.034

    See Also
    --------
    _safe_run : Parallel optimisation wrapper with exception handling.
    _run_single_pp : Single-core optimisation routine.
    _optimiser : Core optimisation problem setup and solver.
    parameter_ranking : Compute parameter estimability ranking from LSA.

    Examples
    --------
    >>> import numpy as np
    >>> from middoe import des_pp
    >>>
    >>> # Define system configuration
    >>> system = {
    ...     'tvo': {'mu0': {'init': 1e15, 'meas': True, 'unc': 1e13, 'sp': 8}},
    ...     'tvi': {'T': {'min': 278, 'max': 313, 'stps': 3, 'cvp': 'CPF',
    ...                   'const': 'rel', 'offt': 5.0, 'offl': 1.0}},
    ...     'tii': {'C0': {'min': 0.15, 'max': 0.35}},
    ...     't_s': (0.0, 180.0), 't_d': (10.0, 10.0)
    ... }
    >>>
    >>> # Define model
    >>> models = {
    ...     'can_m': ['M1'],
    ...     'theta': {'M1': [0.42, 2.5e5, 1.2e11]},
    ...     'mutation': {'M1': [True, True, True]},
    ...     'V_matrix': {'M1': np.diag([0.01, 0.01, 0.01])}
    ... }
    >>>
    >>> # Configure MBDoE-PP
    >>> des_opt = {
    ...     'ppob': 'D', 'meth': 'DEPS',
    ...     'itr': {'maxpp': 5000, 'toldpp': 1e-8, 'pps': 40},
    ...     'eps': 0.01, 'plt': False
    ... }
    >>>
    >>> # Run MBDoE-PP
    >>> design = des_pp.mbdoe_pp(des_opt, system, models, round=2)
    >>> print(f"D-optimality value: {design['pp_obj']:.4e}")
    """

    # Preserve older positional calls: mbdoe_pp(des_opt, system, models, round).
    if isinstance(insilicos, (int, np.integer)) and round == 0:
        round = int(insilicos)
        insilicos = None

    # Set default values for new options if not provided
    if 'conv_rep' not in des_opt:
        des_opt['conv_rep'] = True  # Default: report convergence metrics
    if 'priors' not in des_opt:
        des_opt['priors'] = True  # Default: include priors (Mp) in FIM

    # Assuming active solvers and mutation info available
    active_solvers = models['can_m']
    solver_name = active_solvers[0] if active_solvers else None

    if solver_name and 'LSA' in models and solver_name in models['LSA']:
        Z_init = models['LSA'][solver_name]
        initial_ranking = parameter_ranking(Z_init)
        logger.debug(f"Initial parameter ranking (most to least estimable) for solver '{solver_name}': {initial_ranking}")
        des_opt['initial_ranking'] = initial_ranking
    else:
        logger.debug(f"No initial LSA matrix found for solver '{solver_name}' to compute initial ranking.")
    if num_parallel_runs > 1:
        with Pool(num_parallel_runs) as pool:
            results_list = pool.map(
                _safe_run,
                [(des_opt, system, models, insilicos, core_number, round)
                 for core_number in range(num_parallel_runs)]
            )

        successful = [res for res in results_list if res is not None]
        if not successful:
            raise RuntimeError(
                "All MBDOE-PP optimisation runs failed. "
                "Try increasing 'maxpp', adjusting bounds, or relaxing constraints."
            )

        # each res is now: (design_decisions, pp_obj, swps, core_number)
        best_design_decisions, best_pp_obj, best_swps, best_core = max(
            successful, key=lambda x: x[1]
        )

    else:
        try:
            # _run_single_pp returns (design_decisions, pp_obj, swps)
            best_design_decisions, best_pp_obj, best_swps = _run_single_pp(
                des_opt, system, models, insilicos=insilicos, core_number=0, round=round
            )
            best_core = 0
        except Exception as e:
            raise RuntimeError(f"Single-core optimisation failed: {e}")

        print("Design your experiment based on:")
        print("  tii   :", best_design_decisions['tii'])
        print("  tvi   :", best_design_decisions['tvi'])
        print("  swps  :", best_design_decisions['swps'])
        print("  St    :", best_design_decisions['St'])
        print("  pp_obj:", best_design_decisions['pp_obj'])

    # Write final short TXT summary in design folder
    _write_final_results_txt(
        round_num=round,
        core_number=best_core,
        design_decisions=best_design_decisions,
        pp_obj=best_pp_obj,
    )

    return best_design_decisions


def _safe_run(args):
    r"""
    Safely execute a single MBDoE-PP optimisation run with exception handling.

    This wrapper function is used for parallel execution to prevent individual
    core failures from crashing the entire optimisation. Exceptions are caught,
    logged, and returned as None. Logger is configured at the start of each run.

    Parameters
    ----------
    args : tuple
        Packed arguments for _run_single_pp:
            - des_opt : dict
                Design optimisation settings.
            - system : dict
                System configuration.
            - models : dict
                Model definitions.
            - insilicos : dict, optional
                In-silico true/reference parameter settings for reporting.
            - core_num : int
                Core/process identifier for logging.
            - round : int
                Experimental design round number.

    Returns
    -------
    result : tuple or None
        If successful: (design_decisions, pp_obj, swps).
        If failed: None.

    Notes
    -----
    This function is intended for use with multiprocessing.Pool.map().
    Exception tracebacks are printed to stderr for debugging.

    See Also
    --------
    _run_single_pp : Core single-run optimisation logic.
    mbdoe_pp : Main entry point that manages parallel execution.
    """
    if len(args) == 5:
        des_opt, system, models, core_num, round = args
        insilicos = None
    else:
        des_opt, system, models, insilicos, core_num, round = args

    configure_logger()
    try:
        # _run_single_pp returns: (design_decisions, pp_obj, swps)
        design_decisions, pp_obj, swps = _run_single_pp(
            des_opt, system, models, insilicos=insilicos, core_number=core_num, round=round
        )
        # Attach the core number so we know which worker won
        return design_decisions, pp_obj, swps, core_num
    except Exception as e:
        print(f"Core {core_num} FAILED: {e}")
        traceback.print_exc()
        return None


def _run_single_pp(des_opt, system, models, insilicos=None, core_number=0, round=0):
    import numpy as np

    tf = system['t_s'][1]
    ti = system['t_d']

    # Time-variant inputs
    tv_iphi_vars = list(system['tvi'].keys())
    tv_iphi_max = [system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_min = [system['tvi'][var]['min'] for var in tv_iphi_vars]
    tv_iphi_seg = [system['tvi'][var]['stps'] + 1 for var in tv_iphi_vars]
    tv_iphi_const = [system['tvi'][var]['const'] for var in tv_iphi_vars]
    tv_iphi_offsett = [system['tvi'][var]['offt'] / tf for var in tv_iphi_vars]
    tv_iphi_offsetl = [system['tvi'][var]['offl'] / system['tvi'][var]['max'] for var in tv_iphi_vars]
    tv_iphi_cvp = {var: system['tvi'][var]['cvp'] for var in tv_iphi_vars}

    # Time-invariant inputs
    ti_iphi_vars = list(system['tii'].keys())
    ti_iphi_max = [system['tii'][var]['max'] for var in ti_iphi_vars]
    ti_iphi_min = [system['tii'][var]['min'] for var in ti_iphi_vars]

    # Time-variant outputs
    tv_ophi_vars = [var for var in system['tvo'].keys() if system['tvo'][var].get('meas', True)]
    tv_ophi_seg = [system['tvo'][var]['sp'] for var in tv_ophi_vars]
    tv_ophi_offsett = [system['tvo'][var]['offt'] / tf for var in tv_ophi_vars]
    tv_ophi_sampling = {var: system['tvo'][var]['samp_s'] for var in tv_ophi_vars}
    tv_ophi_forcedsamples = {var: [v / tf for v in system['tvo'][var]['samp_f']] for var in tv_ophi_vars}

    # Time-invariant outputs
    ti_ophi_vars = [var for var in system['tio'].keys() if system['tio'][var].get('meas', True)]

    active_solvers = models['can_m']
    theta_parameters = models['theta']

    design_criteria = des_opt['pp_ob']
    maxpp = des_opt['itr']['maxpp']
    tolpp = des_opt['itr']['tolpp']
    population_size = des_opt['itr']['pps']
    eps = des_opt['eps']

    if 'V_matrix' in models:
        V_matrix = {solver: np.array(models['V_matrix'][solver]) for solver in models['can_m']}
    else:
        V_matrix = {solver: np.array([[1e5 if i == j else 0 for j in range(len(models['theta'][solver]))] for i in range(len(models['theta'][solver]))]) for solver in models['can_m']}

    if 'mutation' in models:
        mutation = models['mutation']
    else:
        mutation = {solver: [True] * len(models['theta'][solver]) for solver in models['can_m']}

    pltshow = des_opt['plt']
    optmethod = des_opt.get('meth', 'L')

    # Call optimizer and get metrics history
    result, index_dict, metrics_history = _optimiser(
        tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
        tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min,
        tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett, tv_ophi_sampling, tv_ophi_forcedsamples,
        ti_ophi_vars,
        tf, ti,
        active_solvers, theta_parameters,
        eps, maxpp, tolpp, population_size,
        mutation, V_matrix, design_criteria,
        system, models, optmethod,
        des_opt,
        insilicos=insilicos,
        core_number=core_number,
        round_num=round
    )

    x_final = getattr(result, 'X', getattr(result, 'x', None))
    if x_final is None:
        raise ValueError("Optimization result has neither 'X' nor 'x' attribute.")

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
            models,
            des_opt,
            insilicos=insilicos,
            metrics_history=None,
        )
    except ValueError as e:
        if "MBDoE optimiser kernel was unsuccessful" in str(e):
            print(f"[INFO] Kernel infeasibility in core {core_number}, round {round}. Skipping.")
            return None
        else:
            raise

    # Only export convergence metrics if conv_rep is True
    if des_opt.get('conv_rep', True):
        _export_metrics_to_excel(
            metrics_history,
            design_criteria,
            core_number,
            pp_obj,
            round_num=round
        )

    phi, phit, swps, St = _reporter(
        phi, phit, swps, St, pp_obj,
        t_values,
        tv_ophi_dict, ti_ophi_dict,
        tv_iphi_vars, tv_iphi_max,
        ti_iphi_vars, ti_iphi_max,
        tf, design_criteria, round, pltshow, core_number
    )

    design_decisions = {
        'tii': phi,
        'tvi': phit,
        'swps': swps,
        'St': St,
        'pp_obj': pp_obj,
        't_values': t_values
    }

    return design_decisions, pp_obj, swps


# Global lock for thread-safe Excel writing across parallel workers
excel_write_lock = Lock()

class MetricsCallback(Callback):
    def __init__(self, metrics_history, criterion, worker_num, core_number, round_num,
                 evaluation_func=None):
        super().__init__()
        self.metrics_history = metrics_history
        self.criterion = criterion
        self.evaluation_func = evaluation_func
        self.generation = 0

    def notify(self, algorithm):
        self.generation += 1

        if algorithm.pop is not None:
            best_f = algorithm.pop.get("F")
            best_x = algorithm.pop.get("X")

            if best_f is not None:
                x_curr = best_x[0] if len(best_x.shape) > 1 else best_x

                # 1. Standard Metrics
                record = {
                    "generation": self.generation,
                    "best_obj": float(best_f[0]),
                    "criterion": self.criterion
                }

                # 2. Full FIM Analysis (SEM/ORB included)
                if self.evaluation_func:
                    try:
                        fim_metrics = self.evaluation_func(x_curr)
                        record.update(fim_metrics)
                    except Exception as e:
                        print(f"FIM Calc Error: {e}")

                # 3. Non-Convexity Check (New Spectral Method)
                try:
                    # Define local objective wrapper for finite diff
                    def local_obj_wrapper(x_vec):
                        res = algorithm.problem.evaluate(x_vec.reshape(1, -1), return_values_of=["F"])
                        return float(res[0, 0])

                    curv_metrics = _estimate_hessian_spectral(
                        local_obj_wrapper, x_curr, h=1e-2, n_directions=15
                    )
                    record.update(curv_metrics)
                except Exception as e:
                    print(f"Curvature Calc Error: {e}")

                self.metrics_history.append(record)

def _fim_spectral_metrics(M: np.ndarray, eps: float = None) -> dict:
    """
    Compute robust spectral metrics of a Fisher Information Matrix M.

    Uses only strictly positive eigenvalues (> eps) for conditioning metrics.
    If no positive eigenvalues exist, treats the FIM as effectively singular.
    """
    if eps is None:
        # Slightly looser than machine epsilon to ignore numerical noise
        eps = np.sqrt(np.finfo(float).eps)

    # Full spectrum
    eigvals_full = np.linalg.eigvalsh(M)
    eig_pos = eigvals_full[eigvals_full > eps]
    eff_rank = eig_pos.size

    if eff_rank > 0:
        min_pos = float(eig_pos.min())
        max_pos = float(eig_pos.max())
        mean_pos = float(eig_pos.mean())
        cond = float(max_pos / min_pos)

        # D-opt surrogate (log-det) and A-opt surrogate (trace of inverse)
        log_det = float(np.sum(np.log(eig_pos)))
        trace_inv = float(np.sum(1.0 / eig_pos))

        total = float(eig_pos.sum())
        eigvar = float(np.var(eig_pos))

        # Gini-like coefficient of eigenvalue inequality
        if eff_rank > 1 and total > 0.0:
            sorted_pos = np.sort(eig_pos)
            i = np.arange(1, eff_rank + 1)
            gini = float(
                (2.0 * np.sum(i * sorted_pos) / (eff_rank * total))
                - (eff_rank + 1.0) / eff_rank
            )
        else:
            gini = 0.0

        # Information concentration in top directions
        if total > 0.0:
            sorted_desc = np.sort(eig_pos)[::-1]
            frac_top1 = float(sorted_desc[0] / total)
            k = min(3, eff_rank)
            frac_top3 = float(np.sum(sorted_desc[:k]) / total)
        else:
            frac_top1 = frac_top3 = 0.0
    else:
        # Effectively singular FIM
        min_pos = max_pos = mean_pos = 0.0
        cond = float("inf")
        log_det = float("-inf")
        trace_inv = float("inf")
        eigvar = 0.0
        gini = 0.0
        frac_top1 = frac_top3 = 0.0

    # Determinant and trace on full spectrum (can be 0 / small)
    det_fim = float(np.linalg.det(M))
    trace_fim = float(np.trace(M))

    metrics = {
        # Classical conditioning / rank metrics
        "condition_number": cond,
        "effective_rank": int(eff_rank),
        "num_positive_eigenvalues": int(eff_rank),
        "min_eigenvalue": min_pos,
        "max_eigenvalue": max_pos,
        "mean_eigenvalue": mean_pos,

        # Scalar strength / spread metrics
        "det_fim": det_fim,
        "trace_fim": trace_fim,
        "log_det_fim": log_det,          # log(det) over positive spectrum
        "trace_inv_fim": trace_inv,      # sum 1/λ_i (A-opt surrogate)
        "eigval_variance": eigvar,
        "eigval_gini": gini,

        # Information concentration
        "info_frac_top1": frac_top1,     # fraction of info in best direction
        "info_frac_top3": frac_top3,     # fraction in best 3 directions
    }

    # Also keep the full spectrum for any downstream analysis if needed
    metrics["_eigvals_full"] = eigvals_full
    metrics["_eigvals_pos"] = eig_pos

    return metrics


def _get_solver_vector(container, key, solver_name):
    if not isinstance(container, dict):
        return None

    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(solver_name)
    return value


def _get_true_theta_vector(des_opt, models, insilicos, solver_name):
    theta_true = _get_solver_vector(insilicos, "theta", solver_name)
    if theta_true is None and isinstance(insilicos, dict):
        theta_true = _get_solver_vector(insilicos, "theta", insilicos.get("tr_m"))
    if theta_true is None:
        theta_true = _get_solver_vector(des_opt, "theta_true", solver_name)
    if theta_true is None:
        theta_true = _get_solver_vector(models, "theta_true", solver_name)
    return theta_true


def _fim_eigen_direction_rows(
        M: np.ndarray,
        parameter_indices=None,
        solver_name=None,
        true_parameter_values=None,
        nominal_parameter_values=None,
        parameter_normalization_values=None,
):
    """
    Return FIM eigen directions as Excel-ready row dictionaries.

    Directions are sorted from largest to smallest eigenvalue. The FIM is
    symmetrized before decomposition to remove small numerical asymmetry.

    The angle to the target direction is reported as an acute angle because
    the sign of an eigenvector is arbitrary.

    FIM eigenvectors here are expressed in the normalized parameter-multiplier
    coordinates used by simula() (theta_physical = theta_norm * theta_c).
    Alignment is meaningful only when both nominal and true/reference parameter
    vectors are supplied. With normalization values, the reported target is the
    scaled correction direction:

        (theta_true - theta_nominal) / theta_normalization

    If the true/reference vector is missing, eigenvalues and eigenvectors are
    still reported, while alignment fields remain NaN and the target definition
    is marked as not_available.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return []

    if parameter_indices is None:
        parameter_indices = list(range(M.shape[0]))
    else:
        parameter_indices = list(parameter_indices)
        if len(parameter_indices) != M.shape[0]:
            parameter_indices = list(range(M.shape[0]))

    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    order = np.argsort(eigvals)[::-1]

    rows = []
    parameter_indices_text = "[" + ", ".join(str(idx) for idx in parameter_indices) + "]"
    target_direction = None
    target_direction_text = None
    target_direction_definition = "not_available"
    true_parameter_values_text = None
    nominal_parameter_values_text = None
    normalization_values_text = None

    def _select_parameter_values(values):
        if values is None:
            return None

        selected = np.asarray(values, dtype=float).ravel()
        if selected.size == M.shape[0]:
            return selected
        if parameter_indices and selected.size > max(parameter_indices):
            return selected[parameter_indices]
        return None

    try:
        true_values = _select_parameter_values(true_parameter_values)
        nominal_values = _select_parameter_values(nominal_parameter_values)
        normalization_values = _select_parameter_values(parameter_normalization_values)

        if true_values is not None:
            true_parameter_values_text = "[" + ", ".join(
                f"{float(v):.12g}" for v in true_values
            ) + "]"

        if nominal_values is not None:
            nominal_parameter_values_text = "[" + ", ".join(
                f"{float(v):.12g}" for v in nominal_values
            ) + "]"

        if normalization_values is not None:
            normalization_values_text = "[" + ", ".join(
                f"{float(v):.12g}" for v in normalization_values
            ) + "]"

        if true_values is not None and nominal_values is not None:
            correction = true_values - nominal_values
            if normalization_values is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    target_direction = np.divide(
                        correction,
                        normalization_values,
                        out=np.full_like(correction, np.nan, dtype=float),
                        where=np.abs(normalization_values) > np.finfo(float).eps,
                    )
                target_direction_definition = "(true - nominal) / normalization"
                if not np.all(np.isfinite(target_direction)):
                    target_direction = None
                    target_direction_definition = "not_available"
            else:
                target_direction = correction
                target_direction_definition = "true - nominal"

        if target_direction is not None:
            target_norm = float(np.linalg.norm(target_direction))
            if target_norm > np.finfo(float).eps:
                target_direction = target_direction / target_norm
                target_direction_text = "[" + ", ".join(
                    f"{float(v):.12g}" for v in target_direction
                ) + "]"
            else:
                target_direction = None
                target_direction_definition = "not_available"
    except (TypeError, ValueError):
        target_direction = None
        target_direction_definition = "not_available"

    for rank, eig_idx in enumerate(order, start=1):
        vector = np.asarray(eigvecs[:, eig_idx], dtype=float)

        if vector.size:
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0:
                vector = -vector

        vector_norm = float(np.linalg.norm(vector))
        angle_to_target = np.nan
        alignment_to_target = np.nan
        misalignment_to_target = np.nan
        if target_direction is not None and vector_norm > np.finfo(float).eps:
            cosine = float(np.dot(vector, target_direction) / vector_norm)
            alignment_to_target = float(np.clip(abs(cosine), 0.0, 1.0))
            angle_to_target = float(np.degrees(np.arccos(alignment_to_target)))
            misalignment_to_target = 1.0 - alignment_to_target

        row = {
            "solver": solver_name,
            "eigen_direction": f"direction_{rank}",
            "eigen_rank": rank,
            "eigenvalue": float(eigvals[eig_idx]),
            "parameter_indices": parameter_indices_text,
            "eigenvector": "[" + ", ".join(f"{float(v):.12g}" for v in vector) + "]",
            "true_parameter_values": true_parameter_values_text,
            "nominal_parameter_values": nominal_parameter_values_text,
            "parameter_normalization_values": normalization_values_text,
            "target_parameter_direction": target_direction_text,
            "target_direction_definition": target_direction_definition,
            "angle_to_target_direction_deg": angle_to_target,
            "alignment_to_target_direction_abs_cosine": alignment_to_target,
            "misalignment_to_target_direction": misalignment_to_target,
        }

        for component_idx, param_idx in enumerate(parameter_indices):
            row[f"param_{param_idx}_component"] = float(vector[component_idx])

        rows.append(row)

    return rows


def _compute_metrics_only(
        x, tv_iphi_vars, tv_iphi_max, ti_iphi_vars, ti_iphi_max,
        tv_ophi_vars, ti_ophi_vars, active_solvers, theta_parameters,
        tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
        tf, eps, mutation, V_matrix, design_criteria,
        index_dict, system, models, des_opt, insilicos=None
):
    """
    Compute FIM-based metrics for a given design vector x.
    Includes corrected ORB (condition-adaptive threshold, full QR)
    and SEM (using consistent FIM, proper entropy normalization).
    """
    try:
        if x is None:
            return {"error": "x is None"}

        x = x.tolist() if not isinstance(x, list) else x
        dt_real = system["t_r"]
        nodes = int(round(tf / dt_real)) + 1
        tlin = np.linspace(0, 1, nodes)

        # Decode design
        ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
        St = {var: np.array(sorted(St[var])) for var in St}
        tvalues_flat = [tp for times in St.values() for tp in times]
        tvalues = np.unique(np.concatenate([tlin, tvalues_flat]).tolist())

        # Setup Simulation
        LSA = defaultdict(lambda: defaultdict(dict))
        solver_name = active_solvers[0]
        theta_c = theta_parameters[solver_name]
        theta = np.ones(len(theta_c))

        # Denormalize inputs
        ti_iphi_data = ti
        swps_data = swps
        phis_c = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phits_c = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        ts_c = tf

        # 1. Nominal Simulation
        tvout, tiout, _ = simula(
            tvalues, swps_data, ti_iphi_data, phis_c, phits_c, ts_c,
            theta, theta_c, tv_iphi_cvp, {}, solver_name, system, models
        )
        y_nom = {var: np.array(tvout[var]) for var in tv_ophi_vars}
        y_nom.update({var: np.array(tiout[var]) for var in ti_ophi_vars})

        # 2. Sensitivities (finite difference)
        free_idx = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
        for p in free_idx:
            theta_mod = theta.copy()
            theta_mod[p] += eps
            tvout_m, tiout_m, _ = simula(
                tvalues, swps_data, ti_iphi_data, phis_c, phits_c, ts_c,
                theta_mod, theta_c, tv_iphi_cvp, {}, solver_name, system, models
            )
            y_mod = {var: np.array(tvout_m[var]) for var in tv_ophi_vars}
            y_mod.update({var: np.array(tiout_m[var]) for var in ti_ophi_vars})
            for var in y_nom:
                LSA[var][solver_name][p] = (y_mod[var] - y_nom[var]) / eps

        # 3. Build sensitivity matrix Q (n_obs x n_params)
        indices = {var: np.isin(tvalues, St[var]) for var in tv_ophi_vars if var in St}
        Q_rows = []
        obs_vars_order = []
        for var, mask in indices.items():
            for k, is_sampled in enumerate(mask):
                if is_sampled:
                    row = [LSA[var][solver_name][p][k] for p in free_idx]
                    Q_rows.append(row)
                    obs_vars_order.append(var)
        Q = np.array(Q_rows, dtype=float)

        # 4. Measurement covariance Σ_y
        n_obs = len(obs_vars_order)
        Sigma_y = np.zeros((n_obs, n_obs))
        for i, var in enumerate(obs_vars_order):
            unc = system.get("tvo", {}).get(var, {}).get("unc", 1.0)
            if unc is None or np.isnan(unc):
                unc = 1.0
            Sigma_y[i, i] = unc ** 2
        Sigma_y_inv = np.linalg.inv(Sigma_y)

        # 5. Parameter prior covariance (reduced to free parameters)
        V_full = V_matrix[solver_name]
        if V_full.shape[0] == len(theta_c):
            V_red = V_full[np.ix_(free_idx, free_idx)]
        else:
            V_red = V_full

        # 6. Fisher Information Matrix
        M = Q.T @ Sigma_y_inv @ Q
        if des_opt.get("priors", True):
            M += np.linalg.inv(V_red)

        # =====================================================================
        # Generic FIM spectral metrics (shared across criteria)
        # =====================================================================
        base_metrics = _fim_spectral_metrics(M)
        eigvals_full = base_metrics.pop("_eigvals_full")
        eig_pos = base_metrics.pop("_eigvals_pos")

        metrics = dict(base_metrics)
        metrics["criterion"] = design_criteria.upper()
        nominal_theta = np.asarray(theta_c, dtype=float)
        theta_true = _get_true_theta_vector(des_opt, models, insilicos, solver_name)

        metrics["_eigen_directions"] = _fim_eigen_direction_rows(
            M,
            parameter_indices=free_idx,
            solver_name=solver_name,
            true_parameter_values=theta_true,
            nominal_parameter_values=nominal_theta,
            parameter_normalization_values=nominal_theta,
        )

        # =====================================================================
        # ORB-specific metrics (corrected estimability structure)
        # =====================================================================
        if design_criteria.upper() == "ORB":
            m, n = Q.shape

            if n > 0:
                # ✅ Apply QR to full matrix Q (not pre-filtered)
                _, R, P = scipy.linalg.qr(Q, pivoting=True, mode="economic")
                diagR = np.abs(np.diag(R))
                maxd = diagR.max() if diagR.size > 0 else 1.0
                # estimability_scores = diagR / maxd if maxd > 0 else diagR
                estimability_scores = diagR

                # ✅ Use condition-adaptive threshold
                tol = max(m, n) * np.finfo(float).eps
                estimable_mask = estimability_scores > tol
                n_informative = int(estimable_mask.sum())
                estimable_values = estimability_scores[estimable_mask] if n_informative > 0 else np.array([])

                metrics.update({
                    "n_params": int(n),
                    "n_informative": int(n_informative),
                    "min_estimability": float(estimable_values.min()) if n_informative > 0 else 0.0,
                    "max_estimability": float(estimable_values.max()) if n_informative > 0 else 0.0,
                    "mean_estimability": float(estimable_values.mean()) if n_informative > 0 else 0.0,
                    "orb_cond_scaled": float(estimable_values[0] / estimable_values[-1]) if n_informative > 1 else 1.0,
                })

                # ✅ Per-parameter estimability (correctly mapped)
                if n_informative > 0:
                    for rank_idx in range(n_informative):
                        param_idx = P[rank_idx]
                        metrics[f"param_{param_idx}_estimability"] = float(estimability_scores[rank_idx])
                        metrics[f"param_{param_idx}_rank_estimability"] = rank_idx + 1
            else:
                metrics.update({
                    "n_params": 0,
                    "n_informative": 0,
                    "min_estimability": 0.0,
                    "max_estimability": 0.0,
                    "mean_estimability": 0.0,
                })

        # =====================================================================
        # SEM-specific metrics (corrected entropy / SEM-style)
        # =====================================================================
        # if design_criteria.upper() == "SEM":
        #     # ✅ Use FIM M (not Gram matrix Z.T @ Z)
        #     trace_M = np.trace(M)
        #
        #     if trace_M > np.finfo(float).eps:
        #         rho = M / trace_M
        #         eig_rho = np.linalg.eigvalsh(rho)
        #         eig_rho = eig_rho[eig_rho > np.finfo(float).eps]
        #
        #         if eig_rho.size > 0:
        #             vn_entropy = -np.sum(eig_rho * np.log(eig_rho + 1e-15))
        #
        #             # ✅ Fix entropy normalization: use number of positive eigenvalues
        #             n_dim = eig_rho.size
        #             max_entropy = np.log(n_dim) if n_dim > 1 else 0.0
        #             entropy_norm = vn_entropy / max_entropy if max_entropy > 0 else 0.0
        #
        #             purity = float(np.sum(eig_rho ** 2))
        #             participation = float(1.0 / purity) if purity > 0 else 0.0
        #             quantum_capacity = float(eig_rho.max())
        #
        #             # Fidelity to uniform state
        #             n = M.shape[0]
        #             uniform_state = np.eye(n) / n
        #             rho_sqrt = scipy.linalg.sqrtm(rho)
        #             fidelity = float(np.trace(scipy.linalg.sqrtm(rho_sqrt @ uniform_state @ rho_sqrt)))
        #             distinguishability = 0.5 * np.linalg.norm(rho - uniform_state, ord="nuc")
        #
        #             gradient_components = -(1.0 + np.log(eig_rho))
        #             grad_norm = float(np.linalg.norm(gradient_components))
        #         else:
        #             vn_entropy = max_entropy = entropy_norm = 0.0
        #             purity = participation = quantum_capacity = 0.0
        #             fidelity = distinguishability = grad_norm = 0.0
        #     else:
        #         vn_entropy = max_entropy = entropy_norm = 0.0
        #         purity = participation = quantum_capacity = 0.0
        #         fidelity = distinguishability = grad_norm = 0.0
        #
        #     metrics.update({
        #         "entropy": float(vn_entropy),
        #         "entropy_normalized": float(entropy_norm),
        #         "max_entropy": float(max_entropy),
        #         "purity": float(purity),
        #         "participation": float(participation),
        #         "quantum_capacity": float(quantum_capacity),
        #         "fidelity": float(fidelity),
        #         "distinguishability": float(distinguishability),
        #         "gradient_norm": float(grad_norm),
        #     })

        if design_criteria.upper() == "SEM":
            # ✅ Use FIM M (not Gram matrix Z.T @ Z)
            trace_M = np.trace(M)

            if trace_M > np.finfo(float).eps:
                # ✅ Step 2: Get eigenvalues of M directly (not normalized rho)
                eig_M = np.linalg.eigvalsh(M)  # 🔄 CHANGED: M instead of rho
                eig_pos = eig_M[eig_M > np.finfo(float).eps]

                if eig_pos.size > 0:
                    # ✅ Step 2: Form probability distribution by normalizing positive eigenvalues
                    eig_rho = eig_pos / eig_pos.sum()  # 🔄 CHANGED: Manual normalization

                    # ✅ Step 3: Shannon entropy (unchanged)
                    vn_entropy = -np.sum(eig_rho * np.log(eig_rho + 1e-15))

                    # ✅ Step 4: Normalized entropy (Option A)
                    n_dim = eig_rho.size
                    max_entropy = np.log(n_dim) if n_dim > 1 else 0.0
                    entropy_norm = vn_entropy / max_entropy if max_entropy > 0 else 0.0

                    purity = float(np.sum(eig_rho ** 2))
                    participation = float(1.0 / purity) if purity > 0 else 0.0
                    quantum_capacity = float(eig_rho.max())

                    # Fidelity to uniform distribution (on the probability simplex)
                    uniform_prob = np.ones(n_dim) / n_dim
                    fidelity = float(np.sum(np.sqrt(eig_rho * uniform_prob)))  # 🔄 CHANGED: Bhattacharyya coefficient
                    distinguishability = 0.5 * np.linalg.norm(eig_rho - uniform_prob, ord=1)  # 🔄 CHANGED: TV distance

                    gradient_components = -(1.0 + np.log(eig_rho + 1e-15))
                    grad_norm = float(np.linalg.norm(gradient_components))
                else:
                    vn_entropy = max_entropy = entropy_norm = 0.0
                    purity = participation = quantum_capacity = 0.0
                    fidelity = distinguishability = grad_norm = 0.0
            else:
                vn_entropy = max_entropy = entropy_norm = 0.0
                purity = participation = quantum_capacity = 0.0
                fidelity = distinguishability = grad_norm = 0.0

            metrics.update({
                "entropy": float(vn_entropy),
                "entropy_normalized": float(entropy_norm),
                "max_entropy": float(max_entropy),
                "purity": float(purity),
                "participation": float(participation),
                "quantum_capacity": float(quantum_capacity),
                "fidelity": float(fidelity),
                "distinguishability": float(distinguishability),
                "gradient_norm": float(grad_norm),
            })

        # Optional: per-parameter eigenvalues & percentages from full spectrum
        if eigvals_full.size > 0:
            total_full = float(eigvals_full.sum())
            order = np.argsort(eigvals_full)[::-1]
            for r, idx in enumerate(order):
                val = float(eigvals_full[idx])
                metrics[f"param_{idx}_eigenvalue"] = val
                metrics[f"param_{idx}_percentage"] = 100.0 * val / total_full if total_full > 0 else 0.0
                metrics[f"param_{idx}_rank"] = r + 1

        return metrics

    except Exception as e:
        print(f"[ERROR] Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


_excel_write_lock = Lock()


def _excel_writer_engine():
    for engine, module_name in (("openpyxl", "openpyxl"), ("xlsxwriter", "xlsxwriter")):
        try:
            __import__(module_name)
            return engine
        except ImportError:
            continue

    raise ImportError(
        "Excel report generation requires openpyxl or XlsxWriter. "
        "Install the reporting dependencies from requirements.txt."
    )


def _export_metrics_to_excel(metrics_history, criterion, worker_num, last_pp_obj, round_num=0, core_number=0):
    with _excel_write_lock:
        design_folder = Path.cwd() / 'design'
        design_folder.mkdir(parents=True, exist_ok=True)
        filename = f"round_{round_num:02d}_{criterion}_worker_{worker_num}.xlsx"
        filepath = design_folder / filename

        df = pd.DataFrame(metrics_history or [])
        if "generation" not in df.columns and "iteration" in df.columns:
            df = df.copy()
            df["generation"] = df["iteration"]

        summary_df = pd.DataFrame([{
            "round": round_num,
            "criterion": criterion,
            "worker": worker_num,
            "core_number": core_number,
            "final_pp_obj": last_pp_obj,
            "metric_records": len(metrics_history or []),
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }])

        eigen_direction_rows = []
        for record in metrics_history or []:
            for eigen_row in record.get("_eigen_directions", []) or []:
                row = {
                    "generation": record.get("generation", record.get("iteration")),
                    "criterion": record.get("criterion", criterion),
                    "best_obj": record.get("best_obj", record.get("objective")),
                }
                row.update(eigen_row)
                eigen_direction_rows.append(row)

        # Separate columns by category – extended to include richer FIM metrics
        conv_cols = [
            c for c in df.columns
            if not str(c).startswith("_")
            if any(x in c for x in [
                'generation', 'obj', 'eigen', 'cond', 'rank',
                'det_fim', 'trace_fim', 'trace_inv_fim', 'log_det_fim',
                'eigval_variance', 'eigval_gini', 'info_frac'
            ])
        ]
        orb_sem_cols = [c for c in df.columns if any(x in c for x in ['orb_', 'sem_', 'entropy'])]
        curv_cols = [c for c in df.columns if 'curv_' in c or 'non_convexity' in c]

        with pd.ExcelWriter(filepath, engine=_excel_writer_engine()) as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            if conv_cols:
                df[conv_cols].to_excel(writer, sheet_name='Convergence', index=False)

            if orb_sem_cols:
                index_cols = ["generation"] if "generation" in df.columns else []
                df[index_cols + orb_sem_cols].to_excel(
                    writer, sheet_name='ORB_SEM_Metrics', index=False
                )

            if curv_cols:
                index_cols = ["generation"] if "generation" in df.columns else []
                df[index_cols + curv_cols].to_excel(
                    writer, sheet_name='Non_Convexity', index=False
                )

            if eigen_direction_rows:
                pd.DataFrame(eigen_direction_rows).to_excel(
                    writer, sheet_name='FIM_Eigen_Directions', index=False
                )

        print(f"[INFO] Exported metrics to {filename}")
        return filepath



def _optimiser(
    tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const,
    tv_iphi_offsett, tv_iphi_offsetl, tv_iphi_cvp,
    ti_iphi_vars, ti_iphi_max, ti_iphi_min,
    tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett_ophi, tv_ophi_sampling, tv_ophi_forcedsamples,
    ti_ophi_vars,
    tf, ti,
    active_models, theta_parameters,
    eps, maxpp, tolpp, population_size,
    mutation, V_matrix, design_criteria,
    system, models, optmethod,
    des_opt,
    insilicos=None,
    core_number=0,
    round_num=0
):
    bounds = []
    x0 = []
    index_dict = {
        'ti': {},
        'swps': {},
        'st': {}
    }

    # Time-invariant inputs bounds and indexing
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
        index_dict=index_dict, system=system, models=models,
        des_opt=des_opt,
        insilicos=insilicos
    )

    metrics_history = []

    def evaluate_full_metrics(x):
        return _compute_metrics_only(
            x, tv_iphi_vars, tv_iphi_max, ti_iphi_vars, ti_iphi_max,
            tv_ophi_vars, ti_ophi_vars, active_models, theta_parameters,
            tv_iphi_cvp, tv_ophi_forcedsamples, tv_ophi_sampling,
            tf, eps, mutation, V_matrix, design_criteria,
            index_dict, system, models, des_opt, insilicos
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
                    diff = x[i2]-x[i1] if const == 'inc' else x[i1]-x[i2]
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

    metrics_history = []

    callback = MetricsCallback(
        metrics_history=metrics_history,
        criterion=design_criteria,
        worker_num=core_number,
        core_number=core_number,
        round_num=round_num,
        evaluation_func=evaluate_full_metrics,  # NEW
    )

    if optmethod == 'PS':
        algorithm = PatternSearch()
        res_refine = pymoo_minimize(problem, algorithm, termination=('n_gen', maxpp), callback=callback, verbose=True)
    elif optmethod == 'DE':
        algorithm = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.7)
        res_refine = pymoo_minimize(problem, algorithm,
                                   termination=('n_gen', maxpp),
                                   callback=callback,
                                   verbose=True,
                                   constraint_tolerance=tolpp,
                                   save_history=True)
    elif optmethod == 'DEPS':
        algorithm_de = DE(pop_size=population_size, sampling=LHS(), variant='DE/rand/1/bin', CR=0.9)
        metrics_history_de = []
        callback_de = MetricsCallback(metrics_history=metrics_history_de,
                                      criterion=design_criteria,
                                      worker_num=core_number,
                                      core_number=core_number,
                                      round_num=round_num,
                                      evaluation_func=evaluate_full_metrics)
        res_de = pymoo_minimize(problem, algorithm_de,
                               termination=('n_gen', maxpp // 2),
                               callback=callback_de,
                               verbose=True,
                               constraint_tolerance=tolpp,
                               save_history=True)
        metrics_history.extend(metrics_history_de)
        algorithm_refine = PatternSearch()
        callback_ps = MetricsCallback(metrics_history=metrics_history,
                                     criterion=design_criteria,
                                     worker_num=core_number,
                                     core_number=core_number,
                                     round_num=round_num,
                                     evaluation_func=evaluate_full_metrics)
        res_refine = pymoo_minimize(problem, algorithm_refine,
                                   termination=('n_gen', int(maxpp)),
                                   x0=res_de.X,
                                   callback=callback_ps,
                                   verbose=True,
                                   constraint_tolerance=tolpp)
    else:
        raise ValueError("optmethod must be 'PS', 'DE', or 'DEPS'")

    return res_refine, index_dict, metrics_history


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
    models,
    des_opt,
    insilicos=None,
    metrics_history=None,
    iteration=None,
):
    """
    Objective function wrapper for MBDoE-PP.
    This version EXACTLY matches your _pp_runner signature.
    It also logs every evaluation into metrics_history.
    """

    try:
        (
            _,   # design
            _,   # dynamics
            _,   # internal info
            pp_obj,  # objective
            _, _, _, _
        ) = _pp_runner(
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
            design_criteria,  # MBDOE_PP_criterion
            index_dict,
            system,
            models,
            des_opt,
            insilicos=insilicos,
            metrics_history=metrics_history,
            iteration=iteration
        )

        # -------------------------------------------------------------
        # Handle NaNs
        # -------------------------------------------------------------
        if pp_obj is None or np.isnan(pp_obj):
            print(f"[Warning] NaN objective at iteration {iteration}")
            f_val = 1e6
        else:
            f_val = -float(pp_obj)  # pymoo minimises → negative

        # -------------------------------------------------------------
        # LOG evaluation
        # -------------------------------------------------------------
        if metrics_history is not None:
            metrics_history.append({
                "iteration": iteration,
                "objective": pp_obj,
                "neg_objective": f_val,
                "x": x.copy(),
            })

        return f_val

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Exception in _pp_of at iter {iteration}: {e}")
        traceback.print_exc()

        # Log even failures
        if metrics_history is not None:
            metrics_history.append({
                "iteration": iteration,
                "objective": None,
                "neg_objective": None,
                "x": x.copy(),
                "error": str(e),
            })

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
        models,
        des_opt,
        insilicos=None,
        metrics_history=None,
        iteration=None
):
    if x is None:
        raise ValueError("MBDoE optimiser kernel was unsuccessful, increase iteration to avoid constraint violations.")
    x = x.tolist() if not isinstance(x, list) else x

    dt_real = system['t_r']
    nodes = int(round(tf / dt_real)) + 1
    tlin = np.linspace(0, 1, nodes)

    ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
    St = {var: np.array(sorted(St[var])) for var in St}
    t_values_flat = [tp for times in St.values() for tp in times]
    t_values = np.unique(np.concatenate((tlin, t_values_flat))).tolist()

    LSA = defaultdict(lambda: defaultdict(dict))
    J_dot_matrix = defaultdict(
        lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                          len(theta_parameters[active_solvers[0]])))
    )
    indices = {var: np.isin(t_values, St[var]) for var in tv_ophi_vars if var in St}
    M0 = defaultdict(
        lambda: np.zeros((len(theta_parameters[active_solvers[0]]),
                          len(theta_parameters[active_solvers[0]])))
    )
    M = {}
    tv_ophi, ti_ophi, phit_interp = {}, {}, {}
    free_params_by_solver = {}

    for solver_name in active_solvers:
        thetac = theta_parameters[solver_name]
        theta = np.array([1.0] * len(thetac))
        ti_iphi_data = ti
        swps_data = swps
        phisc = {var: ti_iphi_max[i] for i, var in enumerate(ti_iphi_vars)}
        phitsc = {var: tv_iphi_max[i] for i, var in enumerate(tv_iphi_vars)}
        tsc = tf

        tv_out, ti_out, interp_data = simula(t_values, swps_data, ti_iphi_data,
                                             phisc, phitsc, tsc,
                                             theta, thetac,
                                             tv_iphi_cvp, {}, solver_name,
                                             system, models)
        tv_ophi[solver_name] = tv_out
        ti_ophi[solver_name] = ti_out
        phit_interp = interp_data

        y_values_dict = {var: np.array(tv_out[var]) for var in tv_ophi_vars}
        y_values_dict.update({var: np.array([ti_out[var]]) for var in ti_ophi_vars})
        y_combined = dict(y_values_dict)

        free_params_indices = [i for i, is_free in enumerate(mutation[solver_name]) if is_free]
        free_params_by_solver[solver_name] = free_params_indices

        # Finite-difference sensitivities
        for para_idx in free_params_indices:
            modified_theta = theta.copy()
            modified_theta[para_idx] += eps

            tv_out_mod, ti_out_mod, _ = simula(t_values, swps_data, ti_iphi_data,
                                               phisc, phitsc, tsc,
                                               modified_theta, thetac,
                                               tv_iphi_cvp, {}, solver_name,
                                               system, models)

            y_i_values_dict = {var: np.array(tv_out_mod[var]) for var in tv_ophi_vars}
            y_i_values_dict.update({var: np.array([ti_out_mod[var]]) for var in ti_ophi_vars})
            y_modified_combined = dict(y_i_values_dict)

            for var in indices:
                LSA[var][solver_name][para_idx] = (
                                                          y_modified_combined[var] - y_combined[var]
                                                  ) / eps

        # Build sensitivity matrix Q
        rows = []
        obs_vars = []

        for var in indices:
            mask = indices[var]
            for k, is_meas in enumerate(mask):
                if is_meas:
                    row = [LSA[var][solver_name][p][k] for p in free_params_indices]
                    rows.append(row)
                    obs_vars.append(var)

        solver_LSA = np.array(rows, dtype=float)  # Q matrix

        # Build measurement covariance Sigma_y
        std_dev = {}
        for grp, keys in [('tvo', tv_ophi_vars), ('tio', ti_ophi_vars)]:
            for var in keys:
                unc = system.get(grp, {}).get(var, {}).get('unc', 1.0)
                if unc is None or np.isnan(unc):
                    unc = 1.0
                std_dev[var] = unc

        N_obs = len(obs_vars)
        Sigma_y = np.zeros((N_obs, N_obs))
        for i, var in enumerate(obs_vars):
            Sigma_y[i, i] = std_dev.get(var, 1.0) ** 2
        Sigma_y_inv = np.linalg.inv(Sigma_y)

        # Fisher Information Matrix from measurements
        Q = solver_LSA
        M_fisher = Q.T @ Sigma_y_inv @ Q

        if J_dot_matrix[solver_name].shape != M_fisher.shape:
            J_dot_matrix[solver_name] = np.zeros_like(M_fisher)
        J_dot_matrix[solver_name] += M_fisher
        M0[solver_name] = J_dot_matrix[solver_name]

        # Reduce V_matrix to free parameters
        if V_matrix[solver_name].shape[0] == len(free_params_indices):
            reduced_V_matrix = V_matrix[solver_name]
        elif V_matrix[solver_name].shape[0] == len(theta_parameters[solver_name]):
            reduced_V_matrix = V_matrix[solver_name][np.ix_(free_params_indices, free_params_indices)]
        else:
            raise ValueError(
                f"V_matrix shape {V_matrix[solver_name].shape} doesn't match either free params or total params"
            )

        # Regularize prior covariance before inverting
        alpha = 1e-8
        V_reg = reduced_V_matrix + alpha * np.eye(reduced_V_matrix.shape[0])
        Mp = np.linalg.inv(V_reg)

        # Build final FIM (with or without priors)
        if des_opt.get('priors', True):
            M[solver_name] = M0[solver_name] + Mp
        else:
            M[solver_name] = M0[solver_name]

    # =====================================================================
    # CRITERION-SPECIFIC OBJECTIVE COMPUTATION
    # =====================================================================
    solver_name = active_solvers[0] if active_solvers else None

    if MBDOE_PP_criterion == 'D':
        pp_obj = np.linalg.det(M[solver_name])

    elif MBDOE_PP_criterion == 'A':
        pp_obj = np.trace(M[solver_name])

    elif MBDOE_PP_criterion == 'E':
        pp_obj = np.min(np.linalg.eigvalsh(M[solver_name]))

    elif MBDOE_PP_criterion == 'ME':
        pp_obj = -np.linalg.cond(M[solver_name])

    elif MBDOE_PP_criterion == 'ORB':
        # QR decomposition with column pivoting
        m, n = Q.shape

        _, R, P = scipy.linalg.qr(Q, pivoting=True, mode='economic')
        diagR = np.abs(np.diag(R))
        maxd = diagR.max() if diagR.size > 0 else 1.0
        # estimability_scores = diagR / maxd if maxd > 0 else diagR
        estimability_scores = diagR

        # Filter only at machine precision
        tol = max(m, n) * np.finfo(float).eps
        estimable_mask = estimability_scores > tol
        effective_rank = int(estimable_mask.sum())
        estimable_scores = estimability_scores[estimable_mask] if effective_rank > 0 else np.array([])

        # Harmonic mean objective
        pp_obj = effective_rank / np.sum(1.0 / estimable_scores) if effective_rank > 0 else 0.0

        # Store metrics - FIX: Use filtered scores for condition number
        ninformative = effective_rank
        cond_scaled = (estimable_scores[0] / estimable_scores[-1]
                       if effective_rank > 1 else 1.0)


    # elif MBDOE_PP_criterion == 'SEM':
    #     # ✅ FIX 4-6: Use consistent FIM M (not Z.T @ Z), fix entropy normalization
    #     trace_M = np.trace(M[solver_name])
    #
    #     if trace_M > np.finfo(float).eps:
    #         # Normalize FIM to density matrix
    #         rho = M[solver_name] / trace_M
    #         eigvals = np.linalg.eigvalsh(rho)
    #         eigvals_positive = eigvals[eigvals > np.finfo(float).eps]
    #
    #         if eigvals_positive.size > 0:
    #             # Von Neumann entropy
    #             shannon_entropy = -np.sum(eigvals_positive * np.log(eigvals_positive + 1e-15))
    #
    #             # ✅ FIX 5: Maximum entropy based on number of positive eigenvalues
    #             n_effective = len(eigvals_positive)
    #             max_entropy = np.log(n_effective) if n_effective > 1 else 0.0
    #
    #             # Normalized entropy (objective to maximize)
    #             pp_obj = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    #
    #             # Additional QEM metrics
    #             purity = float(np.sum(eigvals_positive ** 2))
    #             participation = float(1.0 / purity) if purity > 0 else 0.0
    #             quantum_capacity = float(eigvals_positive.max())
    #
    #             # Fidelity to maximally mixed state
    #             n_full = M[solver_name].shape[0]
    #             uniform_state = np.eye(n_full) / n_full
    #             rho_sqrt = scipy.linalg.sqrtm(rho)
    #             fidelity = float(np.trace(scipy.linalg.sqrtm(rho_sqrt @ uniform_state @ rho_sqrt)))
    #             distinguishability = 0.5 * np.linalg.norm(rho - uniform_state, ord="nuc")
    #
    #             # Gradient norm
    #             gradient_components = -(1.0 + np.log(eigvals_positive))
    #             gradient_norm = float(np.linalg.norm(gradient_components))
    #         else:
    #             pp_obj = shannon_entropy = max_entropy = 0.0
    #             purity = participation = quantum_capacity = 0.0
    #             fidelity = distinguishability = gradient_norm = 0.0
    #     else:
    #         pp_obj = shannon_entropy = max_entropy = 0.0
    #         purity = participation = quantum_capacity = 0.0
    #         fidelity = distinguishability = gradient_norm = 0.0

    elif MBDOE_PP_criterion == 'SEM':
        # ✅ FIX 4-6: Use consistent FIM M (not Z.T @ Z), fix entropy normalization
        trace_M = np.trace(M[solver_name])

        if trace_M > np.finfo(float).eps:
            # ✅ Step 2: Get eigenvalues of M directly (not normalized rho)
            eigvals = np.linalg.eigvalsh(M[solver_name])  # 🔄 CHANGED: M instead of rho
            eigvals_positive = eigvals[eigvals > np.finfo(float).eps]

            if eigvals_positive.size > 0:
                # ✅ Step 2: Form probability distribution by normalizing positive eigenvalues
                eigvals_normalized = eigvals_positive / eigvals_positive.sum()  # 🔄 CHANGED: Manual normalization

                # ✅ Step 3: Von Neumann entropy (unchanged)
                shannon_entropy = -np.sum(eigvals_normalized * np.log(eigvals_normalized + 1e-15))

                # ✅ Step 4: Maximum entropy and normalized objective
                n_effective = len(eigvals_normalized)
                max_entropy = np.log(n_effective) if n_effective > 1 else 0.0

                # Normalized entropy (objective to maximize) - Option A
                pp_obj = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

                # Additional QEM metrics (on probability distribution)
                purity = float(np.sum(eigvals_normalized ** 2))
                participation = float(1.0 / purity) if purity > 0 else 0.0
                quantum_capacity = float(eigvals_normalized.max())

                # Fidelity to uniform probability distribution
                uniform_prob = np.ones(n_effective) / n_effective  # 🔄 CHANGED: Uniform probability vector
                fidelity = float(
                    np.sum(np.sqrt(eigvals_normalized * uniform_prob)))  # 🔄 CHANGED: Bhattacharyya coefficient
                distinguishability = 0.5 * np.linalg.norm(eigvals_normalized - uniform_prob,
                                                          ord=1)  # 🔄 CHANGED: Total variation distance

                # Gradient norm
                gradient_components = -(1.0 + np.log(eigvals_normalized + 1e-15))
                gradient_norm = float(np.linalg.norm(gradient_components))
            else:
                pp_obj = shannon_entropy = max_entropy = 0.0
                purity = participation = quantum_capacity = 0.0
                fidelity = distinguishability = gradient_norm = 0.0
        else:
            pp_obj = shannon_entropy = max_entropy = 0.0
            purity = participation = quantum_capacity = 0.0
            fidelity = distinguishability = gradient_norm = 0.0

    else:
        pp_obj = None

    # =====================================================================
    # METRICS COLLECTION (for convergence tracking)
    # =====================================================================
    spec_metrics = _fim_spectral_metrics(M[solver_name])
    eigvals_full = spec_metrics.pop("_eigvals_full")
    eigvals_positive = spec_metrics.pop("_eigvals_pos")

    metrics = {"criterion": MBDOE_PP_criterion, "objective": pp_obj}
    metrics.update(spec_metrics)
    nominal_theta = np.asarray(theta_parameters[solver_name], dtype=float)
    theta_true = _get_true_theta_vector(des_opt, models, insilicos, solver_name)

    metrics["_eigen_directions"] = _fim_eigen_direction_rows(
        M[solver_name],
        parameter_indices=free_params_by_solver.get(solver_name),
        solver_name=solver_name,
        true_parameter_values=theta_true,
        nominal_parameter_values=nominal_theta,
        parameter_normalization_values=nominal_theta,
    )

    # Per-parameter eigenvalue diagnostics
    if eigvals_full.size > 0:
        total_full = float(eigvals_full.sum())
        sorted_indices = np.argsort(eigvals_full)[::-1]
        for rank, idx in enumerate(sorted_indices):
            eigenvalue = float(eigvals_full[idx])
            metrics[f"param_{idx}_eigenvalue"] = eigenvalue
            metrics[f"param_{idx}_percentage"] = (
                100.0 * eigenvalue / total_full if total_full > 0 else 0.0
            )
            metrics[f"param_{idx}_rank"] = rank + 1

    # Add ORB-specific metrics
    if MBDOE_PP_criterion == "ORB":
        metrics.update({
            "n_params": int(n),
            "n_informative": int(ninformative),
            "min_estimability": float(estimable_scores.min()) if effective_rank > 0 else 0.0,
            "max_estimability": float(estimable_scores.max()) if effective_rank > 0 else 0.0,
            "mean_estimability": float(estimable_scores.mean()) if effective_rank > 0 else 0.0,
            "orb_cond_scaled": float(cond_scaled),
        })
        if effective_rank > 0:
            for idx in range(effective_rank):
                param_idx = P[idx]
                metrics[f"param_{param_idx}_estimability"] = float(estimability_scores[idx])
                metrics[f"param_{param_idx}_rank_estimability"] = idx + 1

    # Add SEM-specific metrics
    if MBDOE_PP_criterion == "SEM" and trace_M >= np.finfo(float).eps:
        metrics.update({
            "entropy": float(shannon_entropy),
            "entropy_normalized": float(pp_obj),
            "max_entropy": float(max_entropy),
            "purity": float(purity),
            "participation": float(participation),
            "quantum_capacity": float(quantum_capacity),
            "fidelity": float(fidelity),
            "distinguishability": float(distinguishability),
            "gradient_norm": float(gradient_norm),
        })

    if metrics_history is not None:
        if iteration is not None:
            metrics["iteration"] = iteration
        metrics_history.append(metrics)

    return ti, swps, St, pp_obj, t_values, tv_ophi, ti_ophi, phit_interp


def _estimate_hessian_spectral(
        objective_func,
        x_best,
        h=1e-3,
        tol_neg=1e-8,
        n_directions=20,  # Number of random directions to sample
        random_state=None,
):
    """
    Estimate non-convexity via directional second derivatives.

    Instead of just diagonals, this samples random directions on the unit sphere
    to find 'hidden' negative curvature (saddle points) that are not axis-aligned.

    Returns dict with both diagonal and spectral non-convexity metrics.
    """
    x_best = np.asarray(x_best, dtype=float).ravel()
    n = x_best.size

    if n == 0:
        return {}

    # 1. Diagonal Hessian (Legacy/Cheap Check)
    H_diag = np.zeros(n)
    f0 = float(objective_func(x_best))

    for i in range(n):
        e_i = np.zeros_like(x_best)
        e_i[i] = 1.0
        try:
            f_plus = float(objective_func(x_best + h * e_i))
            f_minus = float(objective_func(x_best - h * e_i))
            H_diag[i] = (f_plus - 2.0 * f0 + f_minus) / (h ** 2)
        except Exception:
            H_diag[i] = np.nan

    # 2. Spectral Sampling (Random Directions)
    rng = np.random.default_rng(random_state)
    dir_curvatures = []

    # Create random unit vectors
    if n_directions > 0:
        rand_dirs = rng.normal(size=(n_directions, n))
        # Normalize to unit sphere
        norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_dirs = rand_dirs / (norms + 1e-16)

        for v in rand_dirs:
            try:
                # Directional derivative approx: v^T H v
                f_p = float(objective_func(x_best + h * v))
                f_m = float(objective_func(x_best - h * v))
                curv = (f_p - 2.0 * f0 + f_m) / (h ** 2)
                dir_curvatures.append(curv)
            except:
                continue

    dir_curvatures = np.array(dir_curvatures)

    # 3. Aggregated Metrics
    return {
        # Axis-aligned metrics (Diagonal)
        "curv_diag_num_neg": int(np.sum(H_diag < -tol_neg)),
        "curv_diag_min": float(np.nanmin(H_diag)),

        # Spectral metrics (Directional)
        "curv_spec_min": float(np.min(dir_curvatures)) if len(dir_curvatures) > 0 else 0.0,
        "curv_spec_num_neg": int(np.sum(dir_curvatures < -tol_neg)),
        "curv_spec_prob_neg": float(np.mean(dir_curvatures < -tol_neg)) if len(dir_curvatures) > 0 else 0.0,

        # Combined "Godzilla" Metric (Worst curvature found anywhere)
        "non_convexity_index": min(float(np.nanmin(H_diag)),
                                   float(np.min(dir_curvatures)) if len(dir_curvatures) > 0 else 0.0)
    }
