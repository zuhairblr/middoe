# # sc_estima.py
#
# import numpy as np
# import copy
# import scipy.linalg as la
# from middoe.iden_parmest import parmest
# from middoe.iden_uncert import uncert
# from middoe.iden_utils import plot_rCC_vs_k  # Import the plotting function
# from middoe.log_utils import  read_excel
#
#
# def estima(result, system, models, iden_opt, round):
#     """
#     Perform estimability analysis to rank parameters and determine the optimal number of parameters to estimate.
#
#     This function evaluates the estimability of model parameters by analyzing the sensitivity matrix
#     and ranking parameters based on their contribution to the model's output. It also determines the
#     optimal number of parameters to estimate using MSE-based selection criteria.
#
#     Parameters
#     ----------
#     result : dict
#         The result from the last identification (estimation - uncertainty analysis).
#     system : dict
#         User provided - The model structure information.
#     models : dict
#         User provided - The settings for the modelling process.
#     iden_opt : dict
#         User provided - The settings for the estimation process.
#     round : int
#         The current round of the design - conduction and identification procedure.
#     data : dict
#         Prior information for estimability analysis (observations, inputs, etc.).
#
#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - rankings (dict): Parameter rankings for each solver.
#         - k_optimal_values (dict): Optimal number of parameters to estimate for each solver.
#         - rCC_values (dict): Corrected critical ratios for each solver.
#         - J_k_values (dict): Objectives of weighted least square method-based optimization for each solver.
#         - best_uncert_result (dict): Best uncertainty analysis result for each solver.
#     """
#     data = read_excel()
#     rankings = {}
#     k_optimal_values = {}
#     rCC_values = {}
#     J_k_values = {}
#     iden_opt['init']= None
#     iden_opt['log']= False
#     modelsbase = copy.deepcopy(models)
#
#     for sv in models.get('can_m', []):
#         if sv in models.get('thetastart', {}):
#             modelsbase['theta'][sv] = models['thetastart'][sv]
#
#     print (f"Estimability analysis for round {round} is running")
#     for solver, res in result.items():
#         varcovorig = iden_opt['var-cov']
#         iden_opt['var-cov'] = 'M'
#         meth = iden_opt['meth']
#         Z = res['LSA']
#         n_parameters = Z.shape[1]
#
#         ranking_known = parameter_ranking(Z)
#         print(f"Parameter ranking from most estimable to least estimable for {solver} in round {round}: {ranking_known}")
#
#         k_optimal, rCC, J_k, best_uncert_result = parameter_selection(
#             n_parameters, ranking_known, system, modelsbase,
#             iden_opt, solver, round, data, result, varcovorig, meth
#         )
#         print(f"Optimal number of parameters to estimate for {solver}: {k_optimal}")
#
#         rankings[solver] = ranking_known
#         k_optimal_values[solver] = k_optimal
#         rCC_values[solver] = rCC
#         J_k_values[solver] = J_k
#         models['mutation'][solver]=best_uncert_result['results'][solver]['optimization_result']['activeparams']
#         models['theta'][solver] = best_uncert_result['results'][solver]['optimization_result']['scpr']
#         models['V_matrix'][solver] = best_uncert_result['results'][solver]['V_matrix']
#     iden_opt['log']= True
#     iden_opt['init']= None
#     iden_opt['var-cov'] = varcovorig
#
#
#     return rankings, k_optimal_values, rCC_values, J_k_values, best_uncert_result
#
#
# def parameter_ranking(Z):
#     """
#     Rank parameters by estimability via iterative orthogonalization.
#
#     At each iteration, select the column whose component
#     orthogonal to previously selected columns has maximum norm.
#
#     Parameters
#     ----------
#     Z : (m, n) array_like
#         Sensitivity matrix (m samples × n parameters).
#
#     Returns
#     -------
#     ranking : list of int
#         Parameter indices sorted from most estimable to least estimable.
#     """
#     Z = np.asarray(Z)
#     m, n = Z.shape
#
#     remaining = list(range(n))
#     ranking = []
#     # Orthonormal basis of selected columns
#     Q = np.zeros((m, 0))
#
#     for _ in range(n):
#         # Compute residual norms for all remaining columns
#         norms = []
#         for j in remaining:
#             v = Z[:, j]
#             if Q.shape[1] > 0:
#                 # project v onto span(Q) and subtract
#                 proj = Q @ (Q.T @ v)
#                 r = v - proj
#             else:
#                 r = v
#             norms.append(np.linalg.norm(r))
#
#         # pick the column with max residual norm
#         best_idx = int(np.argmax(norms))
#         j_sel = remaining.pop(best_idx)
#         ranking.append(j_sel)
#
#         # Orthonormalize the selected column and append to Q
#         v = Z[:, j_sel]
#         if Q.shape[1] > 0:
#             v = v - Q @ (Q.T @ v)
#         norm = np.linalg.norm(v)
#         if norm > 0:
#             q = v / norm
#             Q = np.hstack((Q, q[:, None]))
#         # zero vectors (if any) are simply skipped
#
#     return ranking
#
#
# def parameter_selection(n_parameters, ranking_known, system, models, iden_opt, solvera, round, data, result, varcovorig, meth):
#     """
#     Perform MSE-based selection to determine the optimal number of parameters to estimate.
#
#     This function evaluates the estimability of model parameters by iterating over subsets of parameters,
#     computing corrected critical ratios (rCC) and weighted least squares (WLS) objective values (J_k),
#     and identifying the optimal number of parameters to estimate. It also performs uncertainty analysis
#     for the selected subset of parameters.
#
#     Parameters
#     ----------
#     n_parameters : int
#         The total number of parameters.
#     ranking_known : list
#         The list of parameter indices ranked by estimability.
#     system : dict
#         User provided - The model structure information.
#     models : dict
#         User provided - The settings for the modelling process.
#     iden_opt : dict
#         User provided - The settings for the estimation process.
#     solvera : str
#         The name of the model(s).
#     round : int
#         The current round of the design - conduction and identification procedure.
#     data : dict
#         Prior information for estimability analysis (observations, inputs, etc.).
#     result : dict
#         Baseline estimation results (with all parameters free).
#     varcovorig : any
#         Original variance-covariance configuration to restore for best-case.
#     meth : str
#         The method used for parameter estimation.
#
#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - k_optimal (int): The optimal number of parameters for estimation in the ranking.
#         - rCC_values (list): Corrected critical ratios for each subset of parameters.
#         - J_k_values (list): WLS objective values for each subset of parameters.
#         - best_uncert (dict): The best uncertainty analysis result dictionary (or None if all parameters are selected).
#     """
#     rCC_values = []
#     J_k_values = []
#     uncert_results_list = []
#     J_theta = result[solvera]['WLS']
#     print(f"J_theta : {J_theta}")
#
#     # Loop over candidate parameter counts
#     for k in range(1, n_parameters):
#         # Prepare deep copies
#         system_k = copy.deepcopy(system)
#         models_k = copy.deepcopy(models)
#         models_k.pop('normalized_parameters', None)
#         iden_opt_k = copy.deepcopy(iden_opt)
#
#         # Build mask for top-k parameters
#         selected_mask = [False] * len(ranking_known)
#         for i in range(k):
#             selected_mask[ranking_known[i]] = True
#         models_k['mutation'][solvera] = selected_mask
#
#         # Estimate and compute uncertainty
#         results_k = parmest(system_k, models_k, iden_opt_k,  case='freeze')
#         # print(f'theta of parmest is {results_k[solvera]['scpr']}')
#         uncert_k = uncert( results_k, system_k, models_k, iden_opt_k, case='freeze')
#         # print(f'theta of uncert is {results_k[solvera]['optimization_result']['scpr']}')
#         uncert_results_list.append(uncert_k)
#
#         # Extract metrics
#         J_k = uncert_k['results'][solvera]['WLS']
#         n_samples = uncert_k['obs']
#         print(f"J_k {k} parameters: {J_k}")
#
#         # Compute corrected criteria
#         rC = (J_k - J_theta) / (n_parameters - k)
#         rCKub = max(rC - 1, (2 * rC) / (n_parameters - k + 2))
#         rCC = ((n_parameters - k) / n_samples) * (rCKub - 1)
#         print(f"rCC {k} parameters: {rCC}")
#
#         # Store for plotting
#         rCC_values.append(rCC)
#         J_k_values.append(J_k)
#
#     # Append rCC=0 for full set and compute x-axis
#     rCC_values.append(0)
#     x_values = list(range(1, len(rCC_values) + 1))
#
#     # Identify optimal k minimizing rCC
#     k_optimal = int(np.argmin(rCC_values) + 1)
#
#     # If all parameters are selected, no additional run
#     if k_optimal == n_parameters:
#         best_uncert = result
#     else:
#         # Re-run parmest & uncert for best k with original varcov
#         system_best = copy.deepcopy(system)
#         models_best = copy.deepcopy(models)
#         models_best.pop('normalized_parameters', None)
#         iden_opt_best = copy.deepcopy(iden_opt)
#
#         # Apply mask
#         selected_mask = [False] * len(ranking_known)
#         for i in range(k_optimal):
#             selected_mask[ranking_known[i]] = True
#         models_best['mutation'][solvera] = selected_mask
#
#         # Restore original varcov settings
#         iden_opt_best['var-cov'] = varcovorig
#         iden_opt_best['meth'] = meth
#
#         # Execute final estimation and uncertainty
#         best_results = parmest(system_best, models_best, iden_opt_best, case='freeze')
#         best_uncert = uncert( best_results, system_best, models_best, iden_opt_best, case='freeze')
#
#     # Plot results
#     plot_rCC_vs_k(x_values, rCC_values, round, solvera)
#
#     return k_optimal, rCC_values, J_k_values, best_uncert


# sc_estima.py

import numpy as np
import copy
import scipy.linalg as la
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
from middoe.iden_utils import plot_rCC_vs_k
from middoe.log_utils import read_excel


def estima(result, system, models, iden_opt, round):
    r"""
    Perform comprehensive parameter estimability analysis and subset selection.

    This function implements an advanced parameter selection strategy to address
    ill-conditioned inverse problems. It ranks parameters by estimability using
    orthogonal decomposition of the sensitivity matrix, then determines the optimal
    number of parameters to estimate using corrected critical ratio (rCC) analysis.
    This prevents over-parameterization while maintaining model fidelity.

    Parameters
    ----------
    result : dict[str, dict]
        Baseline uncertainty analysis results from all parameters free:
            - result[solver]['LSA']: Local sensitivity array (Jacobian matrix)
            - result[solver]['WLS']: Weighted least squares objective
            - Plus all other metrics from uncert()
    system : dict
        System configuration with variable definitions.
    models : dict
        Model definitions:
            - 'can_m': list[str] — Active model/solver names
            - 'theta': dict[str, list[float]] — Parameter values
            - 'mutation': dict[str, list[bool]] — Parameter masks (updated in-place)
            - 'V_matrix': dict[str, np.ndarray] — Covariance matrices (updated)
            - 'thetastart': dict[str, list[float]], optional — Initial values
    iden_opt : dict
        Identification options:
            - 'meth': str — Optimization method
            - 'var-cov': str — Covariance method (temporarily set to 'M')
            - 'init': Initial values flag (temporarily set to None)
            - 'log': Logging flag (temporarily disabled)
    round : int
        Current round number for tracking and plot labeling.

    Returns
    -------
    rankings : dict[str, list[int]]
        Parameter rankings for each solver, ordered from most to least estimable.
        Indices correspond to positions in original parameter vector.
    k_optimal_values : dict[str, int]
        Optimal number of parameters to estimate for each solver.
    rCC_values : dict[str, list[float]]
        Revised Critical Ratio values for k=1,2,...,n_params for each solver.
    J_k_values : dict[str, list[float]]
        Weighted least squares objective values for each k.
    best_uncert_result : dict
        Uncertainty analysis results for optimal parameter subset:
            - Same structure as uncert() output
            - Parameters ranked beyond k_optimal are fixed

    Side Effects
    ------------
    - Updates models['mutation'][solver] with optimal parameter mask
    - Updates models['theta'][solver] with re-estimated parameter values
    - Updates models['V_matrix'][solver] with covariance for active parameters
    - Temporarily modifies iden_opt settings (restored before return)
    - Generates rCC vs k plots via plot_rCC_vs_k()

    Notes
    -----
    **Estimability Analysis Workflow**:
        1. **Parameter Ranking**: Use orthogonal decomposition to rank parameters
           by their independent contribution to sensitivity (parameter_ranking())
        2. **Sequential Subset Selection**: For k=1,2,...,n-1, estimate parameters
           and compute corrected critical ratios
        3. **Optimal Subset**: Choose k minimizing rCC (best bias-variance trade-off)
        4. **Final Estimation**: Re-run with optimal k using original settings

    **Corrected Critical Ratio (rCC)**:
    The rCC metric quantifies the trade-off between model fit and parameter
    uncertainty. For k parameters estimated:
        \[
        rCC(k) = \frac{n_{params} - k}{n_{obs}} \left( rCK_{ub} - 1 \right)
        \]
    where rCK_{ub} is an upper bound on the corrected Kubáček criterion accounting
    for bias introduced by fixing parameters.

    **Why Subset Selection?**:
    Full parameter sets often lead to:
        - **Ill-conditioning**: High parameter correlations (multicollinearity)
        - **Overfitting**: Model captures noise rather than signal
        - **Large uncertainties**: Poor parameter determination
        - **Non-physical values**: Parameters drift to unrealistic regions

    Subset selection identifies the "estimable core" of parameters that can be
    reliably determined from available data, while fixing poorly determined
    parameters at reasonable values.

    **Orthogonal Ranking**:
    Unlike correlation-based methods, orthogonal ranking (Gram-Schmidt) ensures
    each selected parameter provides genuinely independent information. Parameters
    are ranked by the norm of their component orthogonal to previously selected
    parameters.

    **Temporary Setting Modifications**:
    During subset evaluation:
        - var-cov='M' (measurement-based, fast)
        - init=None (use previous round values)
        - log=False (suppress verbose output)

    Final estimation restores original settings for highest quality results.

    **Integration with MBDoE**:
    Estimability analysis is typically run after preliminary experiments to:
        - Identify which parameters require more informative data
        - Guide next-round experimental design
        - Prevent wasting experiments on non-estimable parameters

    References
    ----------
    .. [1] Brun, R., Reichert, P., & Künsch, H. R. (2001).
       Practical identifiability analysis of large environmental simulation models.
       *Water Resources Research*, 37(4), 1015-1030.

    .. [2] Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2011).
       Geometry of nonlinear least squares with applications to sloppy models and optimization.
       *Physical Review E*, 83(3), 036701.

    .. [3] Franceschini, G., & Macchietto, S. (2008).
       Model-based design of experiments for parameter precision: State of the art.
       *Chemical Engineering Science*, 63(19), 4846-4872.

    See Also
    --------
    parameter_ranking : Orthogonal sensitivity-based ranking.
    parameter_selection : Optimal subset selection via rCC.
    uncert : Uncertainty quantification used iteratively.

    Examples
    --------
    >>> # After initial estimation with all parameters
    >>> result_full = uncert(parmest_result, system, models, iden_opt)
    >>>
    >>> # Perform estimability analysis
    >>> rankings, k_opt, rCC, J_k, best_result = estima(
    ...     result_full, system, models, iden_opt, round=1
    ... )
    >>>
    >>> # Check results
    >>> print(f"Parameter ranking (most→least estimable): {rankings['M1']}")
    >>> # [2, 0, 1]  # e.g., θ3 most estimable, θ2 least
    >>>
    >>> print(f"Optimal number to estimate: {k_opt['M1']}")
    >>> # 2  # Estimate only top 2 parameters
    >>>
    >>> # Mutation mask has been updated
    >>> print(models['mutation']['M1'])
    >>> # [False, True, True]  # θ1 fixed, θ2-θ3 estimated
    >>>
    >>> # Use best_result for subsequent rounds
    >>> # It contains properly scaled parameters and uncertainties
    """
    data = read_excel()
    rankings = {}
    k_optimal_values = {}
    rCC_values = {}
    J_k_values = {}

    # Temporarily modify settings for fast iterative evaluation
    iden_opt['init'] = None
    iden_opt['log'] = False
    modelsbase = copy.deepcopy(models)

    # Restore initial parameter values if available
    for sv in models.get('can_m', []):
        if sv in models.get('thetastart', {}):
            modelsbase['theta'][sv] = models['thetastart'][sv]

    print(f"Estimability analysis for round {round} is running")

    for solver, res in result.items():
        varcovorig = iden_opt['var-cov']
        iden_opt['var-cov'] = 'M'  # Fast measurement-based covariance
        meth = iden_opt['meth']
        Z = res['LSA']
        n_parameters = Z.shape[1]

        # Rank parameters by estimability
        ranking_known = parameter_ranking(Z)
        print(f"Parameter ranking from most estimable to least estimable for {solver} in round {round}: {ranking_known}")

        # Determine optimal subset
        k_optimal, rCC, J_k, best_uncert_result = parameter_selection(
            n_parameters, ranking_known, system, modelsbase,
            iden_opt, solver, round, data, result, varcovorig, meth
        )
        print(f"Optimal number of parameters to estimate for {solver}: {k_optimal}")

        # Store results
        rankings[solver] = ranking_known
        k_optimal_values[solver] = k_optimal
        rCC_values[solver] = rCC
        J_k_values[solver] = J_k

        # Update model with optimal subset
        models['mutation'][solver] = best_uncert_result['results'][solver]['optimization_result']['activeparams']
        models['theta'][solver] = best_uncert_result['results'][solver]['optimization_result']['scpr']
        models['V_matrix'][solver] = best_uncert_result['results'][solver]['V_matrix']

    # Restore original settings
    iden_opt['log'] = True
    iden_opt['init'] = None
    iden_opt['var-cov'] = varcovorig

    return rankings, k_optimal_values, rCC_values, J_k_values, best_uncert_result


def parameter_ranking(Z):
    r"""
    Rank parameters by estimability using iterative orthogonalization (Gram-Schmidt).

    This function implements a greedy orthogonal decomposition of the sensitivity
    matrix to identify which parameters contribute most independently to model
    predictions. At each iteration, it selects the parameter whose sensitivity
    vector has the largest component orthogonal to previously selected parameters.

    Parameters
    ----------
    Z : np.ndarray, shape (m, n)
        Local sensitivity array (Jacobian matrix):
            - m: number of observations (measurements)
            - n: number of parameters
        Each column Z[:,j] represents the sensitivity of all observations to
        parameter j: \( Z_{ij} = \frac{\partial y_i}{\partial \theta_j} \).

    Returns
    -------
    ranking : list[int]
        Parameter indices ordered from most to least estimable. Length n.
        First index has maximum independent information, last has minimum.

    Notes
    -----
    **Algorithm (Greedy Gram-Schmidt)**:
    For k = 1 to n:
        1. For each remaining parameter j:
           - Compute residual: \( r_j = Z_j - Q Q^T Z_j \)
           - Compute norm: \( \|r_j\| \)
        2. Select j* = argmax \( \|r_j\| \)
        3. Orthonormalize: \( q = r_{j*} / \|r_{j*}\| \)
        4. Augment basis: \( Q \leftarrow [Q, q] \)

    **Interpretation**:
    The residual norm \( \|r_j\| \) quantifies how much new, independent
    information parameter j adds beyond what's already captured by selected
    parameters. Larger norms indicate more independent contributions.

    **Comparison with Other Ranking Methods**:
        - **Correlation-based**: Ranks by \( \|Z_j\| \) (ignores redundancy)
        - **Eigenvalue-based**: Ranks by principal components (expensive, global)
        - **Orthogonal (this)**: Greedy but ensures independence (local, fast)

    **Geometric Interpretation**:
    Each parameter defines a direction in observation space (\( \mathbb{R}^m \)).
    Orthogonal ranking builds a maximal volume parallelepiped by greedily
    selecting directions that expand the subspace most.

    **Zero Sensitivity Handling**:
    If a parameter has zero residual (completely redundant), it's assigned
    zero orthogonal norm and ranked last. The function handles numerical
    zeros gracefully by checking norm > 0 before orthonormalization.

    **Computational Complexity**:
    O(mn²) time for n parameters and m observations. Efficient for typical
    problem sizes (n < 20, m < 1000).

    **Numerical Stability**:
    Uses QR-based orthogonalization which is numerically stable. For very
    ill-conditioned problems, consider modified Gram-Schmidt or SVD-based
    alternatives.

    References
    ----------
    .. [1] Golub, G. H., & Van Loan, C. F. (2013).
       *Matrix Computations* (4th ed.), Section 5.2. Johns Hopkins University Press.

    See Also
    --------
    estima : Main function that uses this ranking.
    parameter_selection : Uses ranking to select optimal subset.

    Examples
    --------
    >>> # Sensitivity matrix: 100 observations × 5 parameters
    >>> Z = np.random.randn(100, 5)
    >>> Z[:, 2] += 0.8 * Z[:, 0]  # Parameter 3 correlated with parameter 1
    >>>
    >>> ranking = parameter_ranking(Z)
    >>> print(f"Most estimable parameter: θ{ranking[0]+1}")
    >>> print(f"Least estimable parameter: θ{ranking[-1]+1}")
    >>> # Likely: ranking = [0, 1, 3, 4, 2]
    >>> # Parameter 3 ranked low due to correlation with parameter 1
    >>>
    >>> # Use ranking for subset selection
    >>> k = 3  # Estimate top 3 parameters
    >>> active_params = [False] * 5
    >>> for i in range(k):
    ...     active_params[ranking[i]] = True
    >>> print(f"Active mask: {active_params}")
    >>> # [True, True, False, True, False]
    """
    Z = np.asarray(Z)
    m, n = Z.shape

    remaining = list(range(n))
    ranking = []
    # Orthonormal basis of selected columns
    Q = np.zeros((m, 0))

    for _ in range(n):
        # Compute residual norms for all remaining columns
        norms = []
        for j in remaining:
            v = Z[:, j]
            if Q.shape[1] > 0:
                # Project v onto span(Q) and subtract
                proj = Q @ (Q.T @ v)
                r = v - proj
            else:
                r = v
            norms.append(np.linalg.norm(r))

        # Pick the column with max residual norm
        best_idx = int(np.argmax(norms))
        j_sel = remaining.pop(best_idx)
        ranking.append(j_sel)

        # Orthonormalize the selected column and append to Q
        v = Z[:, j_sel]
        if Q.shape[1] > 0:
            v = v - Q @ (Q.T @ v)
        norm = np.linalg.norm(v)
        if norm > 0:
            q = v / norm
            Q = np.hstack((Q, q[:, None]))
        # Zero vectors are skipped

    return ranking


def parameter_selection(n_parameters, ranking_known, system, models, iden_opt, solvera,
                       round, data, result, varcovorig, meth):
    r"""
    Determine optimal number of parameters to estimate using corrected critical ratio.

    This function implements a systematic parameter subset selection strategy by
    evaluating models with k=1,2,...,n-1 parameters (ranked by estimability) and
    computing the corrected critical ratio (rCC) for each. The optimal k minimizes
    rCC, representing the best trade-off between model fit and parameter uncertainty.

    Parameters
    ----------
    n_parameters : int
        Total number of parameters in the model.
    ranking_known : list[int]
        Parameter indices ranked from most to least estimable (from parameter_ranking()).
    system : dict
        System configuration.
    models : dict
        Model definitions (will be modified for each k evaluation).
    iden_opt : dict
        Identification options.
    solvera : str
        Solver/model name being analyzed.
    round : int
        Current round number for plot labeling.
    data : dict
        Experimental data from read_excel().
    result : dict
        Baseline results with all parameters estimated (for comparison).
    varcovorig : str
        Original variance-covariance method to use for final estimation.
    meth : str
        Original optimization method to use for final estimation.

    Returns
    -------
    k_optimal : int
        Optimal number of parameters to estimate (1 ≤ k ≤ n).
    rCC_values : list[float]
        Revised Critical Ratio for k=1,2,...,n. Length n.
    J_k_values : list[float]
        Weighted least squares objective for k=1,2,...,n-1. Length n-1.
    best_uncert : dict
        Uncertainty analysis results for optimal k:
            - If k=n: returns baseline result (all parameters)
            - If k<n: runs new estimation with optimal subset

    Notes
    -----
    **Corrected Critical Ratio (rCC)**:
    For k parameters estimated (n-k fixed):
        \[
        rCC(k) = \frac{n-k}{n_{obs}} \left( rCK_{ub} - 1 \right)
        \]
    where:
        \[
        rCK_{ub} = \max\left( rC - 1, \frac{2 rC}{n-k+2} \right)
        \]
    and:
        \[
        rC = \frac{J_k - J_\theta}{n - k}
        \]

    **Interpretation**:
        - **rCC < 0**: Fixing parameters improves conditioning more than it hurts fit
        - **rCC ≈ 0**: Optimal balance (minimum rCC indicates best k)
        - **rCC > 0**: Fixing too many parameters degrades fit excessively

    **Selection Strategy**:
        1. Evaluate k=1,2,...,n-1 (n not needed, rCC(n)=0 by construction)
        2. Compute rCC(k) for each subset
        3. Choose k* = argmin rCC(k)
        4. Re-estimate with k* parameters using original settings

    **Why Re-estimate?**:
    During subset evaluation, fast settings are used (var-cov='M'). For the
    optimal subset, a final estimation with original settings (e.g., var-cov='H'
    for Hessian-based covariance) provides highest quality uncertainties.

    **Bias-Variance Trade-off**:
        - **Small k**: Low variance (few parameters to estimate) but high bias (fixed parameters introduce model error)
        - **Large k**: Low bias but high variance (many parameters, poor determination)
        - **Optimal k**: Minimizes total MSE = bias² + variance

    **Computational Cost**:
    Requires n-1 parameter estimations, each solving a nonlinear optimization
    problem. For expensive models, consider:
        - Using faster solver during evaluation (already done: var-cov='M')
        - Reducing iterations (iden_opt['maxit'])
        - Parallel evaluation (if multiple solvers)

    **Plot Generation**:
    Calls plot_rCC_vs_k() to visualize rCC vs k, helping interpret the selection.
    The plot typically shows a U-shape with minimum at optimal k.

    References
    ----------
    .. [1] Kubáček, L. (1996).
       On a linearization of regression models.
       *Applications of Mathematics*, 41(6), 439-456.

    .. [2] Brun, R., Reichert, P., & Künsch, H. R. (2001).
       Practical identifiability analysis of large environmental simulation models.
       *Water Resources Research*, 37(4), 1015-1030.

    See Also
    --------
    estima : Main function that calls this for each solver.
    parameter_ranking : Provides ranking_known input.
    plot_rCC_vs_k : Visualizes rCC evolution.

    Examples
    --------
    >>> # After ranking parameters
    >>> ranking = [2, 0, 1, 3]  # e.g., θ3 most estimable
    >>>
    >>> # Select optimal subset
    >>> k_opt, rCC, J_k, best_result = parameter_selection(
    ...     n_parameters=4,
    ...     ranking_known=ranking,
    ...     system=system,
    ...     models=models,
    ...     iden_opt=iden_opt,
    ...     solvera='M1',
    ...     round=1,
    ...     data=data,
    ...     result=baseline_result,
    ...     varcovorig='H',
    ...     meth='SLSQP'
    ... )
    >>>
    >>> print(f"Optimal k: {k_opt}")
    >>> # 2  (estimate top 2 parameters only)
    >>>
    >>> print(f"rCC values: {rCC}")
    >>> # [1.2, -0.3, 0.1, 0.0]  (minimum at k=2)
    >>>
    >>> # Active parameters: θ3, θ1 (indices 2, 0 from ranking)
    >>> # Fixed parameters: θ2, θ4 (indices 1, 3)
    """
    rCC_values = []
    J_k_values = []
    uncert_results_list = []
    J_theta = result[solvera]['WLS']
    print(f"J_theta (baseline): {J_theta}")

    # Loop over candidate parameter counts
    for k in range(1, n_parameters):
        # Prepare deep copies
        system_k = copy.deepcopy(system)
        models_k = copy.deepcopy(models)
        models_k.pop('normalized_parameters', None)
        iden_opt_k = copy.deepcopy(iden_opt)

        # Build mask for top-k parameters
        selected_mask = [False] * len(ranking_known)
        for i in range(k):
            selected_mask[ranking_known[i]] = True
        models_k['mutation'][solvera] = selected_mask

        # Estimate and compute uncertainty
        results_k = parmest(system_k, models_k, iden_opt_k, case='freeze')
        uncert_k = uncert(results_k, system_k, models_k, iden_opt_k, case='freeze')
        uncert_results_list.append(uncert_k)

        # Extract metrics
        J_k = uncert_k['results'][solvera]['WLS']
        n_samples = uncert_k['obs']
        print(f"J_k ({k} parameters): {J_k}")

        # Compute corrected criteria
        rC = (J_k - J_theta) / (n_parameters - k)
        rCKub = max(rC - 1, (2 * rC) / (n_parameters - k + 2))
        rCC = ((n_parameters - k) / n_samples) * (rCKub - 1)
        print(f"rCC ({k} parameters): {rCC}")

        # Store for analysis
        rCC_values.append(rCC)
        J_k_values.append(J_k)

    # Append rCC=0 for full parameter set
    rCC_values.append(0)
    x_values = list(range(1, len(rCC_values) + 1))

    # Identify optimal k minimizing rCC
    k_optimal = int(np.argmin(rCC_values) + 1)

    # If all parameters selected, use baseline result
    if k_optimal == n_parameters:
        best_uncert = result
    else:
        # Re-run for optimal k with original settings
        system_best = copy.deepcopy(system)
        models_best = copy.deepcopy(models)
        models_best.pop('normalized_parameters', None)
        iden_opt_best = copy.deepcopy(iden_opt)

        # Apply optimal mask
        selected_mask = [False] * len(ranking_known)
        for i in range(k_optimal):
            selected_mask[ranking_known[i]] = True
        models_best['mutation'][solvera] = selected_mask

        # Restore original settings for best quality
        iden_opt_best['var-cov'] = varcovorig
        iden_opt_best['meth'] = meth

        # Execute final estimation and uncertainty
        best_results = parmest(system_best, models_best, iden_opt_best, case='freeze')
        best_uncert = uncert(best_results, system_best, models_best, iden_opt_best, case='freeze')

    # Visualize results
    plot_rCC_vs_k(x_values, rCC_values, round, solvera)

    return k_optimal, rCC_values, J_k_values, best_uncert