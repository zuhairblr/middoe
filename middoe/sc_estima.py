import numpy as np
import copy
import scipy.linalg as la
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
from middoe.iden_utils import plot_rCC_vs_k  # Import the plotting function

def estima(result, system, models, iden_opt, round, data):
    """
    Perform estimability analysis to rank parameters and determine the optimal number of parameters to estimate.

    Parameters:
    result (dict): The result from the last identification (estimation - uncertainty analysis).
    system (dict): User provided - The model structure information.
    models (dict): User provided - The settings for the modelling process.
    iden_opt (dict): User provided - The settings for the estimation process.
    round (int): The current round of the design - conduction and identification procedure.
    framework_settings (dict): User provided - The settings for the framework.
    data (dict): prior information for estimability analysis (observations, inputs, etc.).
    run_solver (function): The function to run the model (simulator-bridger).

    Returns:
    tuple: A tuple containing rankings, rCC values (corrected critical ratios), and J_k values (objectives of weighted least square method based optimization).
    """
    rankings = {}
    k_optimal_values = {}
    rCC_values = {}
    J_k_values = {}
    iden_opt['init']= None
    iden_opt['log']= False
    modelsbase = copy.deepcopy(models)
    # # Override theta in modelsbase with thetastart
    # for sv in models.get('can_m', []):
    #     if sv in models.get('thetastart', {}):
    #         scpr_result = result.get(sv, {}).get('optimization_result', {}).get(sv, {})
    #         if 'scpr' in scpr_result:
    #             modelsbase['theta'][sv] = scpr_result['scpr_raw']
    for sv in models.get('can_m', []):
        if sv in models.get('thetastart', {}):
            modelsbase['theta'][sv] = models['thetastart'][sv]

    print (f"Estimability analysis for round {round} is running")
    for solver, res in result.items():
        varcovorig = iden_opt['var-cov']
        iden_opt['var-cov'] = 'M'
        meth = iden_opt['meth']
        Z = res['LSA']
        n_parameters = Z.shape[1]

        ranking_known = parameter_ranking(Z)
        print(f"Parameter ranking from most estimable to least estimable for {solver} in round {round}: {ranking_known}")

        k_optimal, rCC, J_k, best_uncert_result = parameter_selection(
            n_parameters, ranking_known, system, modelsbase,
            iden_opt, solver, round, data, result, varcovorig, meth
        )
        print(f"Optimal number of parameters to estimate for {solver}: {k_optimal}")

        rankings[solver] = ranking_known
        k_optimal_values[solver] = k_optimal
        rCC_values[solver] = rCC
        J_k_values[solver] = J_k
        models['mutation'][solver]=best_uncert_result['results'][solver]['optimization_result']['activeparams']
        models['theta'][solver] = best_uncert_result['results'][solver]['optimization_result']['scpr']
        models['V_matrix'][solver] = best_uncert_result['results'][solver]['V_matrix']
    iden_opt['log']= True
    iden_opt['init']= None
    iden_opt['var-cov'] = varcovorig


    return rankings, k_optimal_values, rCC_values, J_k_values, best_uncert_result


def parameter_ranking(Z):
    """
    Rank parameters by estimability via iterative orthogonalization.

    At each iteration, select the column whose component
    orthogonal to previously selected columns has maximum norm.

    Parameters
    ----------
    Z : (m, n) array_like
        Sensitivity matrix (m samples × n parameters).

    Returns
    -------
    ranking : list of int
        Parameter indices sorted from most estimable to least estimable.
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
                # project v onto span(Q) and subtract
                proj = Q @ (Q.T @ v)
                r = v - proj
            else:
                r = v
            norms.append(np.linalg.norm(r))

        # pick the column with max residual norm
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
        # zero vectors (if any) are simply skipped

    return ranking


def parameter_selection(n_parameters, ranking_known, system, models, iden_opt, solvera, round, data, result, varcovorig, meth):
    """
    Perform MSE-based selection to determine the optimal number of parameters to estimate.

    Parameters:
    n_parameters (int): The total number of parameters.
    ranking_known (list): The list of parameter indices ranked by estimability.
    system (dict): User provided - The model structure information.
    models (dict): User provided - The settings for the modelling process.
    iden_opt (dict): User provided - The settings for the estimation process.
    solvera (str): The name of the model(s).
    round (int): The current round of the design - conduction and identification procedure.
    data (dict): Prior information for estimability analysis (observations, inputs, etc.).
    result (dict): Baseline estimation results (with all parameters free).
    varcovorig (any): Original variance-covariance configuration to restore for best-case.

    Returns:
    tuple: A tuple containing the optimal number of parameters for estimation in the ranking,
           rCC values (corrected critical ratios), J_k values (WLS objective values),
           and the best uncertainty result dictionary (or None if all parameters are selected)."""
    rCC_values = []
    J_k_values = []
    uncert_results_list = []
    J_theta = result[solvera]['WLS']
    print(f"J_theta : {J_theta}")

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
        results_k = parmest(system_k, models_k, iden_opt_k, data, case='freeze')
        # print(f'theta of parmest is {results_k[solvera]['scpr']}')
        uncert_k = uncert(data, results_k, system_k, models_k, iden_opt_k, case='freeze')
        # print(f'theta of uncert is {results_k[solvera]['optimization_result']['scpr']}')
        uncert_results_list.append(uncert_k)

        # Extract metrics
        J_k = uncert_k['results'][solvera]['WLS']
        n_samples = uncert_k['obs']
        print(f"J_k {k} parameters: {J_k}")

        # Compute corrected criteria
        rC = (J_k - J_theta) / (n_parameters - k)
        rCKub = max(rC - 1, (2 * rC) / (n_parameters - k + 2))
        rCC = ((n_parameters - k) / n_samples) * (rCKub - 1)
        print(f"rCC {k} parameters: {rCC}")

        # Store for plotting
        rCC_values.append(rCC)
        J_k_values.append(J_k)

    # Append rCC=0 for full set and compute x-axis
    rCC_values.append(0)
    x_values = list(range(1, len(rCC_values) + 1))

    # Identify optimal k minimizing rCC
    k_optimal = int(np.argmin(rCC_values) + 1)

    # If all parameters are selected, no additional run
    if k_optimal == n_parameters:
        best_uncert = result
    else:
        # Re-run parmest & uncert for best k with original varcov
        system_best = copy.deepcopy(system)
        models_best = copy.deepcopy(models)
        models_best.pop('normalized_parameters', None)
        iden_opt_best = copy.deepcopy(iden_opt)

        # Apply mask
        selected_mask = [False] * len(ranking_known)
        for i in range(k_optimal):
            selected_mask[ranking_known[i]] = True
        models_best['mutation'][solvera] = selected_mask

        # Restore original varcov settings
        iden_opt_best['var-cov'] = varcovorig
        iden_opt_best['meth'] = meth

        # Execute final estimation and uncertainty
        best_results = parmest(system_best, models_best, iden_opt_best, data, case='freeze')
        best_uncert = uncert(data, best_results, system_best, models_best, iden_opt_best, case='freeze')

    # Plot results
    plot_rCC_vs_k(x_values, rCC_values, round, solvera)

    return k_optimal, rCC_values, J_k_values, best_uncert
