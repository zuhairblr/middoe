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
    print (f"Estimability analysis for round {round} is running")
    for solver, res in result.items():
        Z = res['LSA']
        n_parameters = Z.shape[1]

        ranking_known = parameter_ranking(Z)
        print(f"Parameter ranking from most estimable to least estimable for {solver} in round {round}: {ranking_known}")

        k_optimal, rCC, J_k, best_uncert_result = parameter_selection(
            n_parameters, ranking_known, system, models,
            iden_opt, solver, round, data, result
        )
        print(f"Optimal number of parameters to estimate for {solver}: {k_optimal}")

        rankings[solver] = ranking_known
        k_optimal_values[solver] = k_optimal
        rCC_values[solver] = rCC
        J_k_values[solver] = J_k
    iden_opt['log']= True
    iden_opt['init']= None
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



def parameter_selection(n_parameters, ranking_known, system, models, iden_opt, solvera, round, data, result):
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

    Returns:
    tuple: A tuple containing the optimal number of parameters for estimation in the ranking,
           rCC values (corrected critical ratios), and J_k values (WLS objective values).
    """
    rCC_values = []
    J_k_values = []
    uncert_results_list = []

    J_theta = result[solvera]['WLS']
    print(f"J_theta : {J_theta}")

    for k in range(1, n_parameters):
        # Deepcopy at each iteration to reset clean state
        system_k = copy.deepcopy(system)
        models_k = copy.deepcopy(models)
        models_k.pop('normalized_parameters', None)
        iden_opt_k = copy.deepcopy(iden_opt)

        # Construct mutation mask for top-k ranked parameters
        selected_mask = [False] * len(ranking_known)
        for i in range(k):
            selected_mask[ranking_known[i]] = True

        models_k['mutation'][solvera] = selected_mask

        # Run estimation and uncertainty analysis with k parameters
        results_k_params = parmest(system_k, models_k, iden_opt_k, data, case='freeze')
        uncert_results_k = uncert(data, results_k_params, system_k, models_k, iden_opt_k, case='freeze')
        uncert_results_list.append(uncert_results_k)

        resultk = uncert_results_k['results']
        n_samples = uncert_results_k['obs']

        J_k = resultk[solvera]['WLS']
        print(f"J_k {k} parameters: {J_k}")

        rC = (J_k - J_theta) / (n_parameters - k)
        print(f"rC {k} parameters: {rC}")
        rCKub = max(rC - 1, (2 * rC) / (n_parameters - k + 2))
        print(f"rCKub parameters: {rCKub}")
        rCC = ((n_parameters - k) / n_samples) * (rCKub - 1)
        print(f"rCC parameters: {rCC}")

        rCC_values.append(rCC)
        J_k_values.append(J_k)

    # Add rCC=0 for the full parameter set
    rCC_values.append(0)
    x_values = list(range(1, len(rCC_values) + 1))
    k_optimal = np.argmin(rCC_values) + 1
    # Safely select best uncertainty result
    if k_optimal <= len(uncert_results_list):
        best_uncert_result = uncert_results_list[k_optimal - 1]
    else:
        best_uncert_result = {'results': result, 'obs': None}  # fallback

    # Plot with consistent config (only for display)
    plot_rCC_vs_k(x_values, rCC_values, round, solvera)

    return k_optimal, rCC_values, J_k_values, best_uncert_result
