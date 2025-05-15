import numpy as np
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
    iden_opt['log']= False
    print (f"Estimability analysis for round {round} is running")
    for solver, res in result.items():
        Z = res['LSA']
        n_parameters = Z.shape[1]

        ranking_known = parameter_ranking(Z)
        print(f"Parameter ranking from most estimable to least estimable for {solver} in round {round}: {ranking_known}")

        k_optimal, rCC, J_k = parameter_selection(
            n_parameters, ranking_known, system, models,
            iden_opt, solver, round, data
        )
        print(f"Optimal number of parameters to estimate for {solver}: {k_optimal}")

        rankings[solver] = ranking_known
        k_optimal_values[solver] = k_optimal
        rCC_values[solver] = rCC
        J_k_values[solver] = J_k
    iden_opt['log']= True

    return rankings, k_optimal_values, rCC_values, J_k_values


def parameter_ranking(Z):
    """
    Perform orthogonalization on the given matrix Z to rank parameters based on their estimability.

    Parameters:
    Z (numpy.ndarray): The matrix containing the sensitivity information.

    Returns:
    list: A list of parameter indices ranked from most estimable to least estimable.
    """
    n_samples, n_parameters = Z.shape
    ranking = []
    remaining_columns = list(range(n_parameters))
    magnitudes = [la.norm(Z[:, j]) for j in remaining_columns]
    max_magnitude_index = np.argmax(magnitudes)
    most_estimable = remaining_columns.pop(max_magnitude_index)
    ranking.append(most_estimable)

    for k in range(n_parameters):
        Xk = Z[:, ranking]
        Z_hat = Xk @ np.linalg.pinv(Xk.T @ Xk) @ Xk.T @ Z
        Rk = Z - Z_hat
        magnitudes = [la.norm(Rk[:, j]) for j in remaining_columns]
        max_magnitude_index = np.argmax(magnitudes)
        most_estimable = remaining_columns.pop(max_magnitude_index)
        ranking.append(most_estimable)
        if len(ranking) >= n_parameters:
            return ranking

def parameter_selection(n_parameters, ranking_known, system, models, iden_opt, solvera, round, data):
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
    data (dict): prior information for estimability analysis (observations, inputs, etc.).
    run_solver (function): The function to run the model (simulator-bridger).

    Returns:
    tuple: A tuple containing the optimal number of parameters for estimation in the ranking, rCC values (corrected critical ratios), and J_k values (objectives of weighted least square method based optimization).
    """
    rCC_values = []
    J_k_values = []
    original_mutation = models['mutation'][solvera].copy()
    models['mutation'][solvera] = [True] * len(models['theta'][solvera])

    results_all_params = parmest(
        system,
        models,
        iden_opt,
        data,case= 'freeze')

    uncert_results_all = uncert(data, results_all_params, system, models, iden_opt)
    result = uncert_results_all['results']
    n_samples = uncert_results_all['obs']

    vmax=models['V_matrix'][solvera]
    J_theta = result[solvera]['WLS']

    for k in range(1, n_parameters):
        selected_mask = [False] * len(ranking_known)
        for i in range(k):
            selected_mask[ranking_known[i]] = True

        models['mutation'][solvera] = selected_mask

        results_k_params = parmest(
            system,
            models,
            iden_opt,
            data, case= 'freeze')
        # print(f"Results for {k} parameters: {results_k_params[solvera]['optimization_result'].fun}")

        uncert_results_k = uncert(data, results_k_params, system, models, iden_opt)
        resultk = uncert_results_k['results']
        n_samples = uncert_results_k['obs']

        J_k = resultk[solvera]['WLS']

        rC = (J_k - J_theta) / ((n_parameters - k))
        rCKub = max(rC-1, (2 * rC) / (n_parameters - k + 2))
        rCC = ((n_parameters - k) / n_samples) * (rCKub-1)

        rCC_values.append(rCC)
        J_k_values.append(J_k)
    rCC_values.append(0)
    x_values = list(range(1, len(rCC_values) + 1))
    k_optimal = np.argmin(rCC_values) + 1

    models['mutation'][solvera] = original_mutation
    plot_rCC_vs_k(x_values, rCC_values, round, solvera)

    selected_mask = [True] * len(ranking_known)
    models['mutation'][solvera] = selected_mask
    models['V_matrix'][solvera] = vmax

    return k_optimal, rCC_values, J_k_values

