from multiprocessing import Pool
import time
import os
from middoe.log_utils import run_mbdoe_md, run_mbdoe_pp, write_excel, read_excel, save_rounds, save_to_jac, data_appender
from middoe.krnl_expera import Expera
from middoe.sc_sensa import Sensa
from middoe.iden_parmest import Parmest
from middoe.iden_uncert import Uncert
from middoe.sc_estima import Estima
from middoe.krnl_simula import Simula
from middoe.iden_valida import validation

def run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings, simulator_settings, estimation_settings, GSA_settings):
    """
    Run the entire framework for in-silico experiments, including initial rounds, MD rounds, and PP rounds.

    Parameters:
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    logic_settings (dict): User provided - Logic settings for the MBDOE process, including thresholds and maximum runs.
    model_structure (dict): User provided - Structure of the model, including variables and their properties.
    design_settings (dict): User provided - Design settings for the experiment, including mutation and crossover rates.
    modelling_settings (dict): User provided - Settings related to the modelling process, including theta parameters.
    simulator_settings (dict): User provided - Settings for the simulator, including standard deviation and model name.
    estimation_settings (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
    GSA_settings (dict): User provided - Settings for the Global Sensitivity Analysis (GSA), including whether to perform sensitivity analysis.

    Returns:
    None
    """
    start_time = time.time()
    data_storage = {}
    round_data = {}
    design_decisions = {}
    winner_solver = None
    winner_solver_found = False

    if len(modelling_settings['active_solvers']) == 1:
        winner_solver_found = True
        winner_solver = modelling_settings['active_solvers'][0]
        print("There are no rival models for:", winner_solver)

    if GSA_settings['perform_sensitivity']:
        sobol_results={}
        sobol_results['sobol_analysis_results'], sobol_results['sobol_problem']= Sensa(GSA_settings, modelling_settings, model_structure, framework_settings)

    data_storage = run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, round_data, design_decisions,data_storage)

    if not winner_solver_found:

        winner_solver_found, winner_solver = run_md_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, logic_settings, design_settings,  logic_settings['parallel_sessions'], round_data, design_decisions, data_storage)
    terminate_loop = False
    if winner_solver_found:
        terminate_loop = run_pp_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, logic_settings, design_settings,
                                       logic_settings['parallel_sessions'], winner_solver, round_data,
                                       design_decisions, data_storage)

        if terminate_loop:
            print("Loop terminated successfully.")
    else:
        print("No winner solver found, exiting without termination loop.")

    print(f"Round keys: {round_data.keys()}")
    print('---------------------------------------------')
    # Call cross-validation at the end
    R2_prd, R2_val, parameters = validation(data_storage, model_structure, modelling_settings, estimation_settings, Simula, round_data, framework_settings)


    base_path = framework_settings['path']
    modelling_folder = str(framework_settings['case'])   # No leading backslash here

    # Join the base path and modelling folder
    path = os.path.join(base_path, modelling_folder)

    # Ensure the 'modelling' directory exists
    os.makedirs(path, exist_ok=True)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime of framework: {runtime} seconds")

    folder_path = path
    file_path = os.path.join(folder_path, 'results.jac')
    file_path2 = os.path.join(folder_path, 'results2.jac')

    os.makedirs(folder_path, exist_ok=True)
    save_to_jac(round_data, file_path)
    if GSA_settings['perform_sensitivity']:
        save_to_jac(sobol_results, file_path2)



def run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, round_data, design_decisions, data_storage):
    """
    Run the initial round of in-silico experiments, joint with identification steps.

    Parameters:
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    model_structure (dict): User provided - Structure of the model, including variables and their properties.
    modelling_settings (dict): User provided - Settings related to the modelling process, including theta parameters.
    simulator_settings (dict): User provided - Settings for the simulator, including standard deviation and model name.
    estimation_settings (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
    round_data (dict): collection of saved data from each round of the experiment.
    design_decisions (dict): Design decisions for the experiment, to be performed, insilico or in bench.
    data_storage (dict): Storage for the experimental data observations

    Returns:
    None
    """
    for i in range(len(simulator_settings['classic-des'])):
        j = i + 1
        expr = i + 1
        case = 'classic'
        excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j)
        write_excel(df_combined, expr, excel_path)
        data = read_excel(excel_path)
        data_storage = data_appender(df_combined, expr, data)

        resultpr = Parmest(
            model_structure,
            modelling_settings,
            estimation_settings,
            data_storage,
            Simula
        )

        resultun, theta_parameters, solver_parameters, scaled_params, obs = Uncert(
            data_storage,
            resultpr,
            model_structure,
            modelling_settings,
            estimation_settings,
            Simula
        )

        # if j in [1, 2]:
        #     ranking, k_optimal_value, rCC_values, J_k_values = Estima(
        #         resultun,
        #         model_structure,
        #         modelling_settings,
        #         estimation_settings,
        #         j,
        #         framework_settings,
        #         data_storage,
        #         Simula
        #     )
        # else:
        #     ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
        ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
        trv =save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters, 'classic design', round_data, modelling_settings, scaled_params, estimation_settings, solver_parameters, framework_settings, obs, case, data_storage, model_structure)
    return data_storage

def run_md_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, logic_settings, design_settings, num_parallel_runs,
                  round_data, design_decisions, data_storage):
    """
    Run multiple rounds of Model-Based Design of Experiments (MBDOE) using the MD approach.

    Parameters:
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    model_structure (dict): User provided - Structure of the model, including variables and their properties.
    modelling_settings (dict): User provided - Settings related to the modelling process, including theta parameters.
    simulator_settings (dict): User provided - Settings for the simulator, including standard deviation and model name.
    estimation_settings (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
    logic_settings (dict): User provided - Logic settings for the identification process, including thresholds and maximum runs.
    design_settings (dict): User provided - Design settings for the experiment, including mutation and crossover rates.
    num_parallel_runs (int): Number of parallel runs to execute MBDOE-MD.
    round_data (dict): collection of saved data from each round of the experiment.
    design_decisions (dict): Design decisions for the experiment, to be performed, insilico or in bench.
    data_storage (dict): Storage for the experimental data observations

    Returns:
    tuple: A tuple containing:
        - winner_solver_found (bool): Indicates if a winning solver was found.
        - winner_solver (str or None): The name of the winning solver, if found.
    """
    winner_solver_found = False
    winner_solver = None
    start_round = len(round_data) + 1
    case = 'doe'

    for i in range(logic_settings['max_MD_runs']):
        j = start_round + i
        round = j

        if design_settings['optimization_methods']['mdopt_method'] == 'Local':
            with Pool(num_parallel_runs) as pool:
                results_list = pool.starmap(run_mbdoe_md,
                                            [(design_settings, model_structure, modelling_settings, core_num, framework_settings, round) for core_num in range(num_parallel_runs)])
            best_design_decisions, best_sum_squared_differences, best_swps = max(results_list, key=lambda x: x[1])
            design_decisions.update(best_design_decisions)
        elif design_settings['optimization_methods']['mdopt_method'] == 'Global':
            core_num=1
            results_list = run_mbdoe_md(design_settings, model_structure, modelling_settings, core_num, framework_settings, round)
            best_design_decisions, best_sum_squared_differences, best_swps = results_list
            design_decisions.update(best_design_decisions)



        expr = j
        # excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j, swps=best_swps)
        # write_excel(df_combined, expr, excel_path)
        # # write_to_excel(df_combined, simulator_settings['expr'], excel_path)
        # # data = read_excel_data(excel_path)
        # data = read_excel(excel_path)
        # # data_storage = data_appender(df_combined, expr, data_storage)

        excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j, swps=best_swps)
        write_excel(df_combined, expr, excel_path)
        data = read_excel(excel_path)
        data_storage = data_appender(df_combined, expr, data_storage)

        resultpr = Parmest(
            model_structure,
            modelling_settings,
            estimation_settings,
            data,
            Simula
        )

        resultun, theta_parameters, solver_parameters, scaled_params, obs = Uncert(
            data,
            resultpr,
            model_structure,
            modelling_settings,
            estimation_settings,
            Simula
        )

        ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None

        p_r_threshold = logic_settings['md_rej_tresh']
        p_a_threshold = logic_settings['md_conf_tresh']
        for solver, solver_results in resultun.items():
            if solver_results['P'] < p_r_threshold:
                modelling_settings['active_solvers'].remove(solver)
            if solver_results['P'] > p_a_threshold:
                winner_solver = solver
                winner_solver_found = True
                break

        # Check if this is the last iteration
        if i == logic_settings['max_MD_runs'] - 1:
            # Find the model with the highest 'P' value
            highest_p_solver = max(resultun.items(), key=lambda x: x[1]['P'])[0]
            highest_p_value = resultun[highest_p_solver]['P']
            winner_solver_found = True
            winner_solver = highest_p_solver
            print(
                f"Model discrimination ended after {i + 1} iterations. Model '{winner_solver}' selected as winner based on P value = {highest_p_value}.")

        trv =save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters, 'MBDOE_MD design', round_data, modelling_settings, scaled_params, estimation_settings, solver_parameters, framework_settings, obs, case, data_storage, model_structure)


        if winner_solver_found:
            break



    return winner_solver_found, winner_solver

def run_pp_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, logic_settings, design_settings, num_parallel_runs,
                  winner_solver, round_data, design_decisions, data_storage):
    """
    Run multiple rounds of Model-Based Design of Experiments (MBDOE) using the PP approach.

    Parameters:
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
    model_structure (dict): User provided - Structure of the model, including variables and their properties.
    modelling_settings (dict): User provided - Settings related to the modelling process, including theta parameters.
    simulator_settings (dict): User provided - Settings for the simulator, including standard deviation and model name.
    estimation_settings (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
    logic_settings (dict): User provided - Logic settings for the identification process, including thresholds and maximum runs.
    design_settings (dict): User provided - Design settings for the experiment, including mutation and crossover rates.
    num_parallel_runs (int): Number of parallel runs to execute MBDOE-PP.
    winner_solver (str): The name of the winning solver from the MD rounds.
    round_data (dict): collection of saved data from each round of the experiment.
    design_decisions (dict): Design decisions for the experiment, to be performed, insilico or in bench.
    data_storage (dict): Storage for the experimental data observations

    Returns:
    bool: Indicates if the loop was terminated based on the confidence threshold.
    """
    terminate_loop = False
    start_round = len(round_data) + 1
    case = 'doe'

    for i in range(logic_settings['max_PP_runs']):
        j = start_round + i
        round = j
        modelling_settings['active_solvers'] = [winner_solver]

        if design_settings['optimization_methods']['ppopt_method'] == 'Local':
            with Pool(num_parallel_runs) as pool:
                results_list = pool.starmap(run_mbdoe_pp,
                                            [(design_settings, model_structure, modelling_settings, core_num, framework_settings, round) for core_num in range(num_parallel_runs)])
            best_design_decisions, best_pp_obj, best_swps = max(results_list, key=lambda x: x[1])
            design_decisions.update(best_design_decisions)
        elif design_settings['optimization_methods']['ppopt_method'] == 'Global':
            core_num=1
            results_list = run_mbdoe_pp(design_settings, model_structure, modelling_settings, core_num, framework_settings, round)
            best_design_decisions, best_pp_obj, best_swps = results_list
            design_decisions.update(best_design_decisions)

        expr = j
        excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j, swps=best_swps)
        write_excel(df_combined, expr, excel_path)
        data = read_excel(excel_path)
        data_storage = data_appender(df_combined, expr, data_storage)

        resultpr = Parmest(
            model_structure,
            modelling_settings,
            estimation_settings,
            data,
            Simula
        )

        resultun, theta_parameters, solver_parameters, scaled_params, obs = Uncert(
            data,
            resultpr,
            model_structure,
            modelling_settings,
            estimation_settings,
            Simula
        )

        ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
        trv= save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters, 'MBDOE_PP design', round_data, modelling_settings, scaled_params, estimation_settings, solver_parameters, framework_settings, obs, case, data_storage, model_structure)

        for solver, solver_results in resultun.items():
            if all(t_value > trv[solver] for t_value in solver_results['t_values']):
                terminate_loop = True
                break

        if terminate_loop:
            break

    return terminate_loop

