from middoe.krnl_models import *
import numpy as np
from multiprocessing import Pool
import time
import os
from middoe.log_utils import run_mbdoe_md, run_mbdoe_pp, write_excel, read_excel, save_rounds, save_to_jac, \
    data_appender
from middoe.krnl_expera import Expera
from middoe.sc_sensa import Sensa
from middoe.iden_parmest import Parmest
from middoe.iden_uncert import Uncert
from middoe.sc_estima import Estima
from middoe.krnl_simula import Simula
from middoe.iden_valida import validation


def f17(t, y, phi, phit, theta, te):
    """
    Differential equations for gas-solid reaction kinetics.

    Parameters:
    - t: Time (seconds)
    - y: [y1, y2] where
        y1: Fraction of unoccupied surface (dimensionless, 0 ≤ y1 ≤ 1)
        y2: Conversion (dimensionless, 0 ≤ y2 ≤ 1)
    - phi: Time-invariant variables (dictionary containing):
        S0: Initial surface area per unit volume (m²/m³)
        epsilon_0: Initial porosity (dimensionless, 0 ≤ epsilon_0 ≤ 1)
        Z: Ratio of product/reactant molar volumes (dimensionless)
    - phit: Time-variant variables (dictionary containing):
        C: Gas concentration (mol/m³)
        T: Temperature (Kelvin)
    - theta: Parameters [ks0, Ea_ks, Ds0, Ea_Ds, Dp0, Ea_Dp] where:
        ks0: Pre-exponential factor for ks (m⁴/mol/s)
        Ea_ks: Activation energy for ks (J/mol)
        Ds0: Pre-exponential factor for Ds (m²/s)
        Ea_Ds: Activation energy for Ds (J/mol)
        Dp0: Pre-exponential factor for Dp (m²/s)
        Ea_Dp: Activation energy for Dp (J/mol)
    - te: Time points corresponding to profiles of phit variables (seconds)

    Returns:
    - [dy1_dt, dy2_dt]: Derivatives of y1 and y2 where:
        dy1_dt: Rate of change of fraction of unoccupied surface (s⁻¹)
        dy2_dt: Rate of change of conversion (s⁻¹)
    """
    # Unpack current state variables
    y1, y2 = y  # y1: fraction of unoccupied surface, y2: conversion

    # Interpolate time-variant inputs
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    C = np.interp(t, te_array, phit['C'])  # Gas concentration (mol/m³)
    T = np.interp(t, te_array, phit['T'])  # Temperature (Kelvin)

    # Extract time-invariant variables
    S0 = phi['S0']  # Initial surface area per unit volume (m²/m³)
    epsilon_0 = phi['epsilon_0']  # Initial porosity (dimensionless)
    # Z = phi['Z']  # Ratio of product/reactant molar volumes (dimensionless)
    Z = 2.18
    # VBM = phi['VBM']  # Molar Volumee of solid reactant (m³.mol-1)
    VBM= 1.69e-5

    # Extract parameters from theta
    ks0 = theta[0]  # Pre-exponential factor for ks (m⁴/mol/s)
    Ea_ks = theta[1]  # Activation energy for ks (J/mol)
    Ds0 = theta[2]  # Pre-exponential factor for Ds (m²/s)
    Ea_Ds = theta[3]  # Activation energy for Ds (J/mol)
    Dp0 = theta[4]  # Pre-exponential factor for Dp (m²/s)
    Ea_Dp = theta[5]  # Activation energy for Dp (J/mol)

    # Constants
    R = 8.314  # Universal gas constant (J/(mol*K))

    # Arrhenius equations for reaction rates
    ks = ks0 * np.exp(-Ea_ks / (R * T))  # Chemical reaction rate constant (m⁴/mol/s)
    Ds = Ds0 * np.exp(-Ea_Ds / (R * T))  # Surface diffusion coefficient (m²/s)
    Dp = Dp0 * np.exp(-Ea_Dp / (R * T))  # Product layer diffusion coefficient (m²/s)

    # Equilibrium concentration Ce (mol/m³, function of temperature)
    Ce = (1.826e6 / (R * T)) * np.exp(-19680 / T)

    # Geometric model functions
    jS_X = (1 - y2) ** (2 / 3)  # Reaction surface area function (dimensionless)
    gD_X = (1 - y2) ** (2 / 3) / (((1 + Z / (1 - y1) - 1) * y2) ** (2 / 3))  # Geometric function (dimensionless)
    pD_d_X = (3 * (1 - y2) ** (1 / 3)) / ((1 + (Z / (1 - y1) - 1) * y2) ** (1 / 3)) - (1 - y2) ** (1 / 3) * ((1 + ((Z / (1 - y1) - 1) * y2)) ** (1 / 3) - (1 - y2) ** (1 / 3))
    Ks= ks*Z/Ds # ratio of chemical reaction rate to the Surface diffusion coefficient

    # Differential equations
    # Rate of change of fraction of unoccupied surface (y1, s⁻¹)
    dy1_dt = -(ks * Z / Ds) * (C - Ce) * y1 / (1 + Ks * (C - Ce))

    # Rate of change of conversion (y2, s⁻¹)
    beta = (ks* (1 - epsilon_0)) / (Dp * S0 * VBM)  # Coupling term for product layer diffusion (dimensionless)
    dy2_dt = (ks * S0 / (1 - epsilon_0)) * jS_X * (C - Ce) * ((y1 / (1 + ks * (C - Ce))) + ((1 - y1) / (gD_X + beta * (C - Ce) * pD_d_X )))

    # Return the derivatives as a list
    return [dy1_dt, dy2_dt]

def main():

    def thetadic17():
        theta17 = [2.72e-7, 44760, 7.72e-10, 80210, 2.58e-6, 37990]
        # theta17min = [1e-8, 20000, 1e-11, 40000, 1e-7, 20000]
        # theta17max = [1e-5, 100000, 1e-8, 150000, 1e-4, 80000]
        theta17min = [2.72e-7*0.999, 44760*0.999, 7.72e-10*0.999, 80210*0.999, 2.58e-6*0.999, 37990*0.999]
        theta17max = [2.72e-7*1.001, 44760*1.001, 7.72e-10*1.001, 80210*1.001, 2.58e-6*1.001, 37990*1.001]
        theta17maxs = [max_val / theta for max_val, theta in zip(theta17max, theta17)]
        theta17mins = [min_val / theta for min_val, theta in zip(theta17min, theta17)]


        return theta17, theta17maxs, theta17mins



    #reading parameters from embedded models
    theta05, theta05max, theta05min = thetadic05()
    theta06, theta06max, theta06min = thetadic06()
    theta07, theta07max, theta07min = thetadic07()
    theta08, theta08max, theta08min = thetadic08()
    theta09, theta09max, theta09min = thetadic09()
    theta11, theta11max, theta11min = thetadic11()
    theta12, theta12max, theta12min = thetadic12()
    theta13, theta13max, theta13min = thetadic13()
    theta14, theta14max, theta14min = thetadic14()
    theta15, theta15max, theta15min = thetadic15()
    theta16, theta16max, theta16min = thetadic16()
    theta17, theta17max, theta17min = thetadic17()


    model_structure = {
        'tv_iphi': {
            'T': {'swp': 5, 'constraints': 'rel', 'max': 373.15, 'min': 293.15, 'initial_cvp': 'none',
                   'design_cvp': 'CPF', 'offsetl': 0.1, 'offsett': 0.1},
            'P': {'swp': 5, 'constraints': 'dec', 'max': 5, 'min': 1, 'initial_cvp': 'none',
                   'design_cvp': 'CPF', 'offsetl': 0.1, 'offsett': 0.1}
        },
        'tv_ophi': {
            'y1': {'initials': 0.001, 'measured': True, 'sp': 10, 'unc': 0.02, 'offsett': 0.1, 'matching': '1'}
        },
        'ti_iphi': {
            'rho': {'max': 4000, 'min': 2300},
            'cac': {'max': 54.5, 'min': 10},
            'aps': {'max': 1e-4, 'min': 1e-5},
            'mld': {'max': 40000, 'min': 30000},
        },
        'ti_ophi': {
        },
        't_s': [600, 10800]
    }


    design_settings = {
        'eps': 1e-3,
        'optimization_methods': {
            'ppopt_method': 'Local',
            'mdopt_method': 'Local'
        },
        'criteria': {
            'MBDOE_MD_criterion': 'HR',
            'MBDOE_PP_criterion': 'E'
        },
        'iteration_settings': {
            'nd': 10,
            'nd2': 300,
            'maxmd': 100,
            'tolmd': 1e-3,
            'maxpp': 100,
            'tolpp': 1e-3,
        }
    }

    modelling_settings = {
        'ext_func': {'f17': f17},
        'active_solvers': ['f11'],
        'theta_parameters': {
            'f05': theta05,
            'f06': theta06,
            'f07': theta07,
            'f08': theta08,
            'f09': theta09,
            'f11': theta11,
            'f12': theta12,
            'f13': theta13,
            'f14': theta14,
            'f15': theta15,
            'f16': theta16,
            'f17': theta17
        },
        'bound_max': {
            'f05': theta05max,
            'f06': theta06max,
            'f07': theta07max,
            'f08': theta08max,
            'f09': theta09max,
            'f11': theta11max,
            'f12': theta12max,
            'f13': theta13max,
            'f14': theta14max,
            'f15': theta15max,
            'f16': theta16max,
            'f17': theta17max
        },
        'bound_min': {
            'f05': theta05min,
            'f06': theta06min,
            'f07': theta07min,
            'f08': theta08min,
            'f09': theta09min,
            'f11': theta11min,
            'f12': theta12min,
            'f13': theta13min,
            'f14': theta14min,
            'f15': theta15min,
            'f16': theta16min,
            'f17': theta17min
        },
    }



    # scms
    GSA_settings = {
        'perform_sensitivity': False,
        'phi_nom': [3191, 44.93, 5.5e-5, 36000],
        'phit_nom': [293.15, 1],
        'var_damping': False,
        'par_damping': True,
        'parallel': True,
        'power': 0.7,
        'var_damping_factor': 1.1,
        'par_damping_factor': 1.1,
        'sampling': 2**10,
        'var_sensitivity': False,
        'par_sensitivity': True
    }


    # scms
    simulator_settings = {
        'insilico_model': 'f11',
        'smoothness': 300,
        'classic-des': {
            '1': {'T': 293.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000},
            '2': {'T': 313.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000},
            '3': {'T': 333.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000},
            '4': {'T': 353.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000}
        }
    }


    estimation_settings = {
        'method': 'Local',  # global, local
        'initialization': 'random',   # use 'random' to have random starting point and use None to start from theta_parameters
        'eps': 1e-3,  #usually 1e-3, or None to perform a mesh independent test
        'objf': 'JWLS',  # LS: least squares, MLE: maximum likelihood, Chi: chi-square, JWLS: weighted least squares
        'con_plot': False,
        'fit_plot': True,
        'logging': True
    }

    logic_settings = {
        'max_MD_runs': 1,
        'max_PP_runs': 1,
        'md_conf_tresh': 85,
        'md_rej_tresh': 15,
        'pp_conf_threshold': 1,
        'parallel_sessions': 10
    }

    framework_settings = {
        'path': 'C:\\datasim',
        'case': 1
    }


    def run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                      simulator_settings, estimation_settings, GSA_settings):
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
            sobol_results = {}
            sobol_results['sobol_analysis_results'], sobol_results['sobol_problem'] = Sensa(GSA_settings,
                                                                                            modelling_settings,
                                                                                            model_structure,
                                                                                            framework_settings)

        data_storage = run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings,
                                         estimation_settings, round_data, design_decisions, data_storage)

        if not winner_solver_found:
            winner_solver_found, winner_solver = run_md_rounds(framework_settings, model_structure, modelling_settings,
                                                               simulator_settings, estimation_settings, logic_settings,
                                                               design_settings, logic_settings['parallel_sessions'],
                                                               round_data, design_decisions, data_storage)
        terminate_loop = False
        if winner_solver_found:
            terminate_loop = run_pp_rounds(framework_settings, model_structure, modelling_settings, simulator_settings,
                                           estimation_settings, logic_settings, design_settings,
                                           logic_settings['parallel_sessions'], winner_solver, round_data,
                                           design_decisions, data_storage)

            if terminate_loop:
                print("Loop terminated successfully.")
        else:
            print("No winner solver found, exiting without termination loop.")

        print(f"Round keys: {round_data.keys()}")
        print('---------------------------------------------')
        # Call cross-validation at the end
        R2_prd, R2_val, parameters = validation(data_storage, model_structure, modelling_settings, estimation_settings,
                                                Simula, round_data, framework_settings)

        base_path = framework_settings['path']
        modelling_folder = str(framework_settings['case'])  # No leading backslash here

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

    def run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings,
                          estimation_settings, round_data, design_decisions, data_storage):
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
            excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j)
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

            if j in [1, 2]:
                ranking, k_optimal_value, rCC_values, J_k_values = Estima(
                    resultun,
                    model_structure,
                    modelling_settings,
                    estimation_settings,
                    j,
                    framework_settings,
                    data_storage,
                    Simula
                )
            else:
                ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            # ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'classic design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)
        return data_storage

    def run_md_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings,
                      logic_settings, design_settings, num_parallel_runs,
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
                                                [(design_settings, model_structure, modelling_settings, core_num,
                                                  framework_settings, round) for core_num in range(num_parallel_runs)])
                best_design_decisions, best_sum_squared_differences, best_swps = max(results_list, key=lambda x: x[1])
                design_decisions.update(best_design_decisions)
            elif design_settings['optimization_methods']['mdopt_method'] == 'Global':
                core_num = 1
                results_list = run_mbdoe_md(design_settings, model_structure, modelling_settings, core_num,
                                            framework_settings, round)
                best_design_decisions, best_sum_squared_differences, best_swps = results_list
                design_decisions.update(best_design_decisions)

            expr = j
            # excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings, simulator_settings, design_decisions, j, swps=best_swps)
            # write_excel(df_combined, expr, excel_path)
            # # write_to_excel(df_combined, simulator_settings['expr'], excel_path)
            # # data = read_excel_data(excel_path)
            # data = read_excel(excel_path)
            # # data_storage = data_appender(df_combined, expr, data_storage)

            excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j, swps=best_swps)
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

            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'MBDOE_MD design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)

            if winner_solver_found:
                break

        return winner_solver_found, winner_solver

    def run_pp_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings,
                      logic_settings, design_settings, num_parallel_runs,
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
                                                [(design_settings, model_structure, modelling_settings, core_num,
                                                  framework_settings, round) for core_num in range(num_parallel_runs)])
                best_design_decisions, best_pp_obj, best_swps = max(results_list, key=lambda x: x[1])
                design_decisions.update(best_design_decisions)
            elif design_settings['optimization_methods']['ppopt_method'] == 'Global':
                core_num = 1
                results_list = run_mbdoe_pp(design_settings, model_structure, modelling_settings, core_num,
                                            framework_settings, round)
                best_design_decisions, best_pp_obj, best_swps = results_list
                design_decisions.update(best_design_decisions)

            expr = j
            excel_path, df_combined = Expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j, swps=best_swps)
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
            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'MBDOE_PP design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)

            for solver, solver_results in resultun.items():
                if all(t_value > trv[solver] for t_value in solver_results['t_values']):
                    terminate_loop = True
                    break

            if terminate_loop:
                break

        return terminate_loop


    run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                  simulator_settings, estimation_settings, GSA_settings)




if __name__ == '__main__':
    main()

