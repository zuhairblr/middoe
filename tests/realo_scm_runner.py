from middoe.krnl_models import *
import numpy as np
from multiprocessing import Pool
import time
import os
from middoe.log_utils import run_mbdoe_md, run_mbdoe_pp, write_excel, read_excel, save_rounds, save_to_jac, \
    data_appender
from middoe.krnl_expera import expera
from middoe.sc_sensa import sensa
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
from middoe.sc_estima import estima
from middoe.krnl_simula import simula
from middoe.iden_valida import validation
import pandas as pd

def main():
    theta08, theta08max, theta08min = thetadic08()
    theta09, theta09max, theta09min = thetadic09()
    theta11, theta11max, theta11min = thetadic11()
    theta12, theta12max, theta12min = thetadic12()
    theta13, theta13max, theta13min = thetadic13()
    theta14, theta14max, theta14min = thetadic14()
    theta15, theta15max, theta15min = thetadic15()
    theta16, theta16max, theta16min = thetadic16()

    model_structure = {
        'tv_iphi': {
            'T': {'swp': 5, 'constraints': 'rel', 'max': 373.15, 'min': 293.15, 'initial_cvp': 'none',
                   'design_cvp': 'CPF', 'offsetl': 0.1, 'offsett': 0.1},
            'P': {'swp': 5, 'constraints': 'dec', 'max': 5, 'min': 1, 'initial_cvp': 'none',
                   'design_cvp': 'CPF', 'offsetl': 0.1, 'offsett': 0.1}
        },
        'tv_ophi': {
            'y1': {'initials': 0.001, 'measured': True, 'sp': 10, 'unc': None, 'offsett': 0.1, 'matching': '1'}
        },
        'ti_iphi': {
            'rho': {'max': 4000, 'min': 2000},
            'cac': {'max': 54.5, 'min': 10},
            'aps': {'max': 1e-4, 'min': 2e-6},
            'mld': {'max': 43000, 'min': 30000},
        },
        'ti_ophi': {
        },
        't_s': [600, 3600]
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
        'ext_func': {},
        'active_solvers': ['f08','f09','f11','f12','f13','f14','f15','f16'],
        'theta_parameters': {
            'f08': theta08,
            'f09': theta09,
            'f11': theta11,
            'f12': theta12,
            'f13': theta13,
            'f14': theta14,
            'f15': theta15,
            'f16': theta16
        },
        'bound_max': {
            'f08': theta08max,
            'f09': theta09max,
            'f11': theta11max,
            'f12': theta12max,
            'f13': theta13max,
            'f14': theta14max,
            'f15': theta15max,
            'f16': theta16max
        },
        'bound_min': {
            'f08': theta08min,
            'f09': theta09min,
            'f11': theta11min,
            'f12': theta12min,
            'f13': theta13min,
            'f14': theta14min,
            'f15': theta15min,
            'f16': theta16min
        },
    }

    # scms
    GSA_settings = {
        'perform_sensitivity': False,
        'phi_nom': [2027.7, 42.2, 8.48e-6, 27090],
        'phit_nom': [325.65, 1],
        'var_damping': False,
        'par_damping': False,
        'parallel': True,
        'power': 0.7,
        'var_damping_factor': 1.1,
        'par_damping_factor': 1.1,
        'sampling': 2**12,
        'var_sensitivity': False,
        'par_sensitivity': True
    }


    estimation_settings = {
        'method': 'Global',  # Global, Local
        'initialization': 'random',  # use 'random' to have random starting point and use None to start from theta_parameters
        'eps': 1e-3,  #usually 1e-3, or None to perform a mesh independent test
        'objf': 'JWLS',  # LS: least squares, MLE: maximum likelihood, Chi: chi-square, JWLS: weighted least squares
        'con_plot': False,
        'fit_plot': True,
        'logging': True
    }

    logic_settings = {
        'max_MD_runs': 8,
        'max_PP_runs': 8,
        'md_conf_tresh': 85,
        'md_rej_tresh': 15,
        'pp_conf_threshold': 1,
        'parallel_sessions': 1
    }

    simulator_settings={}

    framework_settings = {
        'path': 'C:\\datasim',
        'case': 4
    }

    def run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings, simulator_settings, estimation_settings, GSA_settings):
        """
        Run the entire framework for in-silico experiments, including initial rounds, MD rounds, and PP rounds.

        Parameters:
        framework_settings (dict): User provided - Settings related to the framework, including paths and case information.
        logic_settings (dict): User provided - Logic settings for the MBDOE process, including thresholds and maximum runs.
        system (dict): User provided - Structure of the model, including variables and their properties.
        des_opt (dict): User provided - Design settings for the experiment, including mutation and crossover rates.
        models (dict): User provided - Settings related to the modelling process, including theta parameters.
        insilicos (dict): User provided - Settings for the simulator, including standard deviation and model name.
        iden_opt (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
        gsa (dict): User provided - Settings for the Global Sensitivity Analysis (GSA), including whether to perform sensitivity analysis.

        Returns:
        None
        """
        start_time = time.time()
        data_storage = {}
        round_data = {}
        design_decisions = {}
        # winner_solver = None
        # winner_solver_found = False

        if len(modelling_settings['active_solvers']) == 1:
            # winner_solver_found = True
            winner_solver = modelling_settings['active_solvers'][0]
            print("There are no rival models for:", winner_solver)

        if GSA_settings['perform_sensitivity']:
            sobol_results={}
            sobol_results['sobol_analysis_results'], sobol_results['sobol_problem']= sensa(GSA_settings, modelling_settings, model_structure, framework_settings)

        data_storage = run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings, round_data, design_decisions,data_storage)

        # if not winner_solver_found:
        #
        #     winner_solver_found, winner_solver = run_md_rounds(framework_settings, system, models, insilicos, iden_opt, logic_settings, des_opt,  logic_settings['parallel_sessions'], round_data, design_decisions, data_storage)
        # terminate_loop = False
        # if winner_solver_found:
        #     terminate_loop = run_pp_rounds(framework_settings, system, models, insilicos, iden_opt, logic_settings, des_opt,
        #                                    logic_settings['parallel_sessions'], winner_solver, round_data,
        #                                    design_decisions, data_storage)
        #
        #     if terminate_loop:
        #         print("Loop terminated successfully.")
        # else:
        #     print("No winner solver found, exiting without termination loop.")
        #
        # print(f"Round keys: {round_data.keys()}")
        print('---------------------------------------------')
        # Call cross-validation at the end
        R2_prd, R2_val, parameters = validation(data_storage, model_structure, modelling_settings, estimation_settings, simula, round_data, framework_settings)


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
        system (dict): User provided - Structure of the model, including variables and their properties.
        models (dict): User provided - Settings related to the modelling process, including theta parameters.
        insilicos (dict): User provided - Settings for the simulator, including standard deviation and model name.
        iden_opt (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
        round_data (dict): collection of saved data from each round of the experiment.
        design_decisions (dict): Design decisions for the experiment, to be performed, insilico or in bench.
        data_storage (dict): Storage for the experimental data observations

        Returns:
        None
        """
        base_path = framework_settings['path']
        modelling_folder = str(framework_settings['case'])  # No leading backslash here
        path = os.path.join(base_path, modelling_folder)
        os.makedirs(path, exist_ok=True)
        excel_path = os.path.join(path, f"benchscale_experiments.xlsx")

        # Load the Excel file and get the sheet names
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names  # Get the names of all sheets

        for i, sheet_name in enumerate(sheet_names):
            j = i + 1
            case = 'classic'
            # Read the current sheet and append it to the combined data
            current_sheet_data = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=0)  # Assuming no need to skip the title row here
            # Store the sheet data in the data_storage dictionary using the sheet name as the key
            data_storage[sheet_name] = current_sheet_data

            resultpr = parmest(
                model_structure,
                modelling_settings,
                estimation_settings,
                data_storage,
                simula
            )

            resultun, theta_parameters, solver_parameters, scaled_params, obs = uncert(
                data_storage,
                resultpr,
                model_structure,
                modelling_settings,
                estimation_settings,
                simula
            )

            # if round in [3]:
            #     ranking, k_optimal_value, rCC_values, J_k_values = Estima(
            #         resultun,
            #         system,
            #         models,
            #         iden_opt,
            #         round,
            #         framework_settings,
            #         data_storage,
            #         Simula
            #     )
            # else:
            #     ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            trv =save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters, 'classic design', round_data, modelling_settings, scaled_params, estimation_settings, solver_parameters, framework_settings, obs, case, data_storage, model_structure)
        return data_storage

    run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                  simulator_settings, estimation_settings, GSA_settings)

if __name__ == '__main__':
    main()

