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


def f17(t, y, phi, phit, theta, te):
    """
    Differential equations for gas-solid reaction kinetics.

    Parameters:
    - t: Time (seconds)
    - y: [y1, y2] where
        y1: Fraction of unoccupied surface (dimensionless, 0 ≤ y1 ≤ 1)
        y2: Conversion (dimensionless, 0 ≤ y2 ≤ 1)
    - tii: Time-invariant variables (dictionary containing):
        S0: Initial surface area per unit volume (m²/m³)
        epsilon_0: Initial porosity (dimensionless, 0 ≤ epsilon_0 ≤ 1)
        Z: Ratio of product/reactant molar volumes (dimensionless)
    - tvi: Time-variant variables (dictionary containing):
        C: Gas concentration (mol/m³)
        T: Temperature (Kelvin)
    - theta: Parameters [ks0, Ea_ks, Ds0, Ea_Ds, Dp0, Ea_Dp] where:
        ks0: Pre-exponential factor for ks (m⁴/mol/s)
        Ea_ks: Activation energy for ks (J/mol)
        Ds0: Pre-exponential factor for Ds (m²/s)
        Ea_Ds: Activation energy for Ds (J/mol)
        Dp0: Pre-exponential factor for Dp (m²/s)
        Ea_Dp: Activation energy for Dp (J/mol)
    - te: Time points corresponding to profiles of tvi variables (seconds)

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
    # Z = tii['Z']  # Ratio of product/reactant molar volumes (dimensionless)
    Z = 2.18
    # VBM = tii['VBM']  # Molar Volumee of solid reactant (m³.mol-1)
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
        'tv_iphi': {  # Time-variant input variables (model input: tvi), each key is a symbol nad key in tvi as well
            'T': {  # Temperature (K)
                'swp': 5,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. Controls the number of discrete steps in the profile.
                'constraints': 'dec',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 373.15,  # Maximum allowable signal level, design space upper bound
                'min': 293.15,  # Minimum allowable signal level, design space lower bound
                'initial_cvp': 'none',  # Initial CVP method (none - no predefined profile)
                'design_cvp': 'CPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offsetl': 10,  # minimum allowed perturbation of signal (ratio)
                'offsett': 600  # minimum allowed perturbation of time (ratio)
            },
            'P': {  # Pressure (bar)
                'swp': 5,
                'constraints': 'rel',
                'max': 5,
                'min': 1,
                'initial_cvp': 'none',
                'design_cvp': 'CPF',
                'offsetl': 0.5,
                'offsett': 600
            }
        },
        'tv_ophi': {  # Time-variant output variables (responses, measured or unmeasured)
            'y1': {  # response variable, here carbonation efficiency
                'initials': 0.001,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a design decision (time-invariant input variable)
                'measured': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 10,  # the amound of samples per each round (run)
                'unc': 0.02,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offsett': 600,  # minimum allowed perturbation of sampling times (ratio)
                'matching': '1'  # Matching criterion for model prediction and data alignment
            }
        },
        'ti_iphi': {  # Time-invariant input variables (tii)
            'rho': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 4000,  # Maximum allowable signal level, design space upper bound
                'min': 2300  # Minimum allowable signal level, design space upper bound
            },
            'cac': {  # 2nd symbolic time-invariant control, Fraction of active CaO in mineral wt%
                'max': 54.5,  # Maximum allowable signal level, design space upper bound
                'min': 10  # Minimum allowable signal level, design space upper bound
            },
            'aps': {  # 3rd symbolic time-invariant control, Average particle size (m)
                'max': 1e-4,  # Maximum allowable signal level, design space upper bound
                'min': 1e-5  # Minimum allowable signal level, design space upper bound
            },
            'mld': {  # 4th symbolic time-invariant control, Molar density of reaction product (mol/m³)
                'max': 40000,  # Maximum allowable signal level, design space upper bound
                'min': 30000  # Minimum allowable signal level, design space upper bound
            },
        },
        'ti_ophi': {  # Time-invariant output variables (empty here, could hold steady state responses that hold no dependency)
        },
        't_s': [600, 10800],  # Time span  (600 s to 10,800 s), duration of numerical perturbations (the rest is precluded from design)
        't_r': 108,  # Time resolution (10 s), minimum time steps for the simulation/design/controls
    }

    design_settings = { # Design settings for the experiment
        'eps': 1e-3, #perturbation size of parameters in SA FDM method (in a normalized to 1 space)
        'optimization_methods': {
            'ppopt_method': 'G_P', # optimization method for MBDoE-PP, 'L': trust-constr method in .py, 'G_P': differential evolution in .py, 'GL': + penalised differential evolution + trust-constr in .py .  Note: L adn G are multi start due to child processes created in framework_settings, GL in single start, and DE of it is a penalised. 'G_P' is suggested for usage over the rest
            'mdopt_method': 'L' # optimization method for MBDoE-MD, 'L': trust-constr method in .py, 'G_P': differential evolution in .py, 'GL': + penalised differential evolution + trust-constr in .py .  Note: L adn G are multi start due to child processes created in framework_settings, GL in single start, and DE of it is a penalised. 'G_P' is suggested for usage over the rest
        },
        'criteria': {
            'MBDOE_MD_criterion': 'HR', # MD optimality criterion, 'HR': Hunter and Reiner, 'BFF': Buzzi-Ferraris and Forzatti
            'MBDOE_PP_criterion': 'E'  # PP optimality criterion, 'D', 'A', 'E', 'ME'
        },
        'iteration_settings': {
            'maxmd': 100, # maximum number of MD runs
            'tolmd': 1e-3, # tolerance for MD optimization
            'maxpp': 100, # maximum number of PP runs
            'tolpp': 1e-2, # tolerance for PP optimization
        }
    }

    modelling_settings = { # Settings related to the rival models and their parameters
        'ext_func': {'f17': f17}, # External functions (models) to be used in the experiment from global space
        'active_models': ['f11'], # Active solvers (rival models) to be used in the experiment
        'sim': {'f11': 'gp'}, # select the simulator of each model (model should be defined in the simulator, sci means in your python environment, gp means gPAS extracted gPROSMs models)
        'gpmodels': {
            'credentials': {'f11': '@@TTmnoa698'},  # credentials for gPAS models, if not needed, leave empty
            'connector': {'f11': 'C:/Users/Tadmin/Desktop/f11/model7.zip'},            # for now only for gPAS readable files, it is the path to zip file
        },
        'theta_parameters': { # Theta parameters for each model
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
        'bound_max': { # Maximum bounds for theta parameters (based on normalized to 1)
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
        'bound_min': { # Minimum bounds for theta parameters (based on normalized to 1)
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
    GSA_settings = { # Settings for the Global Sensitivity Analysis (GSA)
        'perform_sensitivity': False, # Perform sensitivity analysis
        'phi_nom': [3191, 44.93, 5.5e-5, 36000], # Nominal values for the time-invariant variables
        'phit_nom': [293.15, 1], # Nominal values for the time-variant variables
        'var_damping': False, # in case of GSA forvariables, True: use marginal damping factor, False: use design space
        'par_damping': True, # in case of GSA for parameters, True: use marginal damping factor, False: use estimation space
        'parallel': True, # Perform GSA in parallel
        'power': 0.7, # CPU core usage ratio
        'var_damping_factor': 1.1, # Damping factor for variables (in case on)
        'par_damping_factor': 1.1, # Damping factor for parameters (in case on)
        'sampling': 2**10, # Sampling size for GSA, always 2**n
        'var_sensitivity': False, # Perform sensitivity analysis for variables
        'par_sensitivity': True   # Perform sensitivity analysis for parameters
    }


    # scms
    simulator_settings = { # Settings for the insilico data generation
        'insilico_model': 'f11', # selected true model (with nominal values)
        'classic-des': { # classic design settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
            '1': {'T': 293.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000},
            '2': {'T': 313.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000}
            # '3': {'T': 333.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000},
            # '4': {'T': 353.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000}
        }
    }


    estimation_settings = { # Settings for the parameter estimation process
        'method': 'Ls',  # optimisation method, 'G': Global Differential Evolution, 'Ls': Local SLSQP, 'Ln': Local Nelder-Mead
        'initialization': 'random',   # use 'random' to have random starting point and use None to start from theta_parameters nominal values (to be avoided in insilico studies)
        'eps': 1e-2,  # perturbation size of parameters in SA FDM method (in a normalized to 1 space)
        #usually 1e-3, or None to perform a mesh independency test, and auto adjustment
        'objf': 'JWLS',  #loss function, 'LS': least squares, 'MLE': maximum likelihood, 'Chi': chi-square, 'JWLS': weighted least squares
        'con_plot': False, # plot the confidence volumes
        'fit_plot': True, # plot the fitting results
        'logging': True # log the results
    }

    logic_settings = { # Logic settings for the workflow
        'max_MD_runs': 1, # maximum number of MBDoE-MD runs
        'max_PP_runs': 1, # maximum number of MBDoE-PP runs
        'md_conf_tresh': 85, # discrimination acceptance test:  minimum P-value of a model to get accepted (%)
        'md_rej_tresh': 15, # discrimination acceptance test:  maximum P-value of a model to get rejected (%)
        'pp_conf_threshold': 1, # precision acceptance test:  times the ref statistical T value in worst case scenario
        'parallel_sessions': 1 # number of parallel sessions to be used in the workflow
    }

    framework_settings = { # Framework settings for saving the results
        'path': 'C:\\datasim', # path to save the results
        'case': 8 # case number as the name of subfolder to be created and getting used for restoring
    }

    def run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                      simulator_settings, estimation_settings, GSA_settings):
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
        start_time = time.time()  # Track the starting time for measuring total runtime
        data_storage = {}  # Storage for collected data in different rounds
        round_data = {}  # Tracks results of identification per round of estimation
        design_decisions = {}  # Tracks experimental design decisions
        winner_solver = None  # Tracks the model with the highest discrimination performance
        winner_solver_found = False  # Tracks if a winning model is identified

        # Check if only one active model exists; if true, no MD analysis needed
        if len(modelling_settings['active_models']) == 1:
            winner_solver_found = True
            winner_solver = modelling_settings['active_models'][0]
            print("There are no rival models for:", winner_solver)

        # Perform Sobol Sensitivity Analysis if enabled in GSA settings
        if GSA_settings['perform_sensitivity']:
            sobol_results = {}  # Dictionary to store Sobol sensitivity results
            sobol_results['sobol_analysis_results'], sobol_results['sobol_problem'] = sensa(
                GSA_settings, modelling_settings, model_structure, framework_settings)

        # Run the initial round of experiments (Classic Design)
        data_storage = run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings,
                                         estimation_settings, round_data, design_decisions, data_storage)

        # If no winner identified, proceed with Model Discrimination rounds
        if not winner_solver_found:
            winner_solver_found, winner_solver = run_md_rounds(framework_settings, model_structure, modelling_settings,
                                                               simulator_settings, estimation_settings, logic_settings,
                                                               design_settings, logic_settings['parallel_sessions'],
                                                               round_data, design_decisions, data_storage)

        terminate_loop = False  # Tracks if the framework's termination condition is met

        # Proceed with Parameter Precision rounds if a winner is identified
        if winner_solver_found:
            terminate_loop = run_pp_rounds(framework_settings, model_structure, modelling_settings, simulator_settings,
                                           estimation_settings, logic_settings, design_settings,
                                           logic_settings['parallel_sessions'], winner_solver, round_data,
                                           design_decisions, data_storage)

            if terminate_loop:
                print("Loop terminated successfully.")
        else:
            print("No winner model found, exiting without termination loop.")

        print(f"Round keys: {round_data.keys()}")
        print('---------------------------------------------')

        # # Perform final validation with cross-validation metrics
        # R2_prd, R2_val, parameters = validation(data_storage, system, models, iden_opt,
        #                                         Simula, round_data, framework_settings)

        # Ensure the output folder structure exists for saving results
        base_path = framework_settings['path']
        modelling_folder = str(framework_settings['case'])
        path = os.path.join(base_path, modelling_folder)
        os.makedirs(path, exist_ok=True)

        # Calculate and display total runtime
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime of framework: {runtime} seconds")

        # Save results in .jac files (JSON compressed files)
        folder_path = path
        file_path = os.path.join(folder_path, 'results.jac')
        file_path2 = os.path.join(folder_path, 'results2.jac')

        os.makedirs(folder_path, exist_ok=True)
        save_to_jac(round_data, file_path)

        # Save Sobol results if sensitivity analysis was conducted
        if GSA_settings['perform_sensitivity']:
            save_to_jac(sobol_results, file_path2)


    def run_initial_round(framework_settings, model_structure, modelling_settings, simulator_settings,
                          estimation_settings, round_data, design_decisions, data_storage):
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
        # Loop through classic design experiments defined in `insilicos`
        for i in range(len(simulator_settings['classic-des'])):
            j = i + 1  # Round counter (starting from 1)
            expr = i + 1  # Experiment ID
            case = 'classic'  # Case label for classic design

            # Generate experimental data (insilico or from input profiles)
            excel_path, df_combined = expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j)

            # Write generated data to Excel and read it back for further processing
            write_excel(df_combined, expr, excel_path)
            data = read_excel(excel_path)

            # Append data to the main storage for future analysis
            data_storage = data_appender(df_combined, expr, data)

            # Perform parameter estimation using the available data
            resultpr = parmest(
                model_structure,
                modelling_settings,
                estimation_settings,
                data_storage,
                simula
            )

            # Perform uncertainty analysis to compute confidence intervals, etc.
            resultun, theta_parameters, solver_parameters, scaled_params, obs = uncert(
                data_storage,
                resultpr,
                model_structure,
                modelling_settings,
                estimation_settings,
                simula
            )

            # # Perform estimability analysis only in the first two rounds
            # if round in [1, 2]:
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
            #     # Skip estimability analysis in later rounds
            #     ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            # Save results for this round, including estimates, uncertainties, and observations
            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'classic design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)

        # Return the collected data for subsequent rounds
        return data_storage

    def run_md_rounds(framework_settings, model_structure, modelling_settings, simulator_settings, estimation_settings,
                      logic_settings, design_settings, num_parallel_runs,
                      round_data, design_decisions, data_storage):
        """
        Run multiple rounds of Model-Based Design of Experiments (MBDOE) using the MD approach.

        Parameters:
        - framework_settings: Settings for framework paths and configurations.
        - system: Defines the model's structure, variables, and configurations.
        - models: Includes settings for theta parameters and active solvers.
        - insilicos: Defines simulation settings for generating data.
        - iden_opt: Settings for parameter estimation, solvers, and confidence checks.
        - logic_settings: Control settings for MD and PP runs, maximum iteration limits, etc.
        - des_opt: Design-specific settings for MD/PP methods and constraints.
        - num_parallel_runs: Number of parallel processes for computational efficiency.
        - round_data: Stores results for each experimental round.
        - design_decisions: Tracks design decisions for experiments.
        - data_storage: Stores collected or generated experimental data.

        Returns:
        tuple: (winner_solver_found, winner_solver) - Boolean for success and name of winning model.
        """
        winner_solver_found = False  # Flag to track if a winner model is found
        winner_solver = None  # Placeholder for the identified winning model
        start_round = len(round_data) + 1  # Start counting from the next available round
        case = 'doe'  # Design of Experiment identifier

        # Iterate through MD rounds (maximum defined by logic_settings)
        for i in range(logic_settings['max_MD_runs']):
            j = start_round + i  # Current round index

            # Check optimization method for MBDoE-MD and apply appropriate logic
            if design_settings['optimization_methods']['mdopt_method'] == 'Local':
                # Parallel execution using multiprocessing for speed improvement
                with Pool(num_parallel_runs) as pool:
                    results_list = pool.starmap(run_mbdoe_md,
                                                [(design_settings, model_structure, modelling_settings, core_num,
                                                  framework_settings, j) for core_num in range(num_parallel_runs)])
                best_design_decisions, best_sum_squared_differences, best_swps = max(results_list, key=lambda x: x[1])
                design_decisions.update(best_design_decisions)  # Update design decisions with the best found
            elif design_settings['optimization_methods']['mdopt_method'] == 'Global':
                core_num = 1
                results_list = run_mbdoe_md(design_settings, model_structure, modelling_settings, core_num,
                                            framework_settings, j)
                best_design_decisions, best_sum_squared_differences, best_swps = results_list
                design_decisions.update(best_design_decisions)

            # Generate experiment data using updated design decisions
            expr = j
            excel_path, df_combined = expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j, swps=best_swps)
            write_excel(df_combined, expr, excel_path)  # Write experiment data to an Excel file
            data = read_excel(excel_path)  # Read data back for processing
            data_storage = data_appender(df_combined, expr, data_storage)  # Append data to storage

            # Perform parameter estimation using Parmest
            resultpr = parmest(model_structure, modelling_settings, estimation_settings, data, simula)

            # Perform uncertainty analysis to identify potential solvers and assess variability
            resultun, theta_parameters, solver_parameters, scaled_params, obs = uncert(
                data, resultpr, model_structure, modelling_settings, estimation_settings, simula)

            ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None  # Default placeholders

            # Check discrimination thresholds for model selection
            p_r_threshold = logic_settings['md_rej_tresh']  # Rejection threshold for solvers
            p_a_threshold = logic_settings['md_conf_tresh']  # Acceptance threshold for solvers
            for solver, solver_results in resultun.items():
                if solver_results['P'] < p_r_threshold:  # Remove rejected solvers
                    modelling_settings['active_models'].remove(solver)
                if solver_results['P'] > p_a_threshold:  # Identify accepted model (winning model)
                    winner_solver = solver
                    winner_solver_found = True
                    break

            # If this is the final iteration, pick the highest P-value model as the winner
            if i == logic_settings['max_MD_runs'] - 1:
                highest_p_solver = max(resultun.items(), key=lambda x: x[1]['P'])[0]
                highest_p_value = resultun[highest_p_solver]['P']
                winner_solver_found = True
                winner_solver = highest_p_solver
                print(
                    f"Model discrimination ended after {i + 1} iterations. Model '{winner_solver}' selected as winner based on P value = {highest_p_value}.")

            # Save round data with key metrics and results
            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'MBDOE_MD design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)

            # Exit loop early if a winner is identified before max iterations
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
        system (dict): User provided - Structure of the model, including variables and their properties.
        models (dict): User provided - Settings related to the modelling process, including theta parameters.
        insilicos (dict): User provided - Settings for the simulator, including standard deviation and model name.
        iden_opt (dict): User provided - Settings for the estimation process, including active solvers and plotting options.
        logic_settings (dict): User provided - Logic settings for the identification process, including thresholds and maximum runs.
        des_opt (dict): User provided - Design settings for the experiment, including mutation and crossover rates.
        num_parallel_runs (int): Number of parallel runs to execute MBDOE-PP.
        winner_solver (str): The name of the winning model from the MD rounds.
        round_data (dict): Collection of saved data from each round of the experiment.
        design_decisions (dict): Design decisions for the experiment, to be performed in-silico or in-bench.
        data_storage (dict): Storage for the experimental data observations.

        Returns:
        bool: Indicates if the loop was terminated based on the confidence threshold.
        """
        terminate_loop = False  # Flag to stop iterations if termination conditions are met
        start_round = len(round_data) + 1  # Start round number for PP experiments
        case = 'doe'  # Case identifier

        for i in range(logic_settings['max_PP_runs']):  # Iterate over maximum PP runs
            j = start_round + i  # Round number tracking
            round = j  # Alias for the round number

            # Assign the winner model as the only active model for the PP round
            modelling_settings['active_models'] = [winner_solver]
            method = design_settings['optimization_methods']['ppopt_method']

            if method in ['L', 'G_P']:
                # Local/Constrained optimization using parallel execution
                with Pool(num_parallel_runs) as pool:
                    results_list = pool.starmap(
                        run_mbdoe_pp,
                        [(design_settings, model_structure, modelling_settings, core_num,
                          framework_settings, round) for core_num in range(num_parallel_runs)]
                    )
                best_design_decisions, best_pp_obj, best_swps = max(results_list, key=lambda x: x[1])
                design_decisions.update(best_design_decisions)

            # elif method == 'G_P':
            #     # Global optimization (pre-screened) using single-core execution
            #     result = run_mbdoe_pp(des_opt, system, models, 0,
            #                           framework_settings, round)
            #     best_design_decisions, best_pp_obj, best_swps = result
            #     design_decisions.update(best_design_decisions)

            elif method == 'GL':
                # Global optimization (single-core execution)
                result = run_mbdoe_pp(design_settings, model_structure, modelling_settings, 0,
                                      framework_settings, round)
                best_design_decisions, best_pp_obj, best_swps = result
                design_decisions.update(best_design_decisions)

            # Data generation and storage
            expr = j  # Experiment ID
            excel_path, df_combined = expera(framework_settings, model_structure, modelling_settings,
                                             simulator_settings, design_decisions, j, swps=best_swps)
            write_excel(df_combined, expr, excel_path)  # Save data to an Excel file
            data = read_excel(excel_path)  # Load data from the file
            data_storage = data_appender(df_combined, expr, data_storage)  # Append data to storage

            # Parameter Estimation and Uncertainty Analysis
            resultpr = parmest(
                model_structure,
                modelling_settings,
                estimation_settings,
                data,
                simula
            )

            resultun, theta_parameters, solver_parameters, scaled_params, obs = uncert(
                data,
                resultpr,
                model_structure,
                modelling_settings,
                estimation_settings,
                simula
            )

            # Save results for further analysis
            ranking, k_optimal_value, rCC_values, J_k_values = None, None, None, None
            trv = save_rounds(j, ranking, k_optimal_value, rCC_values, J_k_values, resultun, theta_parameters,
                              'MBDOE_PP design', round_data, modelling_settings, scaled_params, estimation_settings,
                              solver_parameters, framework_settings, obs, case, data_storage, model_structure)

            # Check termination condition: all t-values must exceed threshold in resultun
            for solver, solver_results in resultun.items():
                if all(t_value > trv[solver] for t_value in solver_results['t_values']):
                    terminate_loop = True  # Set termination flag if condition met
                    break

            if terminate_loop:  # Exit the loop if termination conditions are met
                break

        return terminate_loop


    run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                  simulator_settings, estimation_settings, GSA_settings)




if __name__ == '__main__':
    main()

