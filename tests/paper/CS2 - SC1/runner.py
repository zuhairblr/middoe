#
#
# import numpy as np
#
#
# theta1 = [50000, 75000, 0.4116, 111900, 9905, 30000]
# theta_n1 = [100000, 100000, 1, 100000, 100, 10000]
# theta_min1 = [10000, 0, 0.1, 50000, 10, 10000]
# theta_max1 = [1000000, 200000, 10, 200000, 10000, 200000]
# theta_maxs1 = [hi / nom for hi, nom in zip(theta_max1, theta_n1)]
# theta_mins1 = [lo / nom for lo, nom in zip(theta_min1, theta_n1)]
#
# theta = [0.4116, 111900, 9905]
# theta_n = [1, 100000, 100]
# theta_min = [0.1, 50000, 10]
# theta_max = [ 10, 200000, 10000]
# theta_maxs = [hi / nom for hi, nom in zip(theta_max, theta_n)]
# theta_mins = [lo / nom for lo, nom in zip(theta_min, theta_n)]
#
# def main():
#
#
#     system = {
#         'tvi': {  # Time-variant input variables (models input: tvi), each key is a symbol nad key in tvi as well
#             'u1': {  # Temperature (K)
#                 'stps': 6,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
#                 # Must be a positive integer > 1. swps-1 is the number of steps
#                 'const': 'inc',  # Constraint type: relative state of signal levels in CVPs
#                 # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
#                 'max': 306.15,  # Maximum allowable signal level, des_opt space upper bound
#                 'min': 296.15,  # Minimum allowable signal level, des_opt space lower bound
#                 'cvp': 'LPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
#                 'offl': 0.01,  # minimum allowed perturbation of signal (ratio)
#                 'offt': 0.3  # minimum allowed perturbation of time (ratio)
#             },
#         },
#         'tvo': {  # Time-variant output variables (responses, measured or unmeasured)
#             'y1': {  # response variable, here carbonation efficiency
#                 'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
#                 'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
#                 'sp': 17,  # the amound of samples per each round (run)
#                 'unc': 0.005,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
#                 'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
#                 'samp_s': 1,  # Matching criterion for models prediction and data alignment
#                 'samp_f': [0, 16],  # fixed sampling times
#             },
#             'y2': {  # response variable, here carbonation efficiency
#                 'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
#                 'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
#                 'sp': 17,  # the amound of samples per each round (run)
#                 'unc': 0.005,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
#                 'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
#                 'samp_s': 1,  # Matching criterion for models prediction and data alignment
#                 'samp_f': [0, 16],  # fixed sampling times
#             },
#             'y3': {  # response variable, here carbonation efficiency
#                 'init': 0,
#                 # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
#                 'meas': True,
#                 # Flag indicating if this variable is directly measurable, if False, it is a virtual output
#                 'sp': 17,  # the amound of samples per each round (run)
#                 'unc': 0.005,
#                 # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
#                 'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
#                 'samp_s': 1,  # Matching criterion for models prediction and data alignment
#                 'samp_f': [0, 16],  # fixed sampling times
#             },
#         },
#         'tii': {  # Time-invariant input variables (tii)
#             'y10': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
#                 'max': 0.3,  # Maximum allowable signal level, des_opt space upper bound
#                 'min': 1  # Minimum allowable signal level, des_opt space upper bound
#             },
#             'y20': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
#                 'max': 0.19,  # Maximum allowable signal level, des_opt space upper bound
#                 'min': 1  # Minimum allowable signal level, des_opt space upper bound
#             },
#         },
#         'tio': {  # Time-invariant output variables (empty here, could hold steady state responses that hold no dependency)
#         },
#         't_s': [0, 16],  # Time span  (600 s to 10,800 s), duration of numerical perturbations (the rest is precluded from des_opt)
#         't_r': 0.02,  # Time resolution (10 s), minimum time steps for the simulation/des_opt/controls
#         't_d': 0.3
#     }
#
#
#
#     models = { # Settings related to the rival models and their parameters
#         'can_m': ['M'],  # Active solvers (rival models) to be used in the experiment
#         'krt': {'M': 'pys'},  # Kernel type for each model, 'pys' for python standalone scripts, 'pym' for middoe.krnl_models, 'gpr' for gPAS models
#         # type of the model interface, 'pym' for middoe.krnl_models, 'gpr' for gPAS models, function name for globally defined functions, 'pys' for python standalone scripts
#         'creds': {'M': '@@TTmnoa698'},
#         # credentials for gPAS models, if not needed, leave empty
#         'src': {'M': 'C:/Users/Tadmin/PycharmProjects/middoe/tests/paper/2/model.py'},
#         # for now for gPAS readable files, or python standalone scripts
#
#         'theta': { # Theta parameters for each models
#             'M': theta_n
#         },
#         't_u': { # Maximum bounds for theta parameters (based on normalized to'f20': theta20mins, 1)
#             'M': theta_maxs
#         },
#         't_l': { # Minimum bounds for theta parameters (based on normalized to 1)
#             'M': theta_mins
#         }
#     }
#
#     gsa = { # Settings for the Global Sensitivity Analysis (gsa)
#         'var_s': False,  # Perform sensitivity analysis for variables
#         'par_s': True,  # Perform sensitivity analysis for parameters
#         'var_d': False, # feasible space for variables, fload ratio: use as multiplier to nominals uniformly (e.g. 1.1), False: use system defined space
#         'par_d': False,   # feasible space for parameters, fload ratio: use as multiplier to nominals uniformly(e.g. 1.1), False: use models defined space
#         'samp': 2 ** 12,  # Sampling size for gsa, always 2**n
#         'multi': 0.7,  # Perform gsa in parallel
#         'tii_n': [0.366,0.19], # Nominal values for the time-invariant variables
#         'tvi_n': [296.15], # Nominal values for the time-variant variables
#         'plt': True,  # Plot the results
#     }
#
#
#     #
#     # from middoe.sc_sensa import sensa
#     # sobol_results = sensa(gsa, models, system)
#     #
#     #
#     # from middoe.log_utils import save_to_jac
#     # save_to_jac(sobol_results, purpose="sensa")
#     #
#     # from middoe.log_utils import load_from_jac, save_to_xlsx
#     #
#     # results = load_from_jac()
#     # sensa = results['sensa']
#     # save_to_xlsx(sensa)
#
#
#     insilicos = { # Settings for the insilico data generation
#         'tr_m': 'M', # selected true models (with nominal values)
#         'theta': theta1,
#         'errt': 'abs',  # error type, 'rel' for relative error, 'abs' for absolute error
#         'prels': { # classic des_opt settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
#             '1': {'u1': 296.15, 'y10': 0.366, 'y20': 0.19},
#             '2': {'u1': 306.15, 'y10': 0.366, 'y20': 0.19},
#         }
#     }
#
#     modelsc = {  # Settings related to the rival models and their parameters
#         'can_m': ['M'],  # Active solvers (rival models) to be used in the experiment
#         'krt': {'M': 'pys'},
#         # Kernel type for each model, 'pys' for python standalone scripts, 'pym' for middoe.krnl_models, 'gpr' for gPAS models
#         # type of the model interface, 'pym' for middoe.krnl_models, 'gpr' for gPAS models, function name for globally defined functions, 'pys' for python standalone scripts
#         'creds': {'M': '@@TTmnoa698'},
#         # credentials for gPAS models, if not needed, leave empty
#         'src': {'M': 'C:/Users/Tadmin/PycharmProjects/middoe/tests/paper/2/model2.py'},
#         # for now for gPAS readable files, or python standalone scripts
#
#         'theta': {  # Theta parameters for each models
#             'M': theta_n1
#         },
#         't_u': {  # Maximum bounds for theta parameters (based on normalized to'f20': theta20mins, 1)
#             'M': theta_maxs1
#         },
#         't_l': {  # Minimum bounds for theta parameters (based on normalized to 1)
#             'M': theta_mins1
#         }
#     }
#
#
#     from middoe.krnl_expera import expera
#     # expera(system, models, insilicos, design_decisions={}, expr=1)
#     #
#     #
#     # expera(system, models, insilicos, design_decisions={}, expr=2)
#
#     iden_opt = { # Settings for the parameter estimation process
#         'meth': 'G',  # optimisation method, 'G': Global Differential Evolution, 'Ls': Local SLSQP, 'Ln': Local Nelder-Mead
#         'init': None,   # use 'rand' to have random starting point and use None to start from theta_parameters nominal values (to be avoided in insilico studies)
#         'eps': 1e-3,  # perturbation size of parameters in SA FDM method (in a normalized to 1 space)
#         #usually 1e-3, or None to perform a mesh independency test, and auto adjustment
#         'ob': 'WLS',  #loss function, 'LS': least squares, 'MLE': maximum likelihood, 'Chi': chi-square, 'WLS': weighted least squares
#         'c_plt': True, # plot the confidence volumes
#         'f_plt': True, # plot the fitting results
#         'plt_s': True, # show plots while saving
#         'log': True # log the results
#     }
#
#     from middoe.log_utils import  read_excel
#     data = read_excel('indata')
#
#     from middoe.iden_parmest import parmest
#     resultpr = parmest(system, models, iden_opt, data)
#
#     from middoe.iden_uncert import uncert
#     uncert_results = uncert(data, resultpr, system, models, iden_opt)
#     resultun = uncert_results['results']
#     theta_parameters = uncert_results['theta_parameters']
#     solver_parameters = uncert_results['solver_parameters']
#     scaled_params = uncert_results['scaled_params']
#     obs = uncert_results['obs']
#
#     # from middoe.sc_estima import estima
#     # j = 2
#     # ranking, k_optimal_value, rCC_values, J_k_values = estima(resultun, system, models, iden_opt, j, data)
#
#     from middoe.log_utils import  read_excel, save_rounds
#     round_data={}
#     round = 1
#     save_rounds(round, resultun, theta_parameters, 'preliminary', round_data, models, scaled_params,iden_opt,solver_parameters, obs, data, system)
#
#
#
#     des_opt = { # Design settings for the experiment
#         'eps': 1e-3, #perturbation size of parameters in SA FDM method (in a normalized to 1 space)
#         'md_ob': 'BFF',     # MD optimality criterion, 'HR': Hunter and Reiner, 'BFF': Buzzi-Ferraris and Forzatti
#         'pp_ob': 'E',  # PP optimality criterion, 'D', 'A', 'E', 'ME'
#         'plt': True,  # Plot the results
#         'meth': 'L',
#         # optimisation method, 'G': Global Differential Evolution, 'L': Local Pattern Search, 'GL': Global Differential Evolution refined with Local Pattern Search
#         'itr': {
#             'pps': 100, # population size
#             'maxmd': 5, # maximum number of MD runs
#             'tolmd': 1, # tolerance for MD optimization
#             'maxpp':500 ,# maximum number of PP runs
#             'tolpp': 1, # tolerance for PP optimization
#         }
#     }
#
#     #
#     # from middoe.des_pp import mbdoe_pp
#     #
#     # # Loop from round=2 to round=6 (inclusive), which is 5 rounds
#     # for i in range(2, 6):
#     #     expr_tag = i + 1  # expr starts at 3 when round=2
#     #
#     #     print(f"=== Running MBDOE-PP Round {i} ===")
#     #
#     #     # STEP 1: MBDOE-PP Design
#     #     designs = mbdoe_pp(des_opt, system, models, round=i, num_parallel_runs=16)
#     #
#     #     # STEP 2: In silico experiment
#     #     expera(system, modelsc, insilicos, designs, expr=expr_tag, swps=designs['swps'])
#     #
#     #     # STEP 3: Read experimental data
#     #     data = read_excel('indata')
#     #
#     #     # STEP 4: Parameter estimation
#     #     resultpr = parmest(system, models, iden_opt, data)
#     #
#     #     # STEP 5: Uncertainty analysis
#     #     uncert_results = uncert(data, resultpr, system, models, iden_opt)
#     #     resultun = uncert_results['results']
#     #     theta_parameters = uncert_results['theta_parameters']
#     #     solver_parameters = uncert_results['solver_parameters']
#     #     scaled_params = uncert_results['scaled_params']
#     #     obs = uncert_results['obs']
#     #
#     #     # STEP 6: Save round data
#     #     save_rounds(
#     #         i,  # round number
#     #         resultun,
#     #         theta_parameters,
#     #         'MBDOE_PP',
#     #         round_data,
#     #         models,
#     #         scaled_params,
#     #         iden_opt,
#     #         solver_parameters,
#     #         obs,
#     #         data,
#     #         system
#     #     )
#
#     from middoe.log_utils import save_to_jac
#     save_to_jac(round_data, purpose="iden")
#
#
# if __name__ == "__main__":
#     main()
#
# # from middoe.log_utils import load_from_jac
# # results = load_from_jac()
# # iden = results['iden']
# #
# # from middoe.iden_utils import run_postprocessing
# # run_postprocessing(
# #     round_data=results['iden'],
# #     solvers=['M'],
# #     selected_rounds=[ 1, 2, 3, 4, 5],
# #     plot_global_p_and_t=True,
# #     plot_confidence_spaces=True,
# #     plot_p_and_t_tests=True,
# #     export_excel_reports=True,
# #     plot_estimability=True
# # )


def main():

    theta = [50000, 75000, 0.4116, 111900, 9905, 30000]
    theta_n = [100000, 100000, 1, 100000, 100, 10000]
    theta_mins = [10000, 0, 0.1, 50000, 10, 10000]
    theta_maxs = [1000000, 200000, 10, 200000, 10000, 200000]

    system = {
        'tvi': {  # Time-variant input variables (models input: tvi), each key is a symbol nad key in tvi as well
            'u1': {  # Temperature (K)
                'stps': 6,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. swps-1 is the number of steps
                'const': 'inc',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 306.15,  # Maximum allowable signal level, des_opt space upper bound
                'min': 296.15,  # Minimum allowable signal level, des_opt space lower bound
                'cvp': 'LPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offl': 0.01,  # minimum allowed perturbation of signal (ratio)
                'offt': 0.3  # minimum allowed perturbation of time (ratio)
            },
        },
        'tvo': {  # Time-variant output variables (responses, measured or unmeasured)
            'y1': {  # response variable, here carbonation efficiency
                'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 17,  # the amound of samples per each round (run)
                'unc': 0.05,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 16],  # fixed sampling times
            },
            'y2': {  # response variable, here carbonation efficiency
                'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 17,  # the amound of samples per each round (run)
                'unc': 0.05,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 16],  # fixed sampling times
            },
            'y3': {  # response variable, here carbonation efficiency
                'init': 0,
                # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,
                # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 17,  # the amound of samples per each round (run)
                'unc': 0.05,
                # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.3,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 16],  # fixed sampling times
            },
        },
        'tii': {  # Time-invariant input variables (tii)
            'y10': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 1,  # Maximum allowable signal level, des_opt space upper bound
                'min': 0.3  # Minimum allowable signal level, des_opt space upper bound
            },
            'y20': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 1,  # Maximum allowable signal level, des_opt space upper bound
                'min': 0.19  # Minimum allowable signal level, des_opt space upper bound
            },
        },
        'tio': {  # Time-invariant output variables (empty here, could hold steady state responses that hold no dependency)
        },
        't_s': [0, 16],  # Time span  (600 s to 10,800 s), duration of numerical perturbations (the rest is precluded from des_opt)
        't_r': 0.02,  # Time resolution (10 s), minimum time steps for the simulation/des_opt/controls
        't_d': 0.3
    }



    models = { # Settings related to the rival models and their parameters
        'can_m': ['M'],  # Active solvers (rival models) to be used in the experiment
        'krt': {'M': 'pys'},  # Kernel type for each model, 'pys' for python standalone scripts, 'pym' for middoe.krnl_models, 'gpr' for gPAS models
        # type of the model interface, 'pym' for middoe.krnl_models, 'gpr' for gPAS models, function name for globally defined functions, 'pys' for python standalone scripts
        'creds': {'M': '@@TTmnoa698'},
        # credentials for gPAS models, if not needed, leave empty
        'src': {'M': 'C:/Users/Tadmin/PycharmProjects/middoe/tests/paper/CS2 - SC1/model.py'},
        # for now for gPAS readable files, or python standalone scripts

        'theta': { # Theta parameters for each models
            'M': theta_n
        },
        't_u': { # Maximum bounds for theta parameters (based on normalized to'f20': theta20mins, 1)
            'M': theta_maxs
        },
        't_l': { # Minimum bounds for theta parameters (based on normalized to 1)
            'M': theta_mins
        }
    }

    insilicos = { # Settings for the insilico data generation
        'tr_m': 'M', # selected true models (with nominal values)
        'theta': theta,
        'errt': 'rel',  # error type, 'rel' for relative error, 'abs' for absolute error
        'prels': { # classic des_opt settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
            '1': {'u1': 296.15, 'y10': 0.366, 'y20': 0.19},
            '2': {'u1': 306.15, 'y10': 0.366, 'y20': 0.19},
            '3': {'u1': 296.15, 'y10': 0.65, 'y20': 0.595},
            '4': {'u1': 306.15, 'y10': 0.65, 'y20': 0.595},
        }
    }

    iden_opt = {  # Settings for the parameter estimation process
        'meth': 'SLSQP',  # 'SLSQP', 'SQP', 'DE', 'NM', 'BFGS'
        'ms': False,  # multi starting   # True or False
        'sens_m': 'central',  # 'central', 'forward', and 'five' for FDM precision
        'var-cov': 'B',  # 'H' for based on hessidan, and 'M' for based on fisher
        'nboot': 100,
        'init': None,
        # use 'rand' to have random starting point and use None to start from theta_parameters nominal values (to be avoided in insilico studies)
        'eps': 1e-3,  # perturbation size of parameters in SA FDM method (in a normalized to 1 space)e-
        # usually 1e-3, or None to perform a mesh independency test, and auto adjustment
        'ob': 'WLS',
        # loss function, 'LS': least squares, 'MLE': maximum likelihood, 'Chi': chi-square, 'WLS': weighted least squares
        'c_plt': True,  # plot the confidence volumes
        'f_plt': True,  # plot the fitting results
        'plt_s': True,  # show plots while saving
        'log': True  # log the results
    }

    from middoe.log_utils import  read_excel
    data = read_excel('indata')
    from middoe.iden_parmest import parmest
    resultpr = parmest(system, models, iden_opt, data)


    from middoe.iden_uncert import uncert
    uncert_results = uncert(data, resultpr, system, models, iden_opt)
    resultun = uncert_results['results']
    obs = uncert_results['obs']


    # from middoe.sc_estima import estima
    # j = 4
    # ranking, k_optimal_value, rCC_values, J_k_values, best_uncert_result = estima(resultun, system, models, iden_opt, j, data)

if __name__ == "__main__":
    main()