def main():

    theta1 = [0.31, 0.18, 0.65, 0.25, 5]  # true values
    # theta_n1 = [0.372, 0.216, 0.78, 0.3, 6.0] # nominal values +20%
    theta_n1 = [0.372, 0.216, 0.78, 0.3, 6.0] # nominal values +20%
    theta_min1 = [0.31, 0.18, 0.65, 0.25, 5]   # minimum values -50%
    theta_max1 = [0.465, 0.27, 0.975, 0.375, 7.5]  # maximum values +50%
    theta_maxs1 = [hi / nom for hi, nom in zip(theta_max1, theta_n1)]
    theta_mins1 = [lo / nom for lo, nom in zip(theta_min1, theta_n1)]


    system = {
        'tvi': {  # Time-variant input variables (models input: tvi), each key is a symbol nad key in tvi as well
            'u1': {  # Temperature (K)
                'stps': 5,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. swps-1 is the number of steps
                'const': 'rel',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 0.2,  # Maximum allowable signal level, des_opt space upper bound
                'min': 0.05,  # Minimum allowable signal level, des_opt space lower bound
                'cvp': 'CPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offl': 0.01,  # minimum allowed perturbation of signal (ratio)
                'offt': 0.5  # minimum allowed perturbation of time (ratio)
            },
            'u2': {  # Temperature (K)
                'stps': 5,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. swps-1 is the number of steps
                'const': 'dec',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 35,  # Maximum allowable signal level, des_opt space upper bound
                'min': 5,  # Minimum allowable signal level, des_opt space lower bound
                'cvp': 'CPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offl': 5,  # minimum allowed perturbation of signal (ratio)
                'offt': 0.5  # minimum allowed perturbation of time (ratio)
            },
        },
        'tvo': {  # Time-variant output variables (responses, measured or unmeasured)
            'y1': {  # response variable, here carbonation efficiency
                'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 6,  # the amound of samples per each round (run)
                'unc': 0.05,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.5,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 10],  # fixed sampling times
            },
            'y2': {  # response variable, here carbonation efficiency
                'init': 0,  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 6,  # the amound of samples per each round (run)
                'unc': 0.05,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.5,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 10],  # fixed sampling times
            },
        },
        'tii': {  # Time-invariant input variables (tii)
            'y10': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 10,  # Maximum allowable signal level, des_opt space upper bound
                'min': 1  # Minimum allowable signal level, des_opt space upper bound
            },
            'y20': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 10,  # Maximum allowable signal level, des_opt space upper bound
                'min': 1  # Minimum allowable signal level, des_opt space upper bound
            },
        },
        'tio': {  # Time-invariant output variables (empty here, could hold steady state responses that hold no dependency)
        },
        't_s': [0, 10],  # Time span  (600 s to 10,800 s), duration of numerical perturbations (the rest is precluded from des_opt)
        't_r': 0.02,  # Time resolution (10 s), minimum time steps for the simulation/des_opt/controls
        't_d': 0.5
    }

    models = { # Settings related to the rival models and their parameters
        'can_m': ['M'],  # Active solvers (rival models) to be used in the experiment
        'krt': {'M': 'pys'},  # Kernel type for each model, 'pys' for python standalone scripts, 'pym' for middoe.krnl_models, 'gpr' for gPAS models
        # type of the model interface, 'pym' for middoe.krnl_models, 'gpr' for gPAS models, function name for globally defined functions, 'pys' for python standalone scripts
        'creds': {'M': '@@TTmnoa698'},
        # credentials for gPAS models, if not needed, leave empty
        'src': {'M': 'C:/Users/Tadmin/PycharmProjects/middoe/tests/poster2/case 1/model.py'},
        # for now for gPAS readable files, or python standalone scripts

        'theta': { # Theta parameters for each models
            'M': theta_n1
        },
        't_u': { # Maximum bounds for theta parameters (based on normalized to'f20': theta20mins, 1)
            'M': theta_maxs1
        },
        't_l': { # Minimum bounds for theta parameters (based on normalized to 1)
            'M': theta_mins1
        }
    }



    des_opt = { # Design settings for the experiment
        'eps': 1e-3, #perturbation size of parameters in SA FDM method (in a normalized to 1 space)
        'meth': 'L',  # optimisation method, 'G': Global Differential Evolution, 'L': Local Pattern Search, 'GL': Global Differential Evolution refined with Local Pattern Search
        'md_ob': 'BFF',     # MD optimality criterion, 'HR': Hunter and Reiner, 'BFF': Buzzi-Ferraris and Forzatti
        'pp_ob': 'E',  # PP optimality criterion, 'D', 'A', 'E', 'ME'
        'plt': True,  # Plot the results
        'itr': {
            'pps': 30, # population size
            'maxmd': 60, # maximum number of MD runs
            'tolmd': 1, # tolerance for MD optimization
            'maxpp':20 ,# maximum number of PP runs
            'tolpp': 1, # tolerance for PP optimization
        }
    }


    from middoe.des_pp import mbdoe_pp
    # designs = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=16)

    insilicos = { # Settings for the insilico data generation
        'tr_m': 'M', # selected true models (with nominal values)
        'theta': theta1,
        'errt': 'rel',  # error type, 'rel' for relative error, 'abs' for absolute error
        'prels': {
            '1': {'u1': 0.05, 'u2': 5, 'y10': 1, 'y20': 1},
            # classic des_opt settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
        }
    }

    from middoe.krnl_expera import expera

    # expera(system, models, insilicos, design_decisions={}, expr=1)

    # expera(system, models, insilicos, designs, expr=1, swps=designs['swps'])

    from middoe.log_utils import  read_excel
    data = read_excel('indata')



    iden_opt = { # Settings for the parameter estimation process
        'meth': 'Ls',  # optimisation method, 'G': Global Differential Evolution, 'Ls': Local SLSQP, 'Ln': Local Nelder-Mead
        'init': None,   # use 'rand' to have random starting point and use None to start from theta_parameters nominal values (to be avoided in insilico studies)
        'eps': 1e-4,  # perturbation size of parameters in SA FDM method (in a normalized to 1 space)
        #usually 1e-3, or None to perform a mesh independency test, and auto adjustment
        'ob': 'WLS',  #loss function, 'LS': least squares, 'MLE': maximum likelihood, 'Chi': chi-square, 'WLS': weighted least squares
        'c_plt': True, # plot the confidence volumes
        'f_plt': True, # plot the fitting results
        'plt_s': True, # show plots while saving
        'log': True # log the results
    }


    from middoe.iden_parmest import parmest
    resultpr = parmest(system, models, iden_opt, data)

    from middoe.iden_uncert import uncert
    uncert_results = uncert(data, resultpr, system, models, iden_opt)
    resultun = uncert_results['results']
    theta_parameters = uncert_results['theta_parameters']
    solver_parameters = uncert_results['solver_parameters']
    scaled_params = uncert_results['scaled_params']
    obs = uncert_results['obs']



    from middoe.log_utils import  save_rounds
    round_data={}
    round = 1
    save_rounds(round, resultun, theta_parameters, 'preliminary', round_data, models, scaled_params,iden_opt,solver_parameters, obs, data, system)


if __name__ == '__main__':
    main()