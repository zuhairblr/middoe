import numpy as np
from middoe.log_utils import load_from_jac


def model_I(t, y, tii, tvi, theta, te):
    """
    Model I: Monod kinetics with maintenance.
    Assumes specific growth rate saturates with substrate and includes maintenance loss.
    """
    y1, y2 = y
    u1 = np.interp(t, te, tvi['u1'])
    u2 = np.interp(t, te, tvi['u2'])

    mu_max = theta[0]  # maximum specific growth rate (1/h)
    Ks     = theta[1]  # Monod saturation constant (g/L)
    Yxs    = theta[2]  # biomass yield coefficient (g biomass / g substrate)
    m      = theta[3]  # maintenance coefficient (1/h)

    r = mu_max * y2 / (Ks + y2)  # Monod rate
    dy1dt = (r - u1 - m) * y1
    dy2dt = -(r * y1 / Yxs) + u1 * (u2 - y2)
    return [dy1dt, dy2dt]


theta_I = [0.25, 0.25, 0.88, 0.09]

theta_In = [x * 1.5 for x in theta_I]
theta_min_I = [x * 0.5 for x in theta_I]  # 20% below nominal
theta_max_I = [x * 1.5 for x in theta_I]  # 20% above nominal
theta_maxs_I = [hi / nom for hi, nom in zip(theta_max_I, theta_I)]
theta_mins_I = [lo / nom for lo, nom in zip(theta_min_I, theta_I)]


def model_II(t, y, tii, tvi, theta, te):
    """
    Model II: Canoid kinetics.
    Incorporates dependence of growth on both biomass and substrate concentrations.
    """
    y1, y2 = y
    u1 = np.interp(t, te, tvi['u1'])
    u2 = np.interp(t, te, tvi['u2'])

    mu_max     = theta[0]  # maximum specific growth rate (1/h)
    Ks_biomass = theta[1]  # saturation constant influenced by biomass (L/g)
    Yxs        = theta[2]  # biomass yield coefficient
    m          = theta[3]  # maintenance coefficient

    r = mu_max * y2 / (Ks_biomass * y1 + y2)  # Canoid rate
    dy1dt = (r - u1 - m) * y1
    dy2dt = -(r * y1 / Yxs) + u1 * (u2 - y2)
    return [dy1dt, dy2dt]


theta_II = [0.24987, 0.00912, 0.94194, 0.10154]
theta_IIn = [x * 1.5 for x in theta_II]
theta_min_II = [x * 0.5 for x in theta_II]  # 20% below nominal
theta_max_II = [x * 1.5 for x in theta_II]  # 20% above nominal
theta_maxs_II = [hi / nom for hi, nom in zip(theta_max_II, theta_II)]
theta_mins_II = [lo / nom for lo, nom in zip(theta_min_II, theta_II)]


def model_III(t, y, tii, tvi, theta, te):
    """
    Model III: Linear growth kinetics.
    Assumes specific growth rate is linear with substrate.
    """
    y1, y2 = y
    u1 = np.interp(t, te, tvi['u1'])
    u2 = np.interp(t, te, tvi['u2'])

    a   = theta[0]  # linear rate coefficient (1/h·(g/L))
    Yxs = theta[1]  # biomass yield
    m   = theta[2]  # maintenance coefficient

    r = a * y2
    dy1dt = (r - u1 - m) * y1
    dy2dt = -(r * y1 / Yxs) + u1 * (u2 - y2)
    return [dy1dt, dy2dt]


theta_III = [0.02636, 0.68145, 0.03381]
theta_IIIn = [x * 1.5 for x in theta_III]
theta_min_III = [x * 0.5 for x in theta_III]  # 20% below nominal
theta_max_III = [x * 1.5 for x in theta_III]  # 20% above nominal
theta_maxs_III = [hi / nom for hi, nom in zip(theta_max_III, theta_III)]
theta_mins_III = [lo / nom for lo, nom in zip(theta_min_III, theta_III)]


def model_IV(t, y, tii, tvi, theta, te):
    """
    Model IV: Monod kinetics without maintenance.
    Simplified version of Model I, excluding the maintenance term.
    """
    y1, y2 = y
    u1 = np.interp(t, te, tvi['u1'])
    u2 = np.interp(t, te, tvi['u2'])

    mu_max = theta[0]  # max specific growth rate
    Ks     = theta[1]  # Monod constant
    Yxs    = theta[2]  # yield

    r = mu_max * y2 / (Ks + y2)
    dy1dt = (r - u1) * y1
    dy2dt = -(r * y1 / Yxs) + u1 * (u2 - y2)
    return [dy1dt, dy2dt]


theta_IV = [0.15270, 0.40575, 0.47529]
theta_IVn = [x * 1.5 for x in theta_IV]
theta_min_IV = [x * 0.5 for x in theta_IV]  # 20% below nominal
theta_max_IV = [x * 1.5 for x in theta_IV]  # 20% above nominal
theta_maxs_IV = [hi / nom for hi, nom in zip(theta_max_IV, theta_IV)]
theta_mins_IV = [lo / nom for lo, nom in zip(theta_min_IV, theta_IV)]


def main():

    system = {
        'tvi': {  # Time-variant input variables (models input: tvi), each key is a symbol nad key in tvi as well
            'u1': {  # dilution rate (h-1),
                'stps': 5,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. swps-1 is the number of steps
                'const': 'rel',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 0.2,  # Maximum allowable signal level, des_opt space upper bound
                'min': 0.05,  # Minimum allowable signal level, des_opt space lower bound
                'cvp': 'CPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offl': 0.01,  # minimum allowed perturbation of signal
                'offt': 0.5  # minimum allowed perturbation of time
            },
            'u2': {  #substrate concentration (g.L-1)
                'stps': 5,
                'const': 'rel',
                'max': 35,
                'min': 5,
                'cvp': 'LPF',
                'offl': 1,
                'offt': 0.5
            }
        },
        'tvo': {  # Time-variant output variables (responses, measured or unmeasured)
            'y1': {  # biomass concentration (g.L-1)
                'init': 'variable',  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 10,  # the amound of samples per each round (run)
                'unc': 0.15,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.5,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [],  # fixed sampling times
            },
            'y2': {  # substrate concentration (g.L-1)
                'init': 0.01,
                # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,
                # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 10,  # the amound of samples per each round (run)
                'unc': 0.1,
                # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 0.5,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [],  # fixed sampling times
            },
        },
        'tii': {  # Time-invariant input variables (tii)
            'y1_0': {  # 1st symbolic time-invariant control, Density of solid reactant (kg/m³)
                'max': 5.5,  # Maximum allowable signal level, des_opt space upper bound
                'min': 1  # Minimum allowable signal level, des_opt space upper bound
            },
        },
        'tio': {  # Time-invariant output variables (empty here, could hold steady state responses that hold no dependency)
        },
        't_s': [0, 40],  # Time span  (600 s to 10,800 s), duration of numerical perturbations (the rest is precluded from des_opt)
        't_r': 0.16,  # Time resolution (10 s), minimum time steps for the simulation/des_opt/controls
        't_d': 0
    }


    models = { # Settings related to the rival models and their parameters
        'can_m': ['MI', 'MII', 'MIII', 'MIV'],  # Active solvers (rival models) to be used in the experiment
        'krt': {'MI': model_I, 'MII': model_II, 'MIII': model_III, 'MIV': model_IV,},
        # type of the model interface, 'pym' for middoe.krnl_models, 'gpr' for gPAS models, function name for globally defined functions, 'pys' for python standalone scripts
        'creds': {'f20': '@@TTmnoa698', 'f21': '@@TTmnoa698'},
        # credentials for gPAS models, if not needed, leave empty
        'src': {'f20': 'C:/Users/Tadmin/PycharmProjects/tutorialmid1/model_semiconwet.py',
                'f21': 'C:/Users/Tadmin/PycharmProjects/tutorialmid1/model_semiconwet.py'},
        # for now for gPAS readable files, or python standalone scripts

        'theta': { # Theta parameters for each models
            'MI': theta_In,
            'MII': theta_IIn,
            'MIII': theta_IIIn,
            'MIV': theta_IVn,
        },
        't_u': { # Maximum bounds for theta parameters (based on normalized to'f20': theta20mins, 1)
            'MI': theta_maxs_I,
            'MII': theta_maxs_II,
            'MIII': theta_maxs_III,
            'MIV': theta_maxs_IV,
        },
        't_l': { # Minimum bounds for theta parameters (based on normalized to 1)
            'MI': theta_mins_I,
            'MII': theta_mins_II,
            'MIII': theta_mins_III,
            'MIV': theta_mins_IV,
        }
    }

    from middoe.log_utils import save_to_jac


    insilicos = { # Settings for the insilico data generation
        'tr_m': 'MI', # selected true models (with nominal values)
        'theta': theta_I,
        'errt': 'abs',  # error type, 'rel' for relative error, 'abs' for absolute error
        'prels': { # classic des_opt settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
            '1': {'u1': 0.05, 'u2': 30, 'y_0': 1},
            '2': {'u1': 0.1, 'u2': 30, 'y_0': 1},
        }
    }


    from middoe.krnl_expera import expera
    # expera(system, models, insilicos, design_decisions={}, expr=1)
    #
    #
    #
    # expera(system, models, insilicos, design_decisions={}, expr=2)

    models['can_m'].remove('MIV')
    models['can_m'].remove('MIII')


    iden_opt = { # Settings for the parameter estimation process
        'meth': 'G',  # optimisation method, 'G': Global Differential Evolution, 'Ls': Local SLSQP, 'Ln': Local Nelder-Mead
        'init': None,   # use 'rand' to have random starting point and use None to start from theta_parameters nominal values (to be avoided in insilico studies)
        'eps': 1e-3,  # perturbation size of parameters in SA FDM method (in a normalized to 1 space)
        #usually 1e-3, or None to perform a mesh independency test, and auto adjustment
        'ob': 'WLS',  #loss function, 'LS': least squares, 'MLE': maximum likelihood, 'Chi': chi-square, 'WLS': weighted least squares
        'c_plt': True, # plot the confidence volumes
        'f_plt': True, # plot the fitting results
        'plt_s': True, # show plots while saving
        'log': True # log the results
    }


    from middoe.log_utils import  read_excel
    data = read_excel('indata')


    from middoe.iden_parmest import parmest
    resultpr = parmest(system, models, iden_opt, data)


    from middoe.iden_uncert import uncert
    uncert_results = uncert(data, resultpr, system, models, iden_opt)
    resultun = uncert_results['results']
    theta_parameters = uncert_results['theta_parameters']
    solver_parameters = uncert_results['solver_parameters']
    scaled_params = uncert_results['scaled_params']
    obs = uncert_results['obs']



    from middoe.log_utils import  read_excel, save_rounds
    round_data={}
    round = 1
    save_rounds(round, resultun, theta_parameters, 'preliminary', round_data, models, scaled_params,iden_opt,solver_parameters, obs, data, system)



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



    from middoe.des_md import mbdoe_md
    from middoe.des_pp import mbdoe_pp
    designs = mbdoe_md(des_opt, system, models, round=2, num_parallel_runs=16)


    expera(system, models, insilicos, designs, expr=3, swps=designs['swps'])


    data = read_excel('indata')


    resultpr = parmest(system, models, iden_opt, data)


    uncert_results = uncert(data, resultpr, system, models, iden_opt)
    resultun = uncert_results['results']
    theta_parameters = uncert_results['theta_parameters']
    solver_parameters = uncert_results['solver_parameters']
    scaled_params = uncert_results['scaled_params']
    obs = uncert_results['obs']


    round = 2
    trv=save_rounds(round, resultun, theta_parameters, 'MBDOE_MD', round_data, models, scaled_params, iden_opt,
                solver_parameters, obs, data, system)

    # for solver, solver_results in resultun.items():
    #     if solver_results['P'] > 0.90:
    #         models['can_m']= [solver]
    #         if any(t_value < trv[solver] for t_value in solver_results['t_values']):
    #             print(f"Model {solver} is not valid.")
    #             designs2 = mbdoe_pp(des_opt, system, models, round=3, num_parallel_runs=16)
    #             expera(system, models, insilicos, designs2, expr=4, swps=designs2['swps'])
    #             data = read_excel('indata')
    #             resultpr2 = parmest(system, models, iden_opt, data)
    #             uncert_results2 = uncert(data, resultpr2, system, models, iden_opt)
    #             resultun2 = uncert_results2['results']
    #             theta_parameters2 = uncert_results2['theta_parameters']
    #             solver_parameters2 = uncert_results2['solver_parameters']
    #             scaled_params2 = uncert_results2['scaled_params']
    #             obs2 = uncert_results2['obs']
    #             round = 3
    #             save_rounds(round, resultun2, theta_parameters2, 'MBDOE_PP', round_data, models, scaled_params2,
    #                         iden_opt, solver_parameters2, obs2, data, system)
    #

    save_to_jac(round_data, purpose="iden")


    # from middoe.iden_valida import validation
    # R2_prd, R2_val, parameters = validation(data, system, models, iden_opt,round_data)


if __name__ == "__main__":
    main()

# results = load_from_jac()
# iden = results['iden']
#
# from middoe.iden_utils import run_postprocessing
# run_postprocessing(
#     round_data=results['iden'],
#     solvers=['MI', 'MII'],
#     selected_rounds=[ 1, 2],
#     plot_global_p_and_t=True,
#     plot_confidence_spaces=True,
#     plot_p_and_t_tests=True,
#     export_excel_reports=True,
#     plot_estimability=True
# )

