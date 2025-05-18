

import numpy as np
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
theta_min_I = [x * 0.8 for x in theta_I]  # 20% below nominal
theta_max_I = [x * 1.2 for x in theta_I]  # 20% above nominal
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
theta_min_II = [x * 0.8 for x in theta_II]  # 20% below nominal
theta_max_II = [x * 1.2 for x in theta_II]  # 20% above nominal
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
theta_min_III = [x * 0.8 for x in theta_III]  # 20% below nominal
theta_max_III = [x * 1.2 for x in theta_III]  # 20% above nominal
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
theta_min_IV = [x * 0.8 for x in theta_IV]  # 20% below nominal
theta_max_IV = [x * 1.2 for x in theta_IV]  # 20% above nominal
theta_maxs_IV = [hi / nom for hi, nom in zip(theta_max_IV, theta_IV)]
theta_mins_IV = [lo / nom for lo, nom in zip(theta_min_IV, theta_IV)]



def main():


    system = {
        'tvi': {  # Time-variant input variables (models input: tvi), each key is a symbol nad key in tvi as well
            'u1': {  # Temperature (K)
                'stps': 3,  # Number of switching times in CVPs (vector parametrisation resolution in time dimension):
                # Must be a positive integer > 1. swps-1 is the number of steps
                'const': 'rel',  # Constraint type: relative state of signal levels in CVPs
                # 'rel' (relative) ensures relaxation, 'dec' (decreasing) ensures decreasing signal levels, 'inc' (increasing) ensures increasing signal levels
                'max': 0.2,  # Maximum allowable signal level, des_opt space upper bound
                'min': 0.05,  # Minimum allowable signal level, des_opt space lower bound
                'cvp': 'CPF',  # Design CVP method (CPF - constant profile, LPF - linear profile)
                'offl': 0.01,  # minimum allowed perturbation of signal (ratio)
                'offt': 1  # minimum allowed perturbation of time (ratio)
            },
            'u2': {  # Pressure (bar)
                'stps': 3,
                'const': 'rel',
                'max': 35,
                'min': 5,
                'cvp': 'LPF',
                'offl': 1,
                'offt': 1
            }
        },
        'tvo': {  # Time-variant output variables (responses, measured or unmeasured)
            'y1': {  # response variable, here carbonation efficiency
                'init': 'variable',  # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,  # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 10,  # the amound of samples per each round (run)
                'unc': 0.05,  # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 1,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 40],  # fixed sampling times
            },
            'y2': {  # response variable, here carbonation efficiency
                'init': 0.01,
                # Initial value for the response variable, it can be a value, or 'variable' for case it is a des_opt decision (time-invariant input variable)
                'meas': True,
                # Flag indicating if this variable is directly measurable, if False, it is a virtual output
                'sp': 10,  # the amound of samples per each round (run)
                'unc': 0.05,
                # amount of noise (standard deviation) in the measurement, in case of insilico, this is used for simulating a normal distribution of noise to measurement (only measurement)
                'offt': 1,  # minimum allowed perturbation of sampling times (ratio)
                'samp_s': 1,  # Matching criterion for models prediction and data alignment
                'samp_f': [0, 40],  # fixed sampling times
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
        't_r': 0.08,  # Time resolution (10 s), minimum time steps for the simulation/des_opt/controls
        't_d': 0.2
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
            'MI': theta_I,
            'MII': theta_II,
            'MIII': theta_III,
            'MIV': theta_IV,
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

    gsa = { # Settings for the Global Sensitivity Analysis (gsa)
        'var_s': False,  # Perform sensitivity analysis for variables
        'par_s': True,  # Perform sensitivity analysis for parameters
        'var_d': False, # feasible space for variables, fload ratio: use as multiplier to nominals uniformly (e.g. 1.1), False: use system defined space
        'par_d': False,   # feasible space for parameters, fload ratio: use as multiplier to nominals uniformly(e.g. 1.1), False: use models defined space
        'samp': 2 ** 6,  # Sampling size for gsa, always 2**n
        'multi': None,  # Perform gsa in parallel
        'tii_n': [1], # Nominal values for the time-invariant variables
        'tvi_n': [0.05, 30], # Nominal values for the time-variant variables
        'plt': True,  # Plot the results
    }



    from middoe.sc_sensa import sensa
    sobol_results = sensa(gsa, models, system)


    from middoe.log_utils import save_to_jac
    save_to_jac(sobol_results, purpose="sensa")

    from middoe.log_utils import load_from_jac, save_to_xlsx

    results = load_from_jac()
    sensa = results['sensa']
    save_to_xlsx(sensa)


    insilicos = { # Settings for the insilico data generation
        'tr_m': 'MI', # selected true models (with nominal values)
        'prels': { # classic des_opt settings, sheet name is the round run name, each sheet contains the data for the round, iso space.
            '1': {'u1': 0.05, 'u2': 30, 'y_0': 1},
            '2': {'u1': 0.1, 'u2': 30, 'y_0': 1},
            # '3': {'T': 338.15, 'P': 0.17, 'aps': 350, 'slr': 0.1},
            # '4': {'T': 353.15, 'P': 1, 'rho': 3191, 'cac': 44.93, 'aps': 5.5e-5, 'mld': 36000}
        }
    }

    from middoe.krnl_expera import expera
    expera(system, models, insilicos, design_decisions={}, expr=1)


    expera(system, models, insilicos, design_decisions={}, expr=2)

    iden_opt = { # Settings for the parameter estimation process
        'meth': 'Ls',  # optimisation method, 'G': Global Differential Evolution, 'Ls': Local SLSQP, 'Ln': Local Nelder-Mead
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
        'md_ob': 'HR',     # MD optimality criterion, 'HR': Hunter and Reiner, 'BFF': Buzzi-Ferraris and Forzatti
        'pp_ob': 'E',  # PP optimality criterion, 'D', 'A', 'E', 'ME'
        'plt': True,  # Plot the results
        'itr': {
            'pps': 50, # population size
            'maxmd': 5, # maximum number of MD runs
            'tolmd': 1, # tolerance for MD optimization
            'maxpp':20 ,# maximum number of PP runs
            'tolpp': 1, # tolerance for PP optimization
        }
    }

    models['can_m'].remove('MIV')
    models['can_m'].remove('MII')
    models['can_m'].remove('MIII')

    from middoe.des_pp import mbdoe_pp
    designs = mbdoe_pp(des_opt, system, models, round=2, num_parallel_runs=1)



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
    save_rounds(round, resultun, theta_parameters, 'MBDOE_PP', round_data, models, scaled_params, iden_opt,
                solver_parameters, obs, data, system)


    save_to_jac(round_data, purpose="iden")


if __name__ == "__main__":
    main()