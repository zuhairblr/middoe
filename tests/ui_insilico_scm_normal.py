from middoe.krnl_models import *
from tests.log_insilico import run_framework

def main():

    #reading parameters from embedded models
    theta07, theta07max, theta07min = thetadic07()
    theta08, theta08max, theta08min = thetadic08()
    theta09, theta09max, theta09min = thetadic09()
    theta11, theta11max, theta11min = thetadic11()
    theta12, theta12max, theta12min = thetadic12()
    theta13, theta13max, theta13min = thetadic13()
    theta14, theta14max, theta14min = thetadic14()
    theta15, theta15max, theta15min = thetadic15()
    theta22, theta22max, theta22min = thetadic22()


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
        'ext_func': {},
        # 'active_solvers': ['f09', 'f11'],
        'active_solvers': [ 'f11'],
        'theta_parameters': {
            'f22': theta22,
            'f15': theta15,
            'f14': theta14,
            'f13': theta13,
            'f12': theta12,
            'f11': theta11,
            'f09': theta09,
            'f08': theta08,
            'f07': theta07,
        },
        'bound_max': {
            'f22': theta22max,
            'f15': theta15max,
            'f14': theta14max,
            'f13': theta13max,
            'f12': theta12max,
            'f11': theta11max,
            'f09': theta09max,
            'f08': theta08max,
            'f07': theta07max,
        },
        'bound_min': {
            'f22': theta22min,
            'f15': theta15min,
            'f14': theta14min,
            'f13': theta13min,
            'f12': theta12min,
            'f11': theta11min,
            'f09': theta09min,
            'f08': theta08min,
            'f07': theta07min,
        },
        'selected_rounds': [3, 4, 10]
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

    run_framework(framework_settings, logic_settings, model_structure, design_settings, modelling_settings,
                  simulator_settings, estimation_settings, GSA_settings)


if __name__ == '__main__':
    main()
