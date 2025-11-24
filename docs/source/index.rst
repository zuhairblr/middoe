MIDDoE: Model-(based) Identification, Discrimination, and Design of Experiments
===============================================================================

.. image:: https://img.shields.io/pypi/v/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/zuhairblr/middoe.svg
   :target: https://github.com/zuhairblr/middoe/blob/main/LICENSE
   :alt: License

Overview
--------

MIDDoE is an open-source Python package providing an integrated, physics-aware workflow for model identification, discrimination, and optimal experimental design (MBDoE) in dynamic lumped systems. Developed to overcome tool fragmentation in the process industries, MIDDoE unifies sensitivity analysis, estimability analysis, parameter calibration, uncertainty analysis, and both T-optimal and alphabetical MBDoE criteria within a flexible, user-oriented framework.

Key advantages
--------------

- **End-to-end workflow:** Global Sensitivity Analysis, calibration, uncertainty quantification, parameter estimability, MBDoE for model discrimination and parameter precision, and cross-validation.
- **Full constraint support:** Familiar piecewise-constant and piecewise-linear control parameterisations, forced/synchronised sampling, minimum perturbations (level and time), and dead-time exclusion windows.
- **Dual MBDoE strategy:** Implements T-optimal criteria (Hunter–Reiner, Buzzi–Ferraris–Forzatti) and classical D-, A-, E-, ME-optimality metrics.
- **Parameter subset selection:** Rigorous orthogonalisation-based estimability, rCC minimisation, and mutation masks to stabilise ill-conditioned calibration problems.
- **Flexible model kernel:** Accepts built-in models, external Python simulators, or third-party tools (e.g. gPROMS) via a standard interface.

Mathematical framework (symbolic MIMO)
--------------------------------------

MIDDoE operates on the general nonlinear, multiple-input-multiple-output structure:

    M(θ, x(t), u(t), z(t), w)

- θ: model parameters
- x(t): time-variant differential states
- z(t): time-variant algebraic states
- u(t): time-variant controls (CVP-parameterised)
- w: time-invariant controls/design variables
- y(t): measured outputs

All modules exploit local sensitivities, Fisher matrices, variance–covariance structure, and divergence measures, as detailed in the package’s main publication.

Symbolic configuration (parameters and system)
----------------------------------------------

Parameter specification:

.. code-block:: python

   # True parameter vector (for validation/in-silico)
   theta_true = [θ1_true, θ2_true, ..., θN_true]

   # Nominal starting point
   theta_nom = [θ1_nom, θ2_nom, ..., θN_nom]

   # Feasible bounds
   theta_min = [θ1_min, θ2_min, ..., θN_min]
   theta_max = [θ1_max, θ2_max, ..., θN_max]

System configuration:

.. code-block:: python

   system = {
       't_s': [t_start, t_end],   # Experiment duration
       't_r': Δt,                 # Time resolution
       't_d': τ_dead,             # Fraction dead-time start/end

       'tvi': {  # Time-variant inputs
           'u_i': {
               'min': u_i_min,
               'max': u_i_max,
               'stps': N_segments,
               'cvp': 'LPF/CPF',
               'const': 'rel/inc/dec',
               'offl': δ_level,
               'offt': δ_time,
           },
           # ...
       },

       'tvo': {  # Measured outputs
           'y_r': {
               'init': y_r0 or 'variable',
               'meas': True/False,
               'unc': σ_r,
               'sp': N_samples,
               'offt': δ_t_sample,
               'samp_s': sync_id,
               'samp_f': [t_forced_1, ...],
           },
           # ...
       },

       'tii': {  # Time-invariant inputs
           'w_j': {'min': w_j_min, 'max': w_j_max},
           # ...
       },

       'tio': {},  # Time-invariant outputs (optional)
   }

Models configuration:

.. code-block:: python

   models = {
       'can_m': ['M1', ...],
       'krt': {'M1': 'pys', ...},
       'src': {'M1': '/path/to/model.py', ...},
       'creds': {'M1': '', ...},
       'theta': {'M1': theta_nom_M1, ...},
       't_l': {'M1': theta_min_M1, ...},
       't_u': {'M1': theta_max_M1, ...},
       'mutation': {'M1': [True, ...], ...},
   }

Numerical example: Pharmaceutical SC2
-------------------------------------

Parameters:

.. code-block:: python

   theta      = [50000, 75000, 0.4116, 111900, 9905, 30000]   # ground truth
   theta_n    = [100000, 100000, 1, 100000, 100, 10000]       # initial
   theta_mins = [10000, 0, 0.1, 50000, 10, 10000]             # lower bounds
   theta_maxs = [1000000, 200000, 10, 200000, 10000, 200000]  # upper bounds

System:

.. code-block:: python

   system = {
       't_s': [0, 16], 't_r': 0.02, 't_d': 0.3,
       'tvi': {'u1': {
           'stps': 6, 'const': 'inc', 'max': 306.15, 'min': 296.15,
           'cvp': 'LPF', 'offl': 0.01, 'offt': 0.3
       }},
       'tvo': {
           'y1': {'init': 0, 'meas': True, 'sp': 17, 'unc': 0.005,
                   'offt': 0.3, 'samp_s': 1, 'samp_f': [0, 16]},
           'y2': {'init': 0, 'meas': True, 'sp': 17, 'unc': 0.005,
                   'offt': 0.3, 'samp_s': 1, 'samp_f': [0, 16]},
           'y3': {'init': 0, 'meas': True, 'sp': 17, 'unc': 0.005,
                   'offt': 0.3, 'samp_s': 1, 'samp_f': [0, 16]},
       },
       'tii': {
           'y10': {'max': 1, 'min': 0.3},
           'y20': {'max': 1, 'min': 0.19}
       },
       'tio': {}
   }

Model (single candidate, Python kernel):

.. code-block:: python

   models = {
       'can_m': ['M'],
       'krt': {'M': 'pys'},
       'src': {'M': r'C:/Users/Tadmin/PycharmProjects/middoe/tests/paper/sc2/CS2 - SC2/model.py'},
       'creds': {'M': '@@TTmnoa698'},
       'theta': {'M': theta_n},
       't_u': {'M': theta_maxs},
       't_l': {'M': theta_mins},
       'mutation': {'M': [False, False, True, True, True, True]}
   }

In-silico experimenter:

.. code-block:: python

   insilicos = {
       'tr_m': 'M',
       'theta': theta,
       'errt': 'abs',
       'prels': {}
   }

Global Sensitivity Analysis:

.. code-block:: python

   gsa = {
       'var_s': False,
       'par_s': True,
       'var_d': False,
       'par_d': False,
       'samp': 2**12,
       'multi': 0.7,
       'tii_n': [0.508, 0.3925],
       'tvi_n': [301.15],
       'plt': True,
   }
   from middoe.sc_sensa import sensa
   sobol_results = sensa(gsa, models, system)
   from middoe.log_utils import save_to_jac, load_from_jac, save_to_xlsx
   save_to_jac(sobol_results, purpose="sensa")
   results = load_from_jac()
   sensa_data = results['sensa']
   save_to_xlsx(sensa_data)

MBDoE settings:

.. code-block:: python

   des_opt = {
       'eps': 1e-5,
       'md_ob': 'BFF',
       'pp_ob': 'D',
       'plt': True,
       'meth': 'PS',
       'itr': {
           'pps': 100, 'maxmd': 5, 'tolmd': 1, 'maxpp': 100, 'tolpp': 1
       }
   }

Parameter estimation and uncertainty analysis:

.. code-block:: python

   iden_opt = {
       'meth': 'LBFGSB',
       'ms': True,
       'maxit': 500,
       'tol': 0.1,
       'sens_m': 'central',
       'var-cov': 'B',
       'nboot': 50,
       'init': None,
       'eps': 1e-5,
       'ob': 'WLS',
       'c_plt': False,
       'f_plt': True,
       'plt_s': True,
       'log': False
   }

Workflow execution: one round

.. code-block:: python

   from middoe.des_pp import mbdoe_pp
   designs = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=16)

   from middoe.krnl_expera import expera
   expera(system, models, insilicos, designs, expr=1, swps=designs['swps'])

   from middoe.iden_parmest import parmest
   resultpr = parmest(system, models, iden_opt, case='strov')

   from middoe.iden_uncert import uncert
   uncert_results = uncert(resultpr, system, models, iden_opt)
   resultun = uncert_results['results']
   obs = uncert_results['obs']

   from middoe.sc_estima import estima
   round_num = 1
   ranking, k_optimal_value, rCC_values, J_k_values, best_uncert_result = estima(resultun, system, models, iden_opt, round_num)

   from middoe.log_utils import save_rounds
   round_data = {}
   save_rounds(round_num, resultun, 'preliminary', round_data, models, iden_opt, obs, system,
               ranking=ranking, k_optimal_value=k_optimal_value, rCC_values=rCC_values, J_k_values=J_k_values,
               best_uncert_result=best_uncert_result)

Sequential rounds and cross-validation:

.. code-block:: python

   from middoe.iden_valida import validation
   for round_num in range(2, 5):
       # Design, simulate, estimate, analyse, save (see above for pattern)

   validres = validation(system, models, iden_opt, round_data)

   from middoe.log_utils import save_to_jac, load_from_jac
   save_to_jac(round_data, purpose="iden")
   results = load_from_jac()
   iden = results['iden']

   from middoe.iden_utils import run_postprocessing
   run_postprocessing(
       round_data=iden,
       solvers=['M'],
       selected_rounds=[1, 2, 3, 4],
       plot_global_p_and_t=True,
       plot_confidence_spaces=True,
       plot_p_and_t_tests=True,
       export_excel_reports=True,
       plot_estimability=True
   )

Model interface options
-----------------------

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Backend
     - Configuration
     - Typical use
   * - ``'pys'``
     - ``'krt': {'M': 'pys'}, 'src': {'M': 'path/to/model.py'}``
     - External Python script exposing the standard `solve_model` function
   * - ``'pym'``
     - ``'krt': {'M': 'pym'}``
     - Built-in models for rapid prototyping
   * - Function
     - ``'krt': {'M': 'my_global_function'}``
     - Custom function in Python namespace
   * - ``'gpr'``
     - ``'krt': {'M': 'gpr'}, 'creds': {'M': 'user:pass@server'}``
     - gPROMS model via coupling

Time resolution and FDM perturbation
------------------------------------

The time resolution (`t_r`) controls ODE integration accuracy, design mesh, and inner-loop cost. For precise MBDoE and stable sensitivities, use at least 20 points per fastest process time-constant, and check for mesh-independence. FDM perturbation (`eps`) balances truncation and roundoff error—use mesh-independence testing or set to 1e-5 for normalised space.

Citation
--------

If you use MIDDoE in your work, please cite:

   Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
   MIDDoE: An MBDoE Python package for model identification, discrimination,
   and calibration.
   *Digital Chemical Engineering*, 17, 100276.
   https://doi.org/10.1016/j.dche.2025.100276

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
