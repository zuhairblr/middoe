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

MIDDoE is an open-source Python package that provides an end-to-end, physics-aware framework for model identification workflows centred on Model-Based Design of Experiments (MBDoE) for lumped dynamic systems. [file:2][web:15] It integrates global sensitivity analysis, estimability analysis, parameter estimation, uncertainty analysis, MBDoE for model discrimination and parameter precision, and cross-validation within a single, NumPy-based environment. [file:2][file:3]

MIDDoE treats models as generic MIMO systems and decouples simulation from MBDoE logic, allowing users to work with built-in ODE/DAE models, external Python simulators, or third-party tools such as gPROMS through a standardised kernel interface. [file:2][file:3] The framework is designed to preserve experimental and physical constraints (bounds, CVP structure, sampling budgets, dead-time windows) while solving typically non-convex design problems with sequential MBDoE workflows. [file:2][file:3]

Key advantages
--------------

- Integrated workflow: GSA → preliminary calibration → uncertainty analysis → EA → MBDoE-MD/PP → validation, implemented as modular executive modules on a common system/models configuration. [file:2][file:3]
- Physics-aware design: supports piecewise-constant and piecewise-linear control vector parameterisations, minimum level/time perturbations, forced and synchronised sampling, and dead-time regions in a single design space description. [file:2][file:3]
- Dual strategy MBDoE: implements T-optimal MBDoE for model discrimination (Hunter–Reiner and Buzzi–Ferraris–Forzatti) and alphabetical MBDoE for parameter precision (D, A, E, ME). [file:2][file:3]
- Estimability-driven calibration: orthogonalisation-based estimability analysis and MSE-based subset selection to stabilise ill-conditioned problems and decide which parameters to estimate or fix. [file:2][file:3]
- Flexible model kernel: models can be internal (built-in), external Python functions, stand-alone Python scripts, or gPROMS models accessed via a coupling interface, provided they implement the standard MIDDoE kernel signature. [file:2][file:3]

Mathematical framework (symbolic MIMO)
--------------------------------------

MIDDoE assumes a general nonlinear, multiple-input multiple-output (MIMO) model structure \( M \) with parameters \( \theta \), differential and algebraic states, measured outputs, and time-varying and time-invariant controls. [file:2][file:3] Using the notation in the paper, a candidate model is written as
\( M(\theta, x(t), u(t), z(t), w) \) with: [file:2][file:3]

- \( \theta \in \mathbb{R}^{N} \): vector of model parameters to be estimated. [file:2][file:3]
- \( x(t) \in \mathbb{R}^{N_x} \): time-variant differential states governed by \( f \big(x, u, z, w, \theta \big) \). [file:2][file:3]
- \( z(t) \in \mathbb{R}^{N_z} \): time-variant algebraic states governed by \( g \big(x, u, z, w, \theta \big) \). [file:2][file:3]
- \( u(t) \in \mathbb{R}^{N_u} \): time-dependent controls (manipulated variables) defined via CVPs. [file:2][file:3]
- \( w \in \mathbb{R}^{N_w} \): time-invariant controls (initial conditions, setpoints, design variables). [file:2][file:3]
- \( y(t) \in \mathbb{R}^{N_r} \): measured outputs used for calibration and design, related to predictions \( \hat{y}(t) \) through \( y(t) = \hat{y}(t) + \varepsilon(t) \). [file:2][file:3]

All MIDDoE modules work on this abstract MIMO formulation, using local sensitivities, Fisher information matrices, variance–covariance matrices, and divergence metrics as required by GSA, EA, MBDoE-MD, MBDoE-PP, and validation. [file:2][file:3]

Symbolic configuration: parameters and system
---------------------------------------------

This section presents a symbolic configuration of a generic MIDDoE workflow before instantiating it with a concrete numerical example. [file:2][file:3]

Symbolic parameter vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Define true parameters for in-silico studies, the initial nominal vector used for calibration, and feasible bounds:

.. code-block:: python

   # True parameter values (ground truth for in-silico studies)
   theta_true = [θ1_true, θ2_true, ..., θN_true]

   # Initial guess (nominal parameter vector)
   theta_nom = [θ1_nom, θ2_nom, ..., θN_nom]

   # Feasible bounds
   theta_min = [θ1_min, θ2_min, ..., θN_min]
   theta_max = [θ1_max, θ2_max, ..., θN_max]

In a typical MBDoE workflow, \( \theta_{\text{true}} \) is only used for synthetic data generation and validation, while \( \theta_{\text{nom}} \), \( \theta_{\min} \), and \( \theta_{\max} \) define the normalised estimation and design spaces. [file:2][file:3]

Symbolic system dictionary (MIMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system dictionary encodes the experimental design space, control structure, and measurement set-up for a generic MIMO system:

.. code-block:: python

   system = {
       # Time configuration
       't_s': [t_start, t_end],   # Experiment duration
       't_r': Δt,                 # Time resolution for simulation and FDM
       't_d': τ_dead,             # Relative dead-time at start and end of batch

       # Time-variant inputs u(t)
       'tvi': {
           'u_i': {
               'min': u_i_min,
               'max': u_i_max,
               'stps': N_segments,   # number of CVP switching points
               'cvp': 'CPF or LPF',  # constant or linear profiles
               'const': 'rel/inc/dec',
               'offl': δ_level,      # minimum signal change (relative)
               'offt': δ_time        # minimum switching interval (relative)
           },
           # more inputs ...
       },

       # Time-variant outputs y(t)
       'tvo': {
           'y_r': {
               'init': y_r0 or 'variable',
               'meas': True or False,
               'unc': σ_r,           # measurement noise std
               'sp': N_samples,      # maximum number of samples
               'offt': δ_t_sample,   # minimum time separation
               'samp_s': sync_id,    # synchronisation group
               'samp_f': [t_forced_1, ...]
           },
           # more outputs ...
       },

       # Time-invariant inputs w
       'tii': {
           'w_j': {
               'min': w_j_min,
               'max': w_j_max
           },
           # more w ...
       },

       # Time-invariant outputs (steady values)
       'tio': {}
   }

This structure is shared across GSA, MBDoE, parameter estimation, and validation modules, ensuring that controllable inputs and measurable outputs are defined once and reused consistently. [file:2][file:3]

Symbolic models dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~

The models dictionary defines available candidate models, their interfaces, parameter vectors, bounds, and mutation masks:

.. code-block:: python

   models = {
       'can_m': ['M1', 'M2', ...],   # active candidate models

       # Kernel type: 'pys', 'pym', 'gpr', or function name
       'krt': {
           'M1': 'pys',
           'M2': 'pym',
       },

       # Source for 'pys' or external interfaces
       'src': {
           'M1': '/path/to/model_M1.py',
           'M2': ''
       },

       # Credentials for gPROMS ('gpr') interfaces
       'creds': {
           'M1': '',
           'M2': ''
       },

       # Parameter vectors and bounds (model-specific)
       'theta': {
           'M1': theta_nom_M1,
           'M2': theta_nom_M2
       },
       't_l': {
           'M1': theta_min_M1,
           'M2': theta_min_M2
       },
       't_u': {
           'M1': theta_max_M1,
           'M2': theta_max_M2
       },

       # Mutation mask: True → estimated, False → fixed
       'mutation': {
           'M1': [True, True, ...],
           'M2': [True, False, ...]
       }
   }

The same model configuration is used by GSA, MBDoE-MD, MBDoE-PP, parameter estimation, and validation, so that parameter subsets and bounds are interpreted consistently across the workflow. [file:2][file:3]

Numerical example: CS2-style calibration workflow
-------------------------------------------------

This section instantiates the symbolic workflow with a concrete numerical example inspired by the pharmaceutical case study (CS2) in the paper. [file:2][file:3] The model is a dynamic MIMO system with a single time-variant control, two time-invariant inputs, and three measured outputs. [file:2][file:3]

Parameter vectors
~~~~~~~~~~~~~~~~~

Define the true parameters used for in-silico data, the nominal starting point, and feasible bounds:

.. code-block:: python

   # True parameters (ground truth for in-silico data)
   theta = [50000, 75000, 0.4116, 111900, 9905, 30000]

   # Initial / nominal parameter vector
   theta_n = [100000, 100000, 1, 100000, 100, 10000]

   # Feasible bounds
   theta_mins = [10000, 0, 0.1, 50000, 10, 10000]
   theta_maxs = [1000000, 200000, 10, 200000, 10000, 200000]

In practice, MIDDoE internally normalises the parameter space to \([0, 1]^N\) for numerical stability, while maintaining physical units and scales at the client layer. [file:2][file:3]

System dictionary
~~~~~~~~~~~~~~~~~

Define the experimental design space and measurement configuration:

.. code-block:: python

   system = {
       # Time configuration
       't_s': [0, 16],   # 16-hour experiment
       't_r': 0.02,      # time resolution (hours)
       't_d': 0.3,       # dead-time fraction at start and end

       # Time-variant inputs (controls)
       'tvi': {
           'u1': {            # e.g. Temperature (K)
               'stps': 6,     # 6 switching points → 5 segments
               'const': 'inc',# monotonically increasing profile
               'max': 306.15,
               'min': 296.15,
               'cvp': 'LPF',  # piecewise-linear profile
               'offl': 0.01,  # minimum level change (relative)
               'offt': 0.3    # minimum switching interval (relative)
           },
       },

       # Time-variant outputs (measured responses)
       'tvo': {
           'y1': {
               'init': 0,
               'meas': True,
               'sp': 17,
               'unc': 0.005,
               'offt': 0.3,
               'samp_s': 1,
               'samp_f': [0, 16],
           },
           'y2': {
               'init': 0,
               'meas': True,
               'sp': 17,
               'unc': 0.005,
               'offt': 0.3,
               'samp_s': 1,
               'samp_f': [0, 16],
           },
           'y3': {
               'init': 0,
               'meas': True,
               'sp': 17,
               'unc': 0.005,
               'offt': 0.3,
               'samp_s': 1,
               'samp_f': [0, 16],
           },
       },

       # Time-invariant inputs (design variables)
       'tii': {
           'y10': {
               'max': 1.0,
               'min': 0.3
           },
           'y20': {
               'max': 1.0,
               'min': 0.19
           },
       },

       # Time-invariant outputs (none in this example)
       'tio': {}
   }

Here, \( t_r \) controls both the ODE integration mesh and the density of time points available for finite-difference sensitivities, while \( t_d \) reserves the initial and final fractions of the time span as non-perturbable dead zones for controls or sampling. [file:2][file:3]

Models dictionary
~~~~~~~~~~~~~~~~~

Define a single candidate model using a Python stand-alone script:

.. code-block:: python

   models = {
       # Active candidate model(s)
       'can_m': ['M'],

       # Kernel type: 'pys' for stand-alone Python script
       'krt': {
           'M': 'pys'
       },

       # Path to Python model file implementing the standard interface
       'src': {
           'M': r'C:/Users/Tadmin/PycharmProjects/middoe/tests/paper/sc2/CS2 - SC2/model.py'
       },

       # Credentials (used only for 'gpr' / gPROMS coupling)
       'creds': {
           'M': '@@TTmnoa698'
       },

       # Parameter vectors and bounds
       'theta': {
           'M': theta_n
       },
       't_u': {
           'M': theta_maxs
       },
       't_l': {
           'M': theta_mins
       },

       # Mutation (estimation mask): False → fixed, True → estimated
       'mutation': {
           'M': [False, False, True, True, True, True]
       }
   }

The stand-alone model file must expose a function following the MIDDoE kernel signature, accepting time vector, initial states, time-invariant and time-variant inputs, and parameter vector, and returning trajectories for time-variant and time-invariant outputs. [file:2][file:3]

In-silico experimenter
~~~~~~~~~~~~~~~~~~~~~~

Configure the in-silico experimenter for synthetic data generation and method validation:

.. code-block:: python

   insilicos = {
       'tr_m': 'M',     # true model
       'theta': theta,  # true parameter vector
       'errt': 'abs',   # 'abs' (absolute) or 'rel' (relative) error
       'prels': {       # optional classic designs (empty here)
           # 'design_name': {...}
       }
   }

The true model and parameters define the data-generating process, and the error type determines whether measurement noise scales with signal or is absolute. [file:2][file:3]

Global Sensitivity Analysis (GSA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define the GSA settings, run Sobol-based analysis, and save results:

.. code-block:: python

   from middoe.sc_sensa import sensa
   from middoe.log_utils import save_to_jac, load_from_jac, save_to_xlsx

   gsa = {
       'var_s': False,        # sensitivity w.r.t. controls
       'par_s': True,         # sensitivity w.r.t. parameters
       'var_d': False,        # use system-defined bounds for variables
       'par_d': False,        # use models-defined bounds for parameters
       'samp': 2**12,         # base Sobol sample size
       'multi': 0.7,          # fraction of logical CPU cores for parallelism
       'tii_n': [0.508, 0.3925],  # nominal w
       'tvi_n': [301.15],         # nominal u
       'plt': True
   }

   sobol_results = sensa(gsa, models, system)
   save_to_jac(sobol_results, purpose="sensa")

   results = load_from_jac()
   sensa_data = results['sensa']
   save_to_xlsx(sensa_data)

The GSA module computes time-resolved first- and total-order Sobol indices, with sampling either on full feasible domains or damped nominal-centred regions. [file:2][file:3]

MBDoE design settings
~~~~~~~~~~~~~~~~~~~~~

Configure the MBDoE core for parameter precision and optional model discrimination:

.. code-block:: python

   des_opt = {
       # Finite difference perturbation (normalised parameter space)
       'eps': 1e-5,

       # MBDoE-MD objective (unused in this single-model example, but available)
       'md_ob': 'BFF',   # 'HR' or 'BFF'

       # MBDoE-PP objective
       'pp_ob': 'D',     # 'D', 'A', 'E', 'ME'

       'plt': True,      # plot designs
       'meth': 'PS',     # 'PS', 'DE', or 'DEPS'

       'itr': {
           'pps': 100,   # population size (for DE / PS)
           'maxmd': 5,   # maximum MD iterations
           'tolmd': 1,   # MD tolerance
           'maxpp': 100, # maximum PP iterations
           'tolpp': 1    # PP tolerance
       }
   }

MIDDoE executes the design optimisation in a normalised design space, enforcing signal and timing constraints from the system dictionary as hard constraints in the optimiser. [file:2][file:3]

Parameter estimation and uncertainty settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure estimation and uncertainty analysis:

.. code-block:: python

   iden_opt = {
       'meth': 'LBFGSB',   # 'SLSQP', 'LBFGSB', 'DE', 'NMS', 'BFGS', 'TC'
       'ms': True,         # multi-start local optimisation
       'maxit': 500,
       'tol': 0.1,

       # Sensitivity and FDM
       'sens_m': 'central',  # 'central' or 'forward'
       'eps': 1e-5,          # FDM perturbation (normalised space)

       # Variance–covariance computation
       'var-cov': 'B',       # 'H', 'J', or 'B'
       'nboot': 50,          # bootstrap samples if 'B'

       # Initialisation
       'init': None,         # None → nominal; 'rand' → random starts

       # Objective function
       'ob': 'WLS',          # 'LS', 'WLS', 'MLE', 'CS'

       # Plotting and logging
       'c_plt': False,
       'f_plt': True,
       'plt_s': True,
       'log': False
   }

The uncertainty module adapts the finite-difference perturbation (if requested) via a mesh-independence test and supports asymptotic and bootstrap approaches, relying on sensitivities and Fisher information matrices as described in the paper. [file:2][file:3]

One-round MBDoE-PP and calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design the first MBDoE-PP experiment, generate in-silico data, estimate parameters, quantify uncertainty, and perform estimability analysis:

.. code-block:: python

   from middoe.des_pp import mbdoe_pp
   from middoe.krnl_expera import expera
   from middoe.iden_parmest import parmest
   from middoe.iden_uncert import uncert
   from middoe.sc_estima import estima
   from middoe.log_utils import save_rounds

   # 1. Design MBDoE-PP experiment (round 1)
   designs = mbdoe_pp(
       des_opt,
       system,
       models,
       round=1,
       num_parallel_runs=16
   )

   # 2. Generate in-silico data for the designed experiment
   expera(
       system,
       models,
       insilicos,
       designs,
       expr=1,
       swps=designs['swps']
   )

   # 3. Parameter estimation
   resultpr = parmest(
       system,
       models,
       iden_opt,
       case='strov'
   )

   # 4. Uncertainty analysis
   uncert_results = uncert(
       resultpr,
       system,
       models,
       iden_opt
   )
   resultun = uncert_results['results']
   obs = uncert_results['obs']

   # 5. Estimability analysis (up to round 1)
   round_num = 1
   ranking, k_optimal_value, rCC_values, J_k_values, best_uncert_result = estima(
       resultun,
       system,
       models,
       iden_opt,
       round_num
   )

   # 6. Save round data
   round_data = {}
   save_rounds(
       round_num,
       resultun,
       'preliminary',
       round_data,
       models,
       iden_opt,
       obs,
       system,
       ranking=ranking,
       k_optimal_value=k_optimal_value,
       rCC_values=rCC_values,
       J_k_values=J_k_values,
       best_uncert_result=best_uncert_result
   )

The estimability analysis uses orthogonalisation-based ranking and rCC-based subset selection to determine the optimal number of active parameters for subsequent rounds. [file:2][file:3]

Sequential rounds and cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the workflow over multiple MBDoE rounds and perform cross-validation and post-processing:

.. code-block:: python

   from middoe.iden_valida import validation
   from middoe.log_utils import save_to_jac, load_from_jac
   from middoe.iden_utils import run_postprocessing

   # 7. Sequential rounds (r = 2, 3, ...)
   for round_num in range(2, 5):
       # Design new experiment
       design = mbdoe_pp(
           des_opt,
           system,
           models,
           round_num,
           num_parallel_runs=16
       )

       # Generate in-silico data
       expera(
           system,
           models,
           insilicos,
           design,
           expr=round_num,
           swps=design['swps']
       )

       # Re-estimate parameters and uncertainty
       pe_res = parmest(system, models, iden_opt, case=f'round{round_num}')
       unc_res = uncert(pe_res, system, models, iden_opt)

       # Estimability
       rank, k_opt, rCC, J_k, best = estima(
           unc_res['results'],
           system,
           models,
           iden_opt,
           round_num
       )

       # Save this round
       save_rounds(
           round_num,
           unc_res['results'],
           f'round{round_num}',
           round_data,
           models,
           iden_opt,
           unc_res['obs'],
           system,
           ranking=rank,
           k_optimal_value=k_opt,
           rCC_values=rCC,
           J_k_values=J_k,
           best_uncert_result=best
       )

   # 8. Cross-validation over all rounds
   validres = validation(system, models, iden_opt, round_data)

   # 9. Save and reload full identification workflow
   save_to_jac(round_data, purpose="iden")
   results = load_from_jac()
   iden = results['iden']

   # 10. Post-processing and reporting
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

The validation and post-processing modules provide \( R^2 \) metrics, confidence volumes, parameter trajectories, and evolution of estimability and t-values across rounds, mirroring the analyses reported in the case studies. [file:2][file:3]

Model interface options
-----------------------

MIDDoE supports several model back-ends through a unified kernel API:

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Backend
     - Configuration
     - Typical use
   * - ``'pys'``
     - ``'krt': {'M': 'pys'}, 'src': {'M': 'path/to/model.py'}``
     - External Python script exposing the standard solve function.
   * - ``'pym'``
     - ``'krt': {'M': 'pym'}``
     - Built-in models in ``middoe.krnl_models`` for quick testing.
   * - Function
     - ``'krt': {'M': 'my_global_function'}``
     - Globally defined Python function in the current namespace.
   * - ``'gpr'``
     - ``'krt': {'M': 'gpr'}, 'creds': {'M': 'user:pass@server'}``
     - gPROMS model accessed via pygpas-based coupling.

Regardless of back-end, the model must accept a time grid, initial states, time-invariant and time-variant input dictionaries, and parameter vector, and return trajectories for time-variant and time-invariant outputs in a standard dictionary format. [file:2][file:3]

Time resolution and FDM perturbation
------------------------------------

The time resolution \( t_r \) affects ODE integration accuracy, sensitivity matrix conditioning, and computational effort, since a smaller \( t_r \) increases the number of time nodes and hence the cost of simulation, FDM, and MBDoE inner loops. [file:2][file:3] As a rule of thumb, choose \( t_r \) such that the fastest characteristic time is sampled by at least 20 points and verify mesh independence by checking that estimation and design metrics stabilise when \( t_r \) is further reduced. [file:2][file:3]

The finite-difference perturbation \( \varepsilon \) used in ``des_opt['eps']`` and ``iden_opt['eps']`` trades off truncation error against round-off and nonlinearity; values that are too small induce numerical noise, while large values distort local derivative approximations in nonlinear regions. [file:2][file:3] MIDDoE can either use a user-specified step in the normalised parameter space or perform an internal mesh-independence test to select a perturbation in the plateau region of eigenvalues of the variance–covariance matrix, which is particularly important in sloppy or ill-conditioned problems. [file:2][file:3]

Citation
--------

If you use MIDDoE in your work, please cite:

   Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
   MIDDoE: An MBDoE Python package for model identification, discrimination, and calibration.
   Computers & Chemical Engineering. [file:2][web:21]

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
