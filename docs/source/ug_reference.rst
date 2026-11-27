Reference
=========

.. include:: _ribbon.rst

Every file format, dictionary key and equation. For a guided start see :doc:`user_guide`.

.. contents:: On this page
   :local:
   :depth: 1

.. _project-structure:

Project structure
-----------------

Organise your project as follows:

.. code-block:: text

   my_project/
   ├── data.xlsx                           # Your experimental measurements
   │
   ├── model1.py                           # Python file defining your model 1
   ├── model2.py                           # Python file defining your model 2
   ├── modeli.py                           # Python file defining your model i
   │
   └── workflow.py                         # analysis and identification workflow

.. _data-file:

Data file (data.xlsx)
---------------------

MIDDoE reads data from a data.xlsx file in the project repository. Each sheet in ``data.xlsx`` represents one batch/experiment and contains measured outputs (MES), time-invariant controls :math:`\mathbf{w}`, and time-variant controls :math:`\mathbf{u}(t)` in a single table.

.. list-table::
   :widths: 10 15 15 10 15 15 10 10 15
   :header-rows: 1

   * - MES_X:y1
     - MES_Y:y1
     - MES_E:y1
     - MES_X:y2
     - MES_Y:y2
     - MES_E:y2
     - X:all
     - w1
     - u1
   * - :math:`t_{\mathrm{sp},1}^{(y_1)}`
     - :math:`y_1\!\big(t_{\mathrm{sp},1}^{(y_1)}\big)`
     - :math:`\sigma_{y_1}`
     - :math:`t_{\mathrm{sp},1}^{(y_2)}`
     - :math:`y_2\!\big(t_{\mathrm{sp},1}^{(y_2)}\big)`
     - :math:`\sigma_{y_2}`
     - :math:`t_{\mathrm{all},1}`
     - :math:`w_1`
     - :math:`u_1\!\big(t_{\mathrm{all},1}\big)`
   * - :math:`t_{\mathrm{sp},2}^{(y_1)}`
     - :math:`y_1\!\big(t_{\mathrm{sp},2}^{(y_1)}\big)`
     - :math:`\sigma_{y_1}`
     - :math:`t_{\mathrm{sp},2}^{(y_2)}`
     - :math:`y_2\!\big(t_{\mathrm{sp},2}^{(y_2)}\big)`
     - :math:`\sigma_{y_2}`
     - :math:`t_{\mathrm{all},2}`
     - :math:`w_1`
     - :math:`u_1\!\big(t_{\mathrm{all},2}\big)`
   * - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     - :math:`w_1`
     - :math:`\vdots`

- **``MES_X:y1``, ``MES_X:y2``, ..., ``MES_X:yi``:**
  Sampling times :math:`t_{\mathrm{sp}}` used for measurements of :math:`y_1`, :math:`y_2`, ..., :math:`y_i` (system measured response sampled times).

- **``MES_Y:y1``, ``MES_Y:y2``, ..., ``MES_Y:yi``:**
  Measured responses :math:`y_r(t_{\mathrm{sp}})` for measurements of :math:`y_1`, :math:`y_2`, ..., :math:`y_i` (system measured responses), column names must match ``system['tvo']`` keys.

- **``MES_E:y1``, ``MES_E:y2``, ..., ``MES_E:yi``:**
  Measurement standard deviations :math:`\sigma_{y_r}`.

- **``X:all``:**
  Global time grid :math:`t_{\mathrm{all}}` on which the time-variant inputs :math:`u_i(t)` are tabulated (should match the solver precision, stepped by ``system['t_r']``).

- **``w1``, ``w2``, … :**
  Time-invariant controls :math:`w_j`, constant over the experiment; column names must match ``system['tii']`` keys.

- **``u1``, ``u2``, … :**
  Time-variant control profiles :math:`u_i(t_{\mathrm{all}})`; column names must match ``system['tvi']`` keys so that MIDDoE can reconstruct the control trajectories from ``X:all`` and ``u_i``.

**Data file contract.** Every sheet of ``data.xlsx`` is read and treated as one experiment. The columns are interpreted as follows (``expera()`` always writes files that satisfy this contract):

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Contract
   * - ``X:all``
     - Simulation time grid (physical units). It is sorted and de-duplicated (``np.unique``). Every ``MES_X`` time must lie on this grid.
   * - ``MES_X:<y>``, ``MES_Y:<y>``, ``MES_E:<y>``
     - One triple per measured output ``<y>`` (a key of ``system['tvo']``). Recommended: sampling times sorted, unique and on the ``X:all`` grid (not enforced). ``MES_E`` is the **absolute** standard deviation of each measurement and must be > 0. If ``MES_E`` is missing, the output's ``system['tvo'][<y>]['unc']`` is used.
   * - ``<tii>``
     - One column per ``system['tii']`` key, required and constant: only row 0 is used, and a missing, empty or varying column raises ``ValueError``. **dtype caveat:** a column holding only integers reaches the model as ``int64``; a column with a value in row 0 only (blank cells below) arrives as ``float64``.
   * - ``CVP:<tvi>``
     - Required for every ``system['tvi']`` key: ``'CPF'``, ``'LPF'`` or ``'no_CVP'`` (case-sensitive).
   * - ``<tvi>t``, ``<tvi>l``
     - Switching times and switching levels, in **physical** units. Required for ``CPF``/``LPF``.
   * - ``<tvi>``
     - The tabulated profile on ``X:all``. With ``no_CVP`` only its first value is used (constant input).
   * - ``<y>_0``
     - Initial value of output ``<y>`` when ``system['tvo'][<y>]['init'] = 'variable'``. ``<y>_0`` must be a key of ``system['tii']`` (a column alone is not enough).

``parmest()`` and ``uncert()`` read ``./data.xlsx`` from the current working directory by default. Both accept an optional ``data`` argument (``parmest(system, models, iden_opt, case=None, data=None)``, ``uncert(resultpr, system, models, iden_opt, case=None, data=None)``): pass a dict ``{sheet_name: pandas.DataFrame}``, as returned by ``middoe.log_utils.read_excel()``, to use data held in memory instead of the file.

.. _model-file:

Model file (model.py)
---------------------

MIDDoE can couple to models in several ways, selected per model with ``models['krt'][name]``:

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - **Type**
     - **Configuration**
     - **When to use**
   * - External Python script
     - ``'krt': {'M': 'pys'}, 'src': {'M': 'models/my_model.py'}``
     - You have your own ODE solver or external simulator (suggested; also the way to interface external tools). The file must define ``solve_model`` (contract below).
   * - Built-in models
     - ``'krt': {'f05': 'pym'}``
     - For testing. The model name (the key in ``can_m``/``krt``) must equal the name of a function in ``middoe.krnl_models`` (e.g. ``'f05'``); see the API reference.
   * - Python function (callable)
     - ``'krt': {'M': my_rhs}`` (the function object itself, not its name as a string)
     - A right-hand-side function ``f(t, y, phi, phit, theta, te)`` returning ``dy/dt``. MIDDoE integrates it with ``scipy.integrate.solve_ivp`` (LSODA) and names the outputs ``y1``, ``y2``, …, ``yN`` in state order, so the ``system['tvo']`` keys must be ``y1``…``yN``.
   * - gPROMS coupling
     - ``'krt': {'M': 'gpr'}, 'src': {'M': 'model.gPAS'}, 'creds': {'M': 'user:pass@server'}``
     - gPAS file created by gPROMS, to couple with gPROMS models (requires ``pygpas``).

Model paths in ``models['src']`` are resolved relative to the current working directory (or give an absolute path).

**Model file contract ('pys').** MIDDoE treats the model as a black box. The file must define a function ``solve_model`` that MIDDoE calls **positionally** as ``solve_model(t, y0, tii, tvi, theta)``:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Argument
     - Contract
   * - ``t``
     - Physical time points (the ``X:all`` grid).
   * - ``y0``
     - A **list** of initial values in ``system['tvo']`` order, including outputs with ``meas: False``. A numeric ``init`` is used as is; with ``init: 'variable'`` the value is taken from ``tii['<y>_0']``.
   * - ``tii``
     - Dict of time-invariant inputs (physical units, numpy scalars). From a data file the dtype is ``int64`` if the whole column holds integers, otherwise ``float64``.
   * - ``tvi``
     - Dict of time-variant inputs, each an array already sampled on ``t`` (physical units).
   * - ``theta``
     - Parameter values in physical units, in ``models['theta'][name]`` order.
   * - return value
     - ``{'tvo': {name: array of len(t), ...}, 'tio': {name: value, ...}}``. Every **measured** ``system['tvo']`` output must be present with ``len(t)`` finite values; unmeasured (``meas: False``) outputs may be omitted; ``'tio'`` may be empty.

If the model raises, has the wrong signature, or returns a result that breaks this contract, MIDDoE raises ``middoe.krnl_simula.SimulationError`` (the original exception, if any, is chained) instead of continuing with an empty result. A model file that cannot be loaded (wrong path, syntax error, no ``solve_model``) raises ``KeyError``.

.. code-block:: python

   import numpy as np
   from scipy.integrate import solve_ivp

   def solve_model(t, y0, tii, tvi, theta):
       """MIDDoE model interface ('pys')."""
       k1, k2, k3 = theta
       t = np.asarray(t, dtype=float)
       u1 = np.asarray(tvi['u1'], dtype=float)     # sampled on t

       def rhs(tt, y):
           u = np.interp(tt, t, u1)                 # control value at time tt
           ca, cb = y
           return [-k1 * ca * u + k2 * tii['w1'],
                   k1 * ca * u - k3 * cb]

       sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method='LSODA')
       return {
           'tvo': {'y1': sol.y[0], 'y2': sol.y[1]},  # keys = system['tvo'] keys
           'tio': {},
       }

.. _config-dicts:

Configuration Dictionaries
--------------------------


**system Dictionary Definition**

The **system** dictionary is a central hub for all MIDDoE operations. It specifies the system characteristics.

.. code-block:: python

   system = {
       # ========== TIME CONFIGURATION ==========
       't_s': [0, 16],        # Time span: [start, end] (physical time units)
       't_r': 0.02,           # Time resolution: every 0.02 time units
                              # Smaller → better accuracy but slower (must match the X:all grid in data.xlsx)
       't_d': 0.3,            # Dead time (absolute, same units as t_s): no switching or free sampling
                              # in the first 0.3 and the last 0.3 time units (batch start-up/cool-down)

       # ========== MEASUREMENT ERROR TYPE (used by MBDoE) ==========
       'errt': 'abs',         # 'abs' (default): tvo 'unc' is an absolute std dev
                              # 'rel': tvo 'unc' is relative, sigma = max(unc * |y_hat|, sigma_floor)
       'sigma_floor': 1e-6,   # Lower bound on the design sigma with errt='rel' (default 1e-6)
                              # can be overridden per response with system['tvo'][y]['errt']

       # ========== TIME-VARIANT INPUTS: u(t) ==========
       'tvi': {
           'u1': {  # Example: Temperature profile
               'min': 296.15,          # Lower bound (K)
               'max': 306.15,          # Upper bound (K)
               'stps': 6,              # CVP steps: stps=6 → 6 levels and 5 switching times
               'cvp': 'LPF',           # Control profile type:
                                       # 'CPF' = constant piecewise (steps)
                                       # 'LPF' = linear piecewise (ramps)
                                       # 'no_CVP' = constant input
               'const': 'inc',         # Constraint on signal shape:
                                       # 'rel' = no constraint (relaxed)
                                       # 'inc' = monotonically increasing
                                       # 'dec' = monotonically decreasing
               'offl': 0.01,           # Min level change between steps (absolute, same units as min/max: 0.01 K)
               'offt': 0.3             # Min time between switches (absolute, same units as t_s: 0.3)
           },
       },

       # ========== TIME-VARIANT OUTPUTS: y(t) ==========
       'tvo': {
           'y1': {  # Example: Product concentration
               'init': 0,              # Initial value: a number, or 'variable'
                                       # 'variable' = taken from the tii input named 'y1_0'
                                       # (which must then be a key of system['tii'])
               'meas': True,           # Measurable? (True) or virtual output (False)
               'sp': 17,               # Sampling points per batch (including the forced ones in samp_f)
               'unc': 0.005,           # Measurement std dev (absolute if errt='abs', relative if 'rel')
               'offt': 0.3,            # Min interval between sampling times (absolute, same units as t_s)
               'samp_s': 1,            # Sampling group id (integer): outputs with the same id share sampling times
               'samp_f': [0, 16]       # Forced sampling times: always sample at t=0 and t=16
           },
           'y2': {
               'init': 0,
               'meas': True,
               'sp': 17,
               'unc': 0.005,
               'offt': 0.3,
               'samp_s': 1,            # Same group as y1 → synchronized sampling
               'samp_f': [0, 16]
           },
           'y3': {
               'init': 0,
               'meas': True,
               'sp': 17,
               'unc': 0.005,
               'offt': 0.3,
               'samp_s': 1,
               'samp_f': [0, 16]
           },
       },

       # ========== TIME-INVARIANT INPUTS: w ==========
       'tii': {
           'y10': {  # Example: initial concentration of reactant (read by the model itself)
               'min': 0.3,
               'max': 1.0
           },
           'y20': {  # Example: initial concentration of catalyst (read by the model itself)
               'min': 0.19,
               'max': 1.0
           },
       },

       # ========== TIME-INVARIANT OUTPUTS ==========
       'tio': {}  # Empty for this example; used for steady-state responses
   }

Notes on ``system``:

- ``t_d``, ``offt`` (in ``tvi`` and ``tvo``) and ``offl`` are **absolute** values in the units of ``t_s`` and of the input, respectively, not fractions.
- A ``tii`` entry such as ``y10`` above is an ordinary design variable that the model file reads itself (``tii['y10']``) and uses as it likes, e.g. as an initial condition; the ``tvo`` ``init`` is then numeric. The ``<y>_0`` naming is required **only** with ``init: 'variable'``: then MIDDoE takes the initial value of output ``<y>`` from ``tii['<y>_0']``.
- ``unc`` is needed for every measured output by ``expera`` (noise) and by the designs (a design without it assumes σ = 1 in the output's units, with a warning). Estimation reads σ from the ``MES_E`` columns and uses ``unc`` only where a column or cell is empty.
- ``errt`` (system level, or per response in ``system['tvo'][y]['errt']``) tells MBDoE-PP/MD how to interpret ``unc``: ``'abs'`` (default) uses ``unc`` as an absolute standard deviation; ``'rel'`` uses :math:`\sigma = \max(\mathrm{unc}\cdot|\hat{y}|, \sigma_{\mathrm{floor}})` with ``system['sigma_floor']`` (default ``1e-6``). It is independent of ``insilicos['errt']``, which only controls the noise added to in-silico data.

**Model Dictionary Definition**

The **models** dictionary is a central hub for all MIDDoE operations. It specifies the model characteristics.

.. code-block:: python

   # Parameter vectors for in-silico validation (true values, nominal, and bounds)
   theta      = [50000, 75000, 0.4116, 111900, 9905, 30000]   # Ground truth (insilico studies)
   theta_n    = [100000, 100000, 1, 100000, 100, 10000]       # Initial guess
   theta_mins = [10000, 0, 0.1, 50000, 10, 10000]             # Lower bounds
   theta_maxs = [1000000, 200000, 10, 200000, 10000, 200000]  # Upper bounds

   models = {
       # ========== ACTIVE CANDIDATE MODELS ==========
       'can_m': ['M'],  # List of model names to use
                        # Example with multiple models: ['M1', 'M2', 'M3'] in case of need for model discrimination

       # ========== MODEL INTERFACE TYPE ==========
       'krt': {
           'M': 'pys'  # 'pys' = external Python script (file in 'src' defining solve_model)
                       # 'pym' = built-in model; the model name must be a function in middoe.krnl_models
                       # 'gpr' = gPROMS coupling
                       # my_rhs = a callable f(t, y, phi, phit, theta, te) (function object, not a string)
       },

       # ========== MODEL SOURCE PATH ==========
       'src': {
           'M': r'C:/.../model.py'
               # Only needed for 'pys' and 'gpr' type
               # Use raw string (r'...') for Windows paths
       },

       # ========== CREDENTIALS FOR gPROMS ==========
       'creds': {
           'M': ''  # Empty for non-gPROMS models
       },

       # ========== PARAMETER VECTORS AND BOUNDS ==========
       'theta': {  # Starting/nominal values
           'M': theta_n
       },
       't_l': {    # Lower bounds
           'M': theta_mins
       },
       't_u': {    # Upper bounds
           'M': theta_maxs
       },

       # ========== ESTIMATION MASK ==========
       'mutation': {
           'M': [False, False, True, True, True, True]
               # False = fix at nominal value (don't estimate)
               # True = estimate from data
               # Use False for poorly identifiable parameters identified by EA
       }
   }

**insilicos Dictionary Definition**

The **insilicos** dictionary is a settings dictionary to perform insilico experiment data generation.


.. code-block:: python

   # In-silico experimenter (used for synthetic data generation)
   insilicos = {
       'tr_m': 'M',          # True model (ground truth)
       'theta': theta,       # True parameter vector
       'errt': 'abs',        # Noise type for the generated data:
                             # 'abs' = absolute: sigma = unc
                             # 'rel' = relative: sigma = max(unc * |y|, sigma_floor)
       'sigma_floor': 1e-6,  # Lower bound on sigma with errt='rel' (default 1e-6),
                             # so that y = 0 never gives sigma = 0 in MES_E
       'plt': True,          # Small figure of each new data set and its noise (default True),
                             # saved in ./data_plots/experiment_<expr>.png; False = no figure
       'prels': {            # Preliminary (classic, iso-valued) designs, keyed by experiment number
           '1': {'u1': 301.15, 'y10': 0.5, 'y20': 0.4},
       }                     # inputs missing from an entry are set to their 'max'
   }

``insilicos['prels']`` is **always required**. ``expera(system, models, insilicos, design_decisions, expr, swps=None)`` uses ``prels[str(expr)]`` whenever ``design_decisions`` is empty (e.g. ``{}``), i.e. for preliminary experiments; when a design from ``mbdoe_pp``/``mbdoe_md`` is passed, the design is used instead. In both cases the measurement noise uses ``system['tvo'][y]['unc']`` with the ``insilicos['errt']`` interpretation, and ``MES_E`` is written as the absolute σ of each point.

After writing the sheet, ``expera`` draws a small preview (``insilicos['plt']``, default ``True``): for every measured output the noise-free model with the noisy samples (±2σ) and, below it, the noise itself (data − true value) inside its ±2σ band. It is saved in ``./data_plots/experiment_<expr>.png`` and displayed when matplotlib can show figures (a window or a notebook).


**GSA Dictionary Definition**

The **gsa** dictionary is a settings dictionary to perform Global Sensitivity Analysis with Sobol method.

.. code-block:: python

   # Global Sensitivity Analysis configuration
   gsa = {
       'var_s': False,           # Sensitivity w.r.t. controls? (u, w)
       'par_s': True,            # Sensitivity w.r.t. parameters? (θ)
       'var_d': False,           # Control sampling space:
                                 # False = system-defined bounds (min/max)
                                 # float > 1 (e.g. 1.1) = [nominal / var_d, nominal * var_d]
       'par_d': False,           # Parameter sampling space:
                                 # False = models-defined bounds (t_l/t_u)
                                 # float > 1 = [theta / par_d, theta * par_d]
       'samp': 2**12,            # Sobol sample size: use powers of 2
                                 # 2^12 = 4096 base samples
                                 # Total evaluations = (2·N_vars + 2) × base
       'multi': 0.7,             # Parallel execution: 0.7 = use 70% of CPU cores (False = single core)
       'tii_n': [0.508, 0.3925], # Nominal w values (system['tii'] order)
       'tvi_n': [301.15],        # Nominal u values (system['tvi'] order)
       'plt': True               # Plot the Sobol indices (gsa['plt'])
   }

**MBDoE Dictionary Definition**

The **des_opt** dictionary is a settings dictionary to conduct MBDoE-MD and MBDoE-PP optimisations.


.. code-block:: python

   des_opt = {
       # ========== FINITE DIFFERENCE SETUP ==========
       'eps': 1e-5,   # FDM perturbation size (normalised space); required
                      # Smaller → better accuracy, larger → robustness
                      # Typical: 1e-5 to 1e-3. None is not supported here and raises an error
                      # (the automatic eps search exists only in iden_opt)

       # ========== MBDoE-MD OBJECTIVE (Model Discrimination) ==========
       'md_ob': 'BFF', # T-optimality criterion:
                       # 'HR' = Hunter–Reiner (focuses on divergence)
                       # 'BFF' = Buzzi–Ferraris–Forzatti (includes error weighting)

       # ========== MBDoE-PP OBJECTIVE (Parameter Precision) ==========
       'pp_ob': 'D',   # Alphabetical optimality criterion:
                       # 'D' = D-optimality (minimize det of covariance volume)
                       # 'A' = A-optimality (minimize average variance)
                       # 'E' = E-optimality (minimize maximum variance)
                       # 'ME' = Modified E (promote uniform precision)

       # ========== PRIOR INFORMATION (MBDoE-PP) ==========
       'priors': False, # False (default) = FIM without prior term (Sigma_theta^-1 ≈ 0, paper eq. 4)
                        # True = add the inverse of models['V_matrix'] (from uncert) as prior information

       # ========== CONVERGENCE REPORT (MBDoE-PP) ==========
       'conv_rep': False, # False (default) = no per-iteration report
                          # True = record FIM metrics and a curvature probe after every optimiser
                          #   iteration and export them to Excel (about doubles the design time)

       # ========== OPTIMISATION METHOD ==========
       'meth': 'PS',   # Solver type (default 'PS'):
                       # 'PS' = Pattern Search (local, derivative-free)
                       # 'DE' = Differential Evolution (global)
                       # 'DEPS' = DE + PS (hybrid: global then local)

       # ========== ITERATION SETTINGS ==========
       'itr': {
           'pps': 100,     # Population size of 'DE' (not used by 'PS')
           'maxmd': 5,     # Max generations/iterations for MD ('DEPS': DE maxmd//2, then PS maxmd)
           'tolmd': 1,     # Required, but currently without effect (see notes)
           'maxpp': 100,   # Max generations/iterations for PP (as maxmd)
           'tolpp': 1      # Required, but currently without effect (see notes)
       },

       'plt': True     # Plot the designed experiment (des_opt['plt'])
   }

Notes on ``des_opt``:

- **Option aliases.** For backward compatibility with the MIDDoE paper and older scripts, ``meth`` also accepts ``'L'`` (→ ``'PS'``), ``'G'`` (→ ``'DE'``) and ``'GL'`` (→ ``'DEPS'``).
- **Priors.** ``priors`` defaults to ``False``, which matches eq. 4 of the paper (prior term neglected). ``models['V_matrix']`` is stored by ``uncert()`` in physical parameter units; with ``priors=True`` it is converted internally to the normalised (θ/θc) units of the design FIM before it is inverted and added. If no ``V_matrix`` is available, no prior is added. The code used for the MIDDoE paper always added this prior; set ``priors=True`` to reproduce those designs.
- **Measurement error.** The design FIM and the BFF criterion use ``system['tvo'][y]['unc']`` interpreted according to ``system['errt']`` / ``system['tvo'][y]['errt']`` (see the ``system`` notes above).
- **Iteration keys.** ``tolpp``/``tolmd`` are handed to pymoo as a constraint tolerance that pymoo
  0.6 does not use, so they have no effect; the optimisers stop after ``maxpp``/``maxmd``
  generations (or iterations of pattern search).
- **Design space.** Every ``tvi``/``tii`` input needs ``min <= max`` and ``max > 0`` (the design
  works on values divided by ``max``).
- **Plots.** Every core saves its design figure in ``./design/`` (one panel per output with the sampling points on the predicted curve, one per input with its switching times). With ``des_opt['plt'] = True`` the figure of the selected (best) core is also displayed once the design is finished.

**Calibration Dictionary Definition**

The **iden_opt** dictionary is a settings dictionary to conduct parameter estimation and uncertainty analysis.

.. code-block:: python

   iden_opt = {
       # ========== OPTIMISATION SOLVER ==========
       'meth': 'LMBFGS',   # Parameter estimation method:
                           # 'LMBFGS' = Limited-memory BFGS with bounds (L-BFGS-B; local, fast)
                           # 'SLSQP' = Sequential Least Squares (local, constrained)
                           # 'DE' = Differential Evolution (global, slow)
                           # 'NMS' = Nelder–Mead Simplex (local, derivative-free)
                           # 'BFGS' = Standard BFGS (local)
                           # 'TC' = Trust-region Constrained (local, smooth)

       'ms': True,         # Multi-start? (True = try multiple random starts)
       'maxit': 500,       # Maximum iterations per solver attempt
       'tol': 1e-8,        # Convergence tolerance (relative to the objective; see notes)

       # ========== FINITE DIFFERENCE METHOD ==========
       'sens_m': 'central',# FDM scheme:
                           # 'central' = central differences (2nd order, more accurate)
                           # 'forward' = forward differences (1st order)
       'eps': 1e-5,        # FDM perturbation, relative (normalised theta/theta_c units)
                           # None (default) = automatic step test: 50 steps, figure and
                           # Excel in ./meshindep_plots/; uncert() then stores the chosen
                           # step in iden_opt['eps'] (a dict per model), reused later

       # ========== OBJECTIVE FUNCTION ==========
       'ob': 'WLS',        # Cost function:
                           # 'LS'  = Least Squares
                           # 'WLS' = Weighted LS (weights from the measurement sigma)
                           # 'MLE' = Maximum Likelihood (Gaussian, per-point sigma)
                           # 'Chi' = relative least squares, sum(((y - y_hat) / max(|y|, sigma))^2)

       # ========== UNCERTAINTY QUANTIFICATION ==========
       'var-cov': 'B',     # Variance-covariance method:
                           # 'H' = Hessian-based (default)
                           # 'M' = FIM/Jacobian-based, V = (Q^T Sigma^-1 Q)^-1 (fast, linearised)
                           # 'B' = Bootstrap (robust, slow)
       'nboot': 50,        # Bootstrap samples (if var-cov='B')

       # ========== VISUALISATION (used by save_rounds) ==========
       'c_plt': False,     # 95 % confidence-region plots per round (./confidence/)
       'f_plt': True,      # fit, residual and parity plots per round (./modelling/)
       'plt_s': True,      # also display the figures (False: only save them)
       'log': False        # Verbose logging?
   }

Notes on ``iden_opt``:

- **Option aliases.** For backward compatibility with the MIDDoE paper (Table S3) and older scripts, these names are accepted and mapped to the canonical keys: ``meth`` ``'G'`` → ``'DE'``, ``'Ls'`` → ``'SLSQP'``, ``'Ln'`` → ``'NMS'``, ``'SQP'`` → ``'TC'``, ``'NM'`` → ``'NMS'``, ``'LBFGSB'`` → ``'LMBFGS'``; ``ob`` ``'CS'`` → ``'Chi'``; ``var-cov`` ``'J'`` → ``'M'``. Any other unknown value raises a ``ValueError`` listing the valid options.
- **Objective.** ``'Chi'`` is a *relative* least-squares objective, :math:`\sum_r \sum_t \big((y_{r,t} - \hat{y}_{r,t})/\max(|y_{r,t}|, \sigma_{r,t})\big)^2`. The denominator is floored at the point's σ (``MES_E``), so a value measured near zero gets the weight :math:`1/\sigma^2` instead of an unbounded one. Use it for relative errors; with absolute errors use WLS. WLS and MLE use the per-point σ from the ``MES_E`` columns (or ``unc`` when ``MES_E`` is missing).
- **Required keys.** ``meth`` and ``ob`` have no default; a missing one raises ``ValueError``. ``var-cov`` defaults to ``'H'``, ``tol`` to 1e-6, ``maxit`` to 1000, ``ms`` to ``False``, ``eps`` to ``None``.
- **Initialisation.** Without multi-start the optimisation starts from ``models['theta']`` (or from ``models['thetastart']``, the original starting guess, with ``case='strov'``). ``ms=True`` replaces that start by random starts drawn uniformly inside ``t_l``/``t_u`` (about 0.7 × the number of CPU cores of them, run in parallel) and keeps the best; fixed parameters keep their values. The former ``'init'`` key (``None``/``'rand'``) currently has no effect and can be omitted.
- **Optimiser numerics (advanced keys, defaults recommended).**

  - ``'pscale'`` (default ``'log'``): the optimiser works on :math:`\log|\theta/\theta_0|` for every parameter whose bounds exclude 0, so equal steps are equal *relative* changes however far the optimum is from the start; parameters whose bounds include 0 use :math:`\theta/\theta_0`. ``'lin'`` uses :math:`\theta/\theta_0` for all (the previous behaviour). Estimates, Hessian and covariances are defined as before; only the optimiser's path changes.
  - ``'grad'`` (default ``'2-point'``) and ``'grad_step'`` (default ``1e-6``): finite-difference gradient of SLSQP, LMBFGS, TC and BFGS with a relative step (``'3-point'``: central differences, twice the cost). ``'legacy'`` restores SciPy's default (forward differences, absolute step 1e-8), whose gradient is mostly noise for weakly sensitive parameters.
  - ``'fnorm'`` (default ``True``): the optimiser minimises the objective divided by a scale (MLE without its constant :math:`\tfrac12\sum\log 2\pi\sigma^2`, same minimiser): the objective at the start, or for LMBFGS ``min(start value, 1)`` because its test is already relative above 1. It restarts from its result (at most 4 times) while the objective falls below a tenth of the scale. ``'tol'`` then acts relative to the objective at the optimum for every method (LMBFGS/SLSQP ``ftol``, NMS ``fatol``, TC ``gtol``); BFGS and DE keep their own criteria. A ``'tol'`` of 0.1 lets LMBFGS stop when an iteration gains less than 10 %; use 1e-6 to 1e-9 for accurate estimates.
- **Bootstrap start.** With ``var-cov='B'`` every replicate is refitted from the reference estimate and, with ``'ms': True``, from 4 random starts as well; the best fit is kept (``iden_opt['boot_start'] = 'mixed'``, default). ``'ref'`` uses the reference estimate only: fastest, but a replicate can stay in the reference minimum when a poorly identifiable parameter has a better one elsewhere. ``'multi'`` repeats the reference fit's method for every replicate (with ``'ms': True``, a serial multi-start of 20 random starts per replicate: about 4 times slower than ``'mixed'``, same optima on case study 2).
- **Bootstrap estimates.** With ``var-cov='B'`` the reported estimates (``scpr``, ``uncert``'s ``'estimations'``, and ``models['theta']``) are the **mean of the bootstrap refits**; the reference fit is kept in the parmest result as ``scpr_raw``.
- **Bootstrap filter.** With ``var-cov='B'`` a replicate is kept only if its optimum is at most twice the reference optimum (constant-free objective). For a resample of the same data this rejects fits that stopped far from the minimum, not unlucky resamples. ``parmest`` prints how many replicates were kept, rejected or not converged (also in ``result[model].boot_counts``).
- **σ fallback.** Where a measurement has no ``MES_E`` value, ``unc`` is used as its *absolute* σ, with a warning. For relative errors give the per-point σ in ``MES_E`` (``expera`` writes it).
- **Plots.** ``c_plt`` and ``f_plt`` act inside ``save_rounds()``; ``parmest()`` and ``uncert()`` do not plot (``uncert`` only saves the step-size test in ``./meshindep_plots/`` when ``eps`` is None). ``plt_s`` decides whether figures are displayed as well as saved, for ``save_rounds`` (default ``False``), the rCC plot of ``estima`` (``./estimability/``, default ``True``) and the ``validation`` plots (``./validation/``, default ``True``). Figures are only displayed when matplotlib can show them (a window or a notebook); a failing plot gives a warning and never stops the workflow.

  - *Fit plot* (``f_plt``): per experiment, each output with its data (error bars ±2σ, σ from ``MES_E`` or ``unc``), the model, and the 95 % band of the prediction due to the parameter uncertainty; under it the weighted residuals :math:`(y-\hat y)/\sigma` with the ±2 lines; then the inputs with the sampling times. The caption gives :math:`\sum[(y-\hat y)/\sigma]^2/(N-p)` with its 95 % range for a correct model and correct σ. A *parity plot* (measured against model) is saved as well.
  - *Confidence plot* (``c_plt``): densities with the 95 % CI on the diagonal, 95 % joint ellipses (and, with ``var-cov='B'``, the bootstrap replicates) below it, correlation coefficients above it.
  - Parameters are labelled θ1, θ2, ... or with your own names in ``models['theta_names'] = {'M': ['k0', 'Ea', ...]}``.
- **t-values.** A parameter to which the measured outputs have no sensitivity gets ``t = NaN`` rather than an infinite t-value.
- **model_update** (default ``True``): ``estima`` writes its selected subset and subset fit into ``models``, and ``save_rounds(..., best_uncert_result=...)`` stores and plots that subset fit instead of the ``result`` passed. ``False`` leaves ``models`` alone and stores ``result``.
- **Errors.** Failures in the model are reported as ``middoe.krnl_simula.SimulationError`` (see :ref:`model-file`).

.. _practical-workflow:

Complete workflow, step by step
-------------------------------

The calls of an in-silico study in order, with what each one returns and writes. The dictionaries are those of :ref:`config-dicts`; complete, runnable studies are at the end of this section.

Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from middoe.sc_sensa import sensa
   from middoe.des_pp import mbdoe_pp
   from middoe.des_md import mbdoe_md
   from middoe.krnl_expera import expera
   from middoe.iden_parmest import parmest
   from middoe.iden_uncert import uncert
   from middoe.sc_estima import estima
   from middoe.iden_valida import validation
   from middoe.log_utils import save_rounds, save_to_jac, load_from_jac, save_to_xlsx
   from middoe.iden_utils import run_postprocessing

Global Sensitivity Analysis (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify influential parameters before designing experiments:

.. code-block:: python

   # Run Sobol sensitivity analysis
   sobol_results = sensa(gsa, models, system)

   # Save results
   save_to_jac(sobol_results, purpose="sensa")

   # Export to Excel
   results = load_from_jac()
   sensa_data = results['sensa']
   save_to_xlsx(sensa_data, sensa_data['problem'])

**Output:** ``sensa()`` returns a dict with ``'analysis'`` (first-order ``S1`` and total-order ``ST`` Sobol indices per model, response and time point) and ``'problem'`` (per model, the names and bounds of the sampled inputs), plus plots if ``gsa['plt'] = True``. ``save_to_jac(..., purpose="sensa")`` writes ``./sensa_results.jac``; ``save_to_xlsx(sensa, sensa['problem'])`` writes ``./sobol_results.xlsx`` with one sheet per model and response, containing the **total-order indices** ``ST`` only.

Round 1: Initial Design and Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Design the first experiment**

.. code-block:: python

   round = 1
   designs = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=16)

**Returns:** Optimal control profiles (``u(t)``), time-invariant inputs (``w``), and sampling times that minimize parameter uncertainty according to the selected criterion (``des_opt['pp_ob']``).

**Step 2: Generate data**

.. code-block:: python

   # For insilico studies:
   expera(system, models, insilicos, designs, expr=1, swps=designs['swps'])

   # For preliminary (classic) experiments, pass an empty design:
   # expera(system, models, insilicos, {}, expr=1)

   # For real experiments: manually add data to data.xlsx following the design

If ``design_decisions`` is empty, ``expera()`` runs the iso-valued preliminary experiment ``insilicos['prels'][str(expr)]``; otherwise it runs the given design. ``insilicos['prels']`` must be present in both cases.

**Result:** New sheet ``'1'`` (the ``expr`` number) in ``./data.xlsx`` containing measurements for experiment 1.

**Step 3: Estimate parameters**

.. code-block:: python

   resultpr = parmest(system, models, iden_opt, case='strov')

The ``case`` argument of ``parmest()``:

- ``None`` (default): standard estimation; starts from ``models['theta']``. The existing masks in ``models['mutation']`` are kept (fixed parameters stay at their ``models['theta']`` value).
- ``'strov'``: starts from ``models['thetastart']`` instead of ``models['theta']`` (if present). ``thetastart`` is the **original starting guess**, stored once by the first ``uncert`` call, so every round restarts from the same point (as in the MIDDoE paper's case studies); set ``models['thetastart']`` yourself to choose another start. Like ``None``, it keeps the masks in ``models['mutation']``; parameters fixed by ``estima`` take their ``thetastart`` value.
- ``'freeze'``: used internally by ``estima()``; for normal use it behaves like ``None``.

.. note::

   ``estima()`` writes the subset it selects into ``models['mutation']`` and no ``case`` resets it. If each round should re-estimate and re-rank **all** parameters (as in the MIDDoE paper's Case Study 2), free them yourself before calling ``parmest``:

   .. code-block:: python

      for m in models['can_m']:
          models['mutation'][m] = [True] * len(models['mutation'][m])


**Returns:** Estimated parameters :math:`\hat{\boldsymbol{\theta}}`, objective function value, convergence status, and computational time for each model in ``models['can_m']``.

**Step 4: Quantify uncertainty**

.. code-block:: python

   uncert_results = uncert(resultpr, system, models, iden_opt)
   resultun = uncert_results['results']
   obs = uncert_results['obs']

**Returns:** ``uncert_results['results'][m]`` holds, for each model ``m``: the estimates ``'estimations'``, the objective values ``'LS'``, ``'WLS'``, ``'MLE'``, ``'Chi'``, ``'MSE'``, ``'R2_responses'`` and ``'R2_total'``, the variance-covariance matrix ``'V_matrix'``, confidence intervals ``'CI'``, t-values ``'t_values'``, the P-test value ``'P'`` and the sensitivity matrix ``'LSA'``; ``uncert_results['obs']`` is the number of observations. The reference t-value is not part of this dict: ``save_rounds()`` returns it and stores it as ``round_data['Round k']['trv'][m]``.

The ``case`` argument of ``uncert()``: with ``case=None`` (default) ``uncert()`` also **updates** ``models`` in place (``models['theta']``, ``models['V_matrix']``, ``models['LSA']``, and ``models['thetastart']``, which the first call sets to the starting guess and later calls leave unchanged), so the next ``mbdoe_pp``/``mbdoe_md`` call designs at the new estimate. Any other value of ``case`` skips this update.

**Step 5: Evaluate estimability**

.. code-block:: python

   ranking, k_optimal_value, rCC_values, J_k_values, best_uncert_result = \
       estima(resultun, system, models, iden_opt, round)

**Returns:** Parameter ranking by estimability, optimal number of estimable parameters :math:`k^*`, and corrected critical ratio curve :math:`r_{CC,k}`.

**Step 6: Save round results**

.. code-block:: python

   round_data = {}
   save_rounds(round, resultun, 'preliminary', round_data, models, iden_opt, obs, system,
               ranking=ranking, k_optimal_value=k_optimal_value, rCC_values=rCC_values,
               J_k_values=J_k_values, best_uncert_result=best_uncert_result)

**Result:** Round 1 data stored in ``round_data`` dictionary for later analysis.

Sequential Rounds (2, 3, ..., N)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat the workflow to progressively reduce parameter uncertainty:

.. code-block:: python

   for round in range(2, 5):  # Example: rounds 2-4
       # Design next experiment
       designs = mbdoe_pp(des_opt, system, models, round=round, num_parallel_runs=16)

       # Collect data
       expera(system, models, insilicos, designs, expr=round, swps=designs['swps'])

       # Estimate with cumulative dataset
       resultpr = parmest(system, models, iden_opt, case='strov')

       # Quantify uncertainty
       uncert_results = uncert(resultpr, system, models, iden_opt)
       resultun = uncert_results['results']
       obs = uncert_results['obs']

       # Check estimability
       ranking, k_optimal_value, rCC_values, J_k_values, best_uncert_result = \
           estima(resultun, system, models, iden_opt, round)

       # Save round
       save_rounds(round, resultun, 'MBDOE_PP', round_data, models, iden_opt, obs, system,
                   ranking=ranking, k_optimal_value=k_optimal_value, rCC_values=rCC_values,
                   J_k_values=J_k_values, best_uncert_result=best_uncert_result)

**Optional:** Update ``models['mutation']`` between rounds to manually fix poorly estimable parameters instead of using ``estima()``.

**Stopping criteria:**

- All t-values exceed threshold: :math:`t_i > t_{\text{ref}}`

Model Discrimination (Multi-Model Case)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If multiple candidate models compete, design discriminating experiments:

.. code-block:: python

   # Configure for discrimination
   des_opt['md_ob'] = 'BFF'  # or 'HR'

   # Design experiment maximizing model separation
   md_designs = mbdoe_md(des_opt, system, models, round=round)

   # Execute and analyze
   expera(system, models, insilicos, md_designs, expr=round, swps=md_designs['swps'])
   resultpr = parmest(system, models, iden_opt)   # case=None: all models in can_m

**Model selection:** Compare :math:`\chi^2` values and P-test probabilities (see :ref:`ptest`) to identify the best model.

Cross-Validation
~~~~~~~~~~~~~~~~

Validate the calibrated model across all experiments:

.. code-block:: python

   validres = validation(system, models, iden_opt, round_data)

Each fold refits with ``parmest(..., case='strov')`` and your ``iden_opt`` (so ``var-cov='B'`` bootstraps every fold); ``models['mutation']`` is set to the masks of the last round.

**Returns:** Leave-one-out cross-validation results: ``validres['R2_prd']`` and ``validres['R2_val']`` (keyed by fold, then model), and the per-model summaries ``validres['R2_stats'][m]`` and ``validres['MSE_stats'][m]`` (``'prediction_mean'``, ``'prediction_std'``, ``'validation_mean'``, ``'validation_std'``), i.e. :math:`R^2` and MSE on the calibration and validation sets.

Save Final Results
~~~~~~~~~~~~~~~~~~

Persist all workflow data:

.. code-block:: python

   # Save identification results
   save_to_jac(round_data, purpose="iden")

**Result:** Binary file ``./iden_results.jac`` (a Python pickle, in the current working directory) containing all rounds; it is overwritten on each call. ``load_from_jac()`` reads ``./iden_results.jac`` and ``./sensa_results.jac`` and returns ``{'iden': ..., 'sensa': ...}`` (``None`` for a missing file; ``None`` overall when neither exists). Only load ``.jac`` files from trusted sources.

Load and Post-Process
~~~~~~~~~~~~~~~~~~~~~

Reload saved data and generate reports:

.. code-block:: python

   # Load results
   results = load_from_jac()
   iden = results['iden']

   # Generate comprehensive reports
   run_postprocessing(
       round_data=iden,
       solvers=['M'],                    # Model identifiers
       selected_rounds=[1, 2, 3, 4],     # Rounds to analyze
       plot_global_p_and_t=True,         # per-round bar charts: t of every parameter, P of every model
       plot_confidence_spaces=True,      # 95 % regions, estimates ± CI, precision, volume
       plot_p_and_t_tests=True,          # t-test (and P) over the rounds
       export_excel_reports=True,        # Summary tables
       plot_estimability=True,           # Estimability ranking evolution
       show=False,                       # save the figures without opening windows (scripts)
       true_theta=None                   # optional true values (in silico): {'M': [...]} or a list
   )

What ``run_postprocessing`` writes in ``./post_processing/``:

- ``<model>_confidence_space.png``: 95 % confidence regions of all selected rounds on one plot (correlations of the last round in the upper cells).
- ``<model>_parameter_estimates.png``: each parameter's estimate ± 95 % CI per round (open marker: fixed in that round; dashed line: ``true_theta``).
- ``<model>_parameter_std_devs.png``: standard deviation of each parameter as a percentage of its estimate.
- ``<model>_ellipsoid_volume.png``: volume of the 95 % joint region; a change in the number of estimated parameters is marked, as volumes of different dimension cannot be compared.
- ``<model>_pt_tests.png``: absolute t-value of each parameter against the reference t per round (and the model probability P when several models are compared); rounds shaded by design type (the ``design_type`` given to ``save_rounds``: ``'preliminary'``/``'classic'``, ``'MBDOE_MD'``, ``'MBDOE_PP'``, ``'MBDoE-PP ...'`` ...).
- ``t_values_round_<k>.png`` and, with several models, ``p_values_round_<k>.png``.
- ``<model>_report_<k>.xlsx``: per experiment the measurements, the σ used, the model at the sampling times, residuals and weighted residuals, model curves and inputs; sheet ``_summary`` with estimates, sd, 95 % CI, t against the reference t and the fit statistics. ``<model>_report_summary.txt``: the same summary for the last round.
- ``estimability/rCC vs parameters in model_<model>.png``: rCC against k for every round, selected k marked.

**Outputs:**

All output folders are created in the current working directory:

- ``./post_processing/``: ``run_postprocessing()`` figures (parameter and t-value evolution, confidence regions, estimability) and Excel reports
- ``./confidence/``: confidence-ellipsoid plots written by ``save_rounds()`` when ``iden_opt['c_plt'] = True``
- ``./modelling/``: fitting plots written by ``save_rounds()`` when ``iden_opt['f_plt'] = True``
- ``./design/``: MBDoE design reports written by ``mbdoe_pp()``/``mbdoe_md()``

Complete, runnable examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The four case studies of the MIDDoE paper follow this workflow from start to end, as notebooks: see
:doc:`case_studies`.

.. _math:

Background: the equations
-------------------------

MIDDoE operates on lumped dynamic systems governed by differential-algebraic equations:

MIMO model structure
~~~~~~~~~~~~~~~~~~~~
.. math::

   \begin{align}
   \dot{\mathbf{x}}(t) &= \mathbf{f}(\mathbf{x}, \mathbf{z}, \mathbf{u}(t), \mathbf{w}, \boldsymbol{\theta})\\
   \mathbf{0} &= \mathbf{g}(\mathbf{x}, \mathbf{z}, \mathbf{u}(t), \mathbf{w}, \boldsymbol{\theta})\\
   \mathbf{y}(t) &= \mathbf{h}(\mathbf{x}, \mathbf{z}, \mathbf{u}(t), \mathbf{w}, \boldsymbol{\theta}) + \boldsymbol{\varepsilon}(t)
   \end{align}

**Variables and parameters:**

- :math:`\boldsymbol{\theta} \in \mathbb{R}^{N}`: model parameters to be estimated
- :math:`\mathbf{x}(t) \in \mathbb{R}^{N_x}`: time-variant differential states
- :math:`\mathbf{z}(t) \in \mathbb{R}^{N_z}`: time-variant algebraic states
- :math:`\mathbf{u}(t) \in \mathbb{R}^{N_u}`: time-variant controls (manipulated variables)
- :math:`\mathbf{w} \in \mathbb{R}^{N_w}`: time-invariant controls (design variables, initial conditions)
- :math:`\mathbf{y}(t) \in \mathbb{R}^{N_r}`: measured outputs
- :math:`\boldsymbol{\varepsilon}(t)`: measurement error (noise)


Sobol Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Two sample matrices :math:`\mathbf{A}, \mathbf{B} \in [0,1]^{N \times d}` are generated from the scrambled Sobol sequence, where :math:`N` is the number of quasi-random base samples. For each variable :math:`i \in \{1,...,d\}` a hybrid matrix :math:`\mathbf{AB}_i` is generated by replacing the :math:`i`-th column of :math:`\mathbf{A}` with the corresponding column from matrix :math:`\mathbf{B}`, keeping the remaining columns unchanged. The total number of model evaluations required is:

.. math::

   N_{\text{eval}} = N(2d + 2)

This includes model evaluations of all :math:`N` samples of matrices :math:`\mathbf{A}`, :math:`\mathbf{B}`, and each :math:`\mathbf{AB}_i`. First-order Sobol indices, measuring the main effect of each input on output variance, are estimated using Jansen's estimator (as in the MIDDoE paper):

.. math::

   S_i = 1 - \frac{1}{2N \text{Var}(Y)} \sum_{k=1}^{N} [M(\mathbf{B}_k) - M(\mathbf{AB}_i^k)]^2

Total-order Sobol indices, which include all interactions involving input :math:`i`, are computed using the Jansen estimator:

.. math::

   S_{Ti} = \frac{1}{2N \text{Var}(Y)} \sum_{k=1}^{N} [M(\mathbf{A}_k) - M(\mathbf{AB}_i^k)]^2

where :math:`\text{Var}(Y)` represents the total variance of the model output, estimated from the responses of the two base sample matrices A and B.

Fisher Information Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~

The local sensitivity matrix :math:`\mathbf{Q}_r` for each measured response :math:`r` is defined as:

.. math::

   \mathbf{Q}_r = \begin{bmatrix}
   \frac{\partial y_r(\theta_1, t_1)}{\partial \theta_1} & \cdots & \frac{\partial y_r(\theta_N, t_1)}{\partial \theta_N}\\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_r(\theta_1, t_{N_{sp}})}{\partial \theta_1} & \cdots & \frac{\partial y_r(\theta_N, t_{N_{sp}})}{\partial \theta_N}
   \end{bmatrix}

where :math:`\hat{y}_r(\theta, t)` denotes model predictions, :math:`N_{sp}` is the number of sampling points, and :math:`N_{\theta}` is the number of active parameters. This matrix enables construction of the Fisher information matrix :math:`\mathbf{I}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})` as:

.. math::

   \mathbf{I}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}) = \sum_{r} \sum_{r'} \sigma_{r,r'}^{-2} \mathbf{Q}_r^T \mathbf{Q}_r + \boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1}

where prior information of parameters can be neglected (i.e., :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1} = 0`; this is the default, ``des_opt['priors'] = False``, see :ref:`config-dicts`), and :math:`\sigma_{r,r'}` are elements from the inverse of the variance-covariance matrix of measurements errors (MIDDoE assumes independent errors: a diagonal matrix with the σ of each measurement). If the condition number of the Fisher matrix, defined as :math:`\kappa(\mathbf{I})`, exceeds a critical threshold (typically :math:`10^3`), the model is considered sloppy and if this number is extremely high, the matrix is ill-conditioned.

Confidence Intervals and t-Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resulting covariance matrix :math:`\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})` is used to compute Confidence Interval :math:`CI_i` and t-value of estimated parameter :math:`i` at a confidence level of :math:`1 - \alpha` (95%):

.. math::

   CI_i = \sqrt{v_{\theta,ii}} \cdot t_{1 - \alpha/2, \text{DoF}} \quad \text{for } i = 1, ..., N_{\theta}

.. math::

   t_i = \frac{\hat{\theta}_i}{\sqrt{v_{\theta,ii}}}, \quad i = 1, ..., N_{\theta}

where :math:`v_{\theta,ii}` is the :math:`i`-th diagonal element of :math:`\mathbf{V}`, and :math:`t_{1 - \alpha/2, \text{DoF}}` is the upper :math:`1 - \alpha/2` critical value of a Student's t-distribution with :math:`\text{DoF} = N_{obs} - N_{\theta}` degrees of freedom (:math:`N_{obs}`: all measured points of all outputs and experiments; :math:`N_{\theta}`: estimated parameters). The t-test provides a quantitative measure of estimation precision by statistically assessing whether each estimated parameter :math:`\hat{\theta}_i` is significantly different from the parameter noise with a zero mean. This is verified by checking whether the computed :math:`t_i`-value exceeds the reference threshold :math:`t_{\text{ref}}`, corresponding to a Student t-value at the same confidence level.

MBDoE-MD: T-Optimality Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general form of the T-optimal design problem is expressed as:

.. math::

   \boldsymbol{\xi}_{\text{OPT}} = \arg \max_{\boldsymbol{\xi} \in \Omega} \sum_{k=1}^{N_{sp}} \sum_{l=1}^{N_m} \sum_{l'=l+1}^{N_m} T_{l,l'}(\boldsymbol{\xi}, \boldsymbol{\theta}_l, \boldsymbol{\theta}_{l'}, t_k)

.. math::

   T_{l,l'}(\boldsymbol{\xi}, \boldsymbol{\theta}_l, \boldsymbol{\theta}_{l'}, t) = [\mathbf{y}_l(\boldsymbol{\xi}, t) - \mathbf{y}_{l'}(\boldsymbol{\xi}, t)]^T \mathbf{F}_{l,l'}(t) [\mathbf{y}_l(\boldsymbol{\xi}, t) - \mathbf{y}_{l'}(\boldsymbol{\xi}, t)]

where :math:`N_m` denotes the number of competing models, :math:`\boldsymbol{\theta}_l` and :math:`\boldsymbol{\theta}_{l'}` are the respective preliminary estimated parameter vectors for models :math:`l` and :math:`l'`, :math:`\Omega` defines the feasible design space, and :math:`\mathbf{F}_{l,l'}(t)` is a weight factor based on the uncertainties in observation and parameter estimation at different sampling times (which is 1 for HR). For the BFF, :math:`\mathbf{F}_{l,l'}(t)` incorporates model sensitivity and covariance terms, and is defined at each sampling time :math:`t` as:

.. math::

   \mathbf{F}_{l,l'}(t) = [\boldsymbol{\Sigma}_y + \mathbf{W}_l(t) + \mathbf{W}_{l'}(t)]^{-1}

where :math:`\boldsymbol{\Sigma}_y` is the variance-covariance matrix of measurement errors, and :math:`\mathbf{W}_l(t)` and :math:`\mathbf{W}_{l'}(t)` are representing the modeling error contributions, derived from the sensitivity of each model to its parameters. Each :math:`\mathbf{W}_l(t)` is constructed as:

.. math::

   \mathbf{W}_l(t) = \mathbf{Q}_l(t) \mathbf{V}_{\boldsymbol{\theta}_l} \mathbf{Q}_l^T(t)

with :math:`\mathbf{V}_{\boldsymbol{\theta}_l}` the parameter variance-covariance matrix of model :math:`l` (eq. 11 of the MIDDoE paper is printed with its inverse; the code uses :math:`\mathbf{V}`, the standard Buzzi-Ferraris–Forzatti form). The criterion :math:`T_{l,l'}` above uses :math:`\mathbf{F}_{l,l'}(t)`, which is already the inverse of :math:`\boldsymbol{\Sigma}_y + \mathbf{W}_l + \mathbf{W}_{l'}`. Here :math:`\mathbf{Q}_l(t)` is the sensitivity matrix of model :math:`l` with respect to its estimated parameters :math:`\boldsymbol{\theta}_l` from the preliminary estimations, evaluated at time :math:`t`:

.. math::

   \mathbf{Q}_l(t) = \begin{bmatrix}
   \frac{\partial y_{l,1}(t, \boldsymbol{\theta}_{l})}{\partial \theta_{l,1}} & \cdots & \frac{\partial y_{l,1}(t, \boldsymbol{\theta}_{l})}{\partial \theta_{l,N}}\\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_{l,M}(t, \boldsymbol{\theta}_{l})}{\partial \theta_{l,1}} & \cdots & \frac{\partial y_{l,M}(t, \boldsymbol{\theta}_{l})}{\partial \theta_{l,N}}
   \end{bmatrix}

MBDoE-PP: Alphabetical Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MBDoE-PP aims to minimise parameter uncertainty by reducing the size of the confidence hyperellipsoid through optimisation of a scalar metric :math:`\Phi(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))`. The MBDoE-PP problem is formulated as:

.. math::

   \boldsymbol{\xi}_{\text{OPT}} = \arg \min_{\boldsymbol{\xi} \in \Omega} \Phi(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))

.. list-table:: MBDoE-PP alphabetical criteria
   :widths: 15 50 35
   :header-rows: 1

   * - Criterion
     - Definition
     - Mathematical form
   * - D
     - Reducing the confidence region size by minimising the determinant of the :math:`\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})`
     - :math:`\Phi_D = \det(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))`
   * - A
     - Lowering the average variance across all parameter estimates by minimising the trace of :math:`\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})`
     - :math:`\Phi_A = \text{tr}(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))`
   * - E
     - Reducing the variance of the most uncertain parameter by minimising the biggest eigenvalue of :math:`\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})`
     - :math:`\Phi_E = \lambda_{\max}(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))`
   * - ME
     - Ensure uniform precision across parameters by minimises the condition number of the :math:`\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w})`
     - :math:`\Phi_{ME} = \kappa(\mathbf{V}(\boldsymbol{\theta}, \mathbf{U}, \mathbf{w}))`

Estimability Analysis
~~~~~~~~~~~~~~~~~~~~~

A scaled sensitivity matrix :math:`\mathbf{Q}^S \in \mathbb{R}^{N_{tsp} \times N_{\theta}}` is constructed, where :math:`N_{tsp}` is the total number of responses samples (i.e., the sum of :math:`N_{sp}` across all measured outputs), and :math:`N_{\theta}` is the number of model parameters. Each element of :math:`\mathbf{Q}^S`, denoted as :math:`q_{ij}^S`, is computed by scaling the local sensitivities as:

.. math::

   q_{ij}^S = \frac{\partial y_i}{\partial \theta_j} \cdot \frac{\sigma_{\theta_j}}{\sigma_{y_i}}

where :math:`\frac{\partial y_i}{\partial \theta_j}` is the local sensitivity of the :math:`i`-th model output to the :math:`j`-th parameter, :math:`\sigma_{y_i}` is the standard deviation of the measurement error of the :math:`i`-th sample and :math:`\sigma_{\theta_j}` the scale of the :math:`j`-th parameter (MIDDoE uses its current estimate :math:`|\hat\theta_j|`; :math:`N_{\theta}` is the number of currently estimated parameters). An iterative deflation algorithm is then applied to rank parameters. Once the parameters are ranked, the optimal subset is selected by estimating nested top-:math:`k` subsets for :math:`k \in \{1,2,...,N_{\theta} - 1\}` and evaluating their WLS performance (:math:`r_{CC,N_\theta} = 0` is included, so keeping all parameters is a possible outcome). For each :math:`k`, compute the critical ratio :math:`r_{C,k}`:

.. math::

   r_{C,k} = \frac{J_k - J_{N_{\theta}}}{N_{\theta} - k}

where :math:`J_{N_{\theta}}` is the objective value for the full model in which all parameters are estimated. To account for statistical bias and sample size effects, compute the corrected form of the critical ratio :math:`r_{CC,k}`:

.. math::

   r_{CC,k} = \frac{N_{\theta} - k}{N_{tsp}} \left[ \max\left( r_{C,k} - 1, \frac{2r_{C,k}}{N_{\theta}-k+2} \right) - 1 \right]

Determine the optimal number of parameters :math:`k^*` to estimate by identifying the value of :math:`k` that minimises :math:`r_{CC,k}`.

.. _ptest:

P-Test for Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

After calibration of the candidate models, P-test quantifies the relative likelihood that each candidate model provides the best fit to the observed data. This method assigns a probability score :math:`P_i` to each model :math:`i`, reflecting its relative quality based on goodness-of-fit metrics. The probability of the :math:`i`-th model being the best among :math:`N_m` candidates is defined as:

.. math::

   P_i = \frac{1/\chi_i^2}{\sum_{j=1}^{N_m} 1/\chi_j^2} \times 100

where :math:`\chi_i^2` values are obtained after parameter estimation for each model (the ``'Chi'`` value reported by ``uncert()``), and the :math:`P_i` sum to 100. For example, :math:`\chi^2 = [10, 20, 40]` gives :math:`P = [57.1, 28.6, 14.3]`. (Eq. 2 of the MIDDoE paper is printed with a spurious leading ":math:`1 -`"; the form above is the intended one.) Models with lower :math:`\chi^2` values are considered better descriptors of the system. A higher :math:`P_i` indicates stronger evidence in favour of model :math:`i`.
