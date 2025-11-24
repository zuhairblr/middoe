.. image:: https://research.dii.unipd.it/capelab/wp-content/uploads/sites/36/2025/03/logo-Page-5.png
   :alt: CAPE-Lab / MIDDoE
   :align: center
   :width: 400

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

MIDDoE is an open-source Python package for systematic model identification workflows. It provides an integrated, physics-aware framework that combines **Global Sensitivity Analysis (GSA)**, **Estimability Analysis (EA)**, **Parameter Estimation**, **Uncertainty Quantification**, **Model-Based Design of Experiments (MBDoE)**, and **cross validation** for dynamic systems.

MIDDoE unifies the complete identification pipeline from pre-experimental diagnostics through post-analysis reporting—suitable for experimentalists and engineers with limited programming expertise.

1. Installation
============

Install via pip:

.. code-block:: bash

   pip install middoe

MIDDoE requires Python ≥ 3.9 and automatically installs dependencies: NumPy, SciPy, Pandas, Matplotlib, and Pymoo.

2. Mathematical Framework
======================

MIDDoE operates on lumped dynamic systems governed by differential-algebraic equations:

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

3. Workflow Steps
==============

A complete MIDDoE identification workflow for Basic user follows this sequence:

I. System and model establishment
------------------------------

1. **System structure**
   Define the experimental and operational space in the ``system`` dictionary, including all time-variant inputs, time-invariant inputs, measured outputs, and their physical/operational constraints (bounds, sampling limits, dead-times, CVP structure).

2. **Candidate models**
   Implement candidate models as black-box simulators and register them in the ``models`` dictionary, specifying parameter vectors, feasible bounds, estimation masks (``mutation``), and the model interface type (e.g. ``'pys'``, ``'pym'``, function name, or ``'gpr'``).

3. **Global Sensitivity Analysis**
   Configure the ``gsa`` dictionary and call ``sensa()`` from ``middoe.sc_sensa`` to analyse the influence of parameters and/or controls on the outputs, providing a ranking that guides model reduction and design choices.

II. Data creation
-------------

4. **Preliminary data**
   Either create a ``data.xlsx`` file in the project repository containing your experimental measurements, or define the ``insilicos`` dictionary (true model, true parameters, noise type) and generate synthetic data by calling ``expera()`` from ``middoe.krnl_expera``.

5. **MBDoE for discrimination**
   Set up the ``des_opt`` dictionary for model discrimination (``md_ob`` set to ``'HR'`` or ``'BFF'``) and run ``mbdoe_md()`` from ``middoe.des_md`` to design experiments that maximally separate the predictions of rival models.

6. **MBDoE for precision**
   Configure ``des_opt`` for parameter precision (``pp_ob`` set to ``'D'``, ``'A'``, ``'E'``, or ``'ME'``) and run ``mbdoe_pp()`` from ``middoe.des_pp`` to obtain experiments that minimise parameter uncertainty for the selected model(s).

7. **Append data**
   After executing the designed experiments (in the lab or in-silico via ``expera()``), update ``data.xlsx`` or your in-memory data structure so that all newly collected experiments are available for calibration.

III. Model identification
--------------------

8. **Parameter estimation**
   With the current dataset and model definitions, run ``parmest()`` from ``middoe.iden_parmest`` using the ``iden_opt`` dictionary to estimate parameters, compute goodness-of-fit metrics, and obtain convergence information.

9. **Uncertainty analysis**
   Pass the estimation results to ``uncert()`` from ``middoe.iden_uncert`` to evaluate variance–covariance matrices, confidence intervals, t-values, and prediction metrics, characterising the precision of the estimated parameters.

10. **Estimability analysis**
    Run ``estima()`` from ``middoe.sc_estima`` to rank parameters by estimability and determine the optimal subset that can be reliably estimated with the current data, informing which parameters should remain active or be fixed in subsequent rounds.

11. **Round storage**
    Treat the complete sequence (design → experiment → estimation → uncertainty → estimability) as the *i*-th round and save it into a ``round_data`` dictionary by calling ``save_rounds()`` from ``middoe.log_utils``.

12. **Sequential rounds**
    Repeat steps 4–11, updating experimental designs, data, and parameter masks round by round until the desired discrimination and calibration performance indicators (e.g. model selection metrics, confidence intervals, t-tests) are achieved.

IV. Post analysis
----------------

13. **Model validation**
    Perform cross-validation over the full dataset by calling ``validation()`` from ``middoe.iden_valida``, obtaining calibration/validation statistics (e.g. per-response and global \(R^2\), residual analysis) for the final selected model(s).

14. **Save results**
    Persist all identification and design results to disk by calling ``save_to_jac()`` from ``middoe.log_utils`` (e.g. with purposes such as ``"iden"`` or ``"sensa"`` for later reuse).

15. **Load results**
    Reload previously saved workflows and analysis outputs by calling ``load_from_jac()`` from ``middoe.log_utils``, enabling further inspection, reporting, or continuation of the workflow.

16. **Post-processing**
    Generate global reports and visualisations by calling ``run_postprocessing()`` from ``middoe.iden_utils`` on the stored ``round_data``, producing plots (parameter trajectories, confidence ellipsoids, p/t-tests, estimability evolution) and Excel summaries for documentation and decision-making.


4. Project structure
=================

Organise your project as follows:

.. code-block:: text

   my_project/
   ├── data.xlsx                           # Your experimental measurements
   │
   ├── model1.py                           # Python file defining your model 1
   ├── model2.py                           # Python file defining your model 2
   ├── modeli.py                           # Python file defining your model i
   │
   └── workflow.py/                        # analysis and dentification workflow

5. Data structure
==============

MIDDoE reads data from a data.xlsx file in the project repository, Each sheet in ``data.xlsx`` represents one batch/experiment and contains measured outputs (MES), time-invariant controls :math:`\mathbf{w}`, and time-variant controls :math:`\mathbf{u}(t)` in a single table.

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

6. model.py structure
==============

MIDDoE can unboard and tag models in several ways:

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - **Type**
     - **Configuration**
     - **When to use**
   * - External Python script
     - ``'krt': {'M': 'pys'}, 'src': {'M': 'models/my_model.py'}``
     - You have your own ODE solver or external simulator (Suggested over global method and suggested for interfacing external tools)
   * - Built-in models
     - ``'krt': {'M': 'pym'}``
     - For testing; MIDDoE provides basic test models (typically solid-fluid reaction models)
   * - Global Python function
     - ``'krt': {'M': 'my_solve_function'}``
     - Function already defined in your script, in a global space.
   * - gPROMS coupling
     - ``'krt': {'M': 'gpr'}, 'creds': {'M': 'user:pass@server'}``
     - gPAS file created by gPROMS, to couple with gPROMS models.

In general it treats models as a black box and requires a standard model interface. Create a Python file with a function following this signature ('pys') approach:

.. code-block:: python

   def solve_model(t, y0, tii, tvi, theta):
       """
       Standard MIDDoE model interface.

       Parameters
       ----------
       t : list
           Time vector for evaluation.
       y0 : dict
           Initial conditions for differential states.
           Keys are state names, values are initial values.
           Example: {'y1': 0.0, 'y2': 0.5}

       tii : dict
           Time-invariant inputs (design variables, initial concentrations).
           Example: {'w1': 0.5, 'w2': 1.0}

       tvi : dict
           Time-variant inputs (controls as functions of time).
           Each key maps to a list matching time vector length.
           Example: {'u1': [296.15, 297.5, 298.0, ...]}

       theta : list
           Model parameters to be estimated.
           Example: [50000, 75000, 0.4116, 111900, 9905, 30000]

       Returns
       -------
       dict
           Dictionary with response trajectories.
           Keys match measured output names.
           Values are lists matching time vector length.
           Example: {'y1': [0.0, 0.01, 0.02, ...], 'y2': [0.0, 0.005, 0.010, ...]}
       """

       # Example: ODE integration using SciPy
       from scipy.integrate import odeint

       # Extract parameters
       k1, k2, k3, k4, k5, k6 = theta

       # Define differential equations
       def system_odes(state, t_point, u_t, w):
           C_A, C_B = state
           # Reaction kinetics (example)
           u_value = u_t(t_point)  # Get control value at time t_point
           dCA_dt = -k1 * C_A + k2 * w['w1']
           dCB_dt = k1 * C_A - k3 * C_B
           return [dCA_dt, dCB_dt]

       # Interpolate time-variant controls
       from scipy.interpolate import interp1d
       u1_func = interp1d(t, tvi['u1'], kind='linear', fill_value='extrapolate')

       # Integrate
       y0_list = [y0.get('CA', 0.0), y0.get('CB', 0.0)]
       solution = odeint(system_odes, y0_list, t, args=(u1_func, tii))

       # Return in MIDDoE format
       return {
           'y1': solution[:, 0].tolist(),
           'y2': solution[:, 1].tolist()
       }

7. Configuration Dictionaries
==============


**system Dictionary Definition**

The **system** dictionary is a central hub for all MIDDoE operations. It specify system characteristics.

.. code-block:: python

   system = {
       # ========== TIME CONFIGURATION ==========
       't_s': [0, 16],        # Time span: [start, end]
       't_r': 0.02,           # Time resolution: every 0.02
                              # Smaller → better accuracy but slower (it must match with the global time grid in data.xlsx
       't_d': 0.3,            # Dead-time fraction: exclude first 30% and last 30%
                              # for controls and sampling (typical batch startup/cooldown)

       # ========== TIME-VARIANT INPUTS: u(t) ==========
       'tvi': {
           'u1': {  # Example: Temperature profile
               'min': 296.15,          # Lower bound (K)
               'max': 306.15,          # Upper bound (K)
               'stps': 6,              # Number of CVP segments (switching points)
                                       # stps=6 → 5 segments
               'cvp': 'LPF',           # Control profile type:
                                       # 'CPF' = constant piecewise (steps)
                                       # 'LPF' = linear piecewise (ramps)
               'const': 'inc',         # Constraint on signal shape:
                                       # 'rel' = no constraint (relaxed)
                                       # 'inc' = monotonically increasing
                                       # 'dec' = monotonically decreasing
               'offl': 0.01,           # Min level change (fraction): 1% per segment
               'offt': 0.3             # Min time spacing (fraction): 30% of time span
           },
       },

       # ========== TIME-VARIANT OUTPUTS: y(t) ==========
       'tvo': {
           'y1': {  # Example: Product concentration
               'init': 0,              # Initial value (can be number or 'variable')
                                       # 'variable' = design decision (in tii)
               'meas': True,           # Measurable? (True) or virtual output (False)
               'sp': 17,               # Sampling points: max 17 measurements per batch
               'unc': 0.005,           # Measurement uncertainty (std dev): ±0.005 units (homoskedastic error regime)
               'offt': 0.3,            # Min sampling interval (fraction): 30% spacing
               'samp_s': 1,            # Synchronisation group: 1 = sync with other samp_s:1
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
           'y10': {  # Example: Initial concentration of reactant
               'min': 0.3,
               'max': 1.0
           },
           'y20': {  # Example: Initial concentration of catalyst
               'min': 0.19,
               'max': 1.0
           },
           'y20': {  # Example: Initial concentration of catalyst
               'min': 0.19,
               'max': 1.0
           },
       },

       # ========== TIME-INVARIANT OUTPUTS ==========
       'tio': {}  # Empty for this example; used for steady-state responses
   }

**Model Dictionary Definition**

The **model** dictionary is a central hub for all MIDDoE operations. It specify model characteristics.

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
           'M': 'pys'  # 'pys' = external Python script
                       # 'pym' = built-in MIDDoE models
                       # 'gpr' = gPROMS coupling
                       # 'function_name' = global function in namespace
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
       'tr_m': 'M',        # True model (ground truth)
       'theta': theta,     # True parameter vector
       'errt': 'abs',      # Error type:
                           # 'abs' = absolute noise
                           # 'rel' = relative (proportional to signal)
       'prels': {}         # Classic pre-defined designs (empty here) (iso-valued controls)
   }


**GSA Dictionary Definition**

The **gsa** dictionary is a settings dictionary to perform Global Sensitivity Analysis with Sobol method.

   # Global Sensitivity Analysis configuration
   gsa = {
       'var_s': False,           # Sensitivity w.r.t. controls? (u, w)
       'par_s': True,            # Sensitivity w.r.t. parameters? (θ)
       'var_d': False,           # Control sampling space:
                                 # False = system-defined bounds
                                 # float > 1 (e.g., 1.1) = ±10% around nominal
       'par_d': False,           # Parameter sampling space:
                                 # False = models-defined bounds
       'samp': 2**12,            # Sobol sample size: use powers of 2
                                 # 2^12 = 4096 base samples
                                 # Total evaluations ≈ (2·N_vars + 2) × base
       'multi': 0.7,             # Parallel execution: 0.7 = use 70% of CPU cores
       'tii_n': [0.508, 0.3925], # Nominal w values
       'tvi_n': [301.15],        # Nominal u values
       'plt': True               # Generate sensitivity plots?
   }

**MBDoE Dictionary Definition**

The **des_opt** dictionary is a settings dictionary to conduct MBDoE-MD adn MBDoE-PP optimisations.


.. code-block:: python

   des_opt = {
       # ========== FINITE DIFFERENCE SETUP ==========
       'eps': 1e-5,   # FDM perturbation size (normalised space)
                      # Smaller → better accuracy, larger → robustness
                      # None = auto mesh-independency test
                      # Typical: 1e-5 to 1e-3

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

       # ========== OPTIMISATION METHOD ==========
       'meth': 'PS',   # Solver type:
                       # 'PS' = Pattern Search (local, derivative-free)
                       # 'DE' = Differential Evolution (global)
                       # 'DEPS' = DE + PS (hybrid: global then local)

       # ========== ITERATION SETTINGS ==========
       'itr': {
           'pps': 100,     # Population size (for DE/PS)
           'maxmd': 5,     # Max iterations for MD objective
           'tolmd': 1,     # MD convergence tolerance
           'maxpp': 100,   # Max iterations for PP objective
           'tolpp': 1      # PP convergence tolerance
       },

       'plt': True     # Generate design plots?
   }

**Calibration Dictionary Definition**

The **iden_opt** dictionary is a settings dictionary to conduct parameter estimation and uncertainty analysis.

.. code-block:: python

   iden_opt = {
       # ========== OPTIMISATION SOLVER ==========
       'meth': 'LBFGSB',   # Parameter estimation method:
                           # 'LBFGSB' = Limited-memory BFGS (local, fast)
                           # 'SLSQP' = Sequential Least Squares (local, constrained)
                           # 'DE' = Differential Evolution (global, slow)
                           # 'NMS' = Nelder–Mead Simplex (local, derivative-free)
                           # 'BFGS' = Standard BFGS (local)
                           # 'TC' = Trust-region Constrained (local, smooth)

       'ms': True,         # Multi-start? (True = try multiple random starts)
       'maxit': 500,       # Maximum iterations per solver attempt
       'tol': 0.1,         # Convergence tolerance

       # ========== FINITE DIFFERENCE METHOD ==========
       'sens_m': 'central',# FDM scheme:
                           # 'central' = central differences (2nd order, more accurate)
                           # 'forward' = forward differences (1st order)
       'eps': 1e-5,        # FDM perturbation size
                           # None = auto mesh-independency test

       # ========== OBJECTIVE FUNCTION ==========
       'ob': 'WLS',        # Cost function:
                           # 'LS' = Least Squares
                           # 'WLS' = Weighted LS (accounts for measurement uncertainty)
                           # 'MLE' = Maximum Likelihood (if noise is known)
                           # 'CS' = Chi-Square

       # ========== UNCERTAINTY QUANTIFICATION ==========
       'var-cov': 'B',     # Variance-covariance method:
                           # 'H' = Hessian-based (fast, local, assumes linearity)
                           # 'J' = Jacobian-based (faster)
                           # 'B' = Bootstrap (robust, global, slow)
       'nboot': 50,        # Bootstrap samples (if var-cov='B')

       # ========== INITIALISATION ==========
       'init': None,       # Starting point:
                           # None = use theta from models dict
                           # 'rand' = random initialization

       # ========== VISUALISATION ==========
       'c_plt': False,     # Confidence ellipsoid plots?
       'f_plt': True,      # Fitting comparison plots?
       'plt_s': True,      # Show plots while saving?
       'log': False        # Verbose logging?
   }

8. Citation
==============

If you use MIDDoE in your research, please cite:

[1]. Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
     MIDDoE: An MBDoE Python package for model identification, discrimination,
     and calibration.
     *Digital Chemical Engineering*, 17, 100276.
     https://doi.org/10.1016/j.dche.2025.100276

[2]. Tabrizi, S.Z.B., Barbera, E., Da Silva, W.R.L, & Bezzo, F. (2025).
     A Python/Numpy-based package to support model discrimination and identification.
     J.F.M. Van Impe, G. Léonard, S.S. Bhonsale, M.E. Polańska, F. Logist (Eds.).
     *Systems and Control Transactions*, 4 , 1282-1287.
     https://doi.org/10.69997/sct.192104

9. Applied to
==============

MIDDoE has been applied in various EU reports, research works and projects, e.g. :

[1]. Tabrizi, Z., Rodriguez, C., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
     Wet Carbonation of Industrial Recycled Concrete Fines: Experimental Study and Reaction Kinetic Modeling.
     *Ind Eng Chem Res*, 64, 45, 21412–21425.
     https://doi.org/10.1021/acs.iecr.5c02835

10. Documentation Contents
==============

.. toctree::
   :maxdepth: 2

   api

11. Resources
==============

- **GitHub:** https://github.com/zuhairblr/middoe
- **PyPI:** https://pypi.org/project/middoe/
- **Paper:** https://doi.org/10.1016/j.dche.2025.100276
- **Issues & Support:** https://github.com/zuhairblr/middoe/issues
- **Case Studies:** https://github.com/zuhairblr/middoe/tree/main/tests/paper

12. Indices and Tables
==============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`