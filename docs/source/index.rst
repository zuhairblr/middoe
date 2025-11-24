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

MIDDoE is an open-source Python package for systematic model identification workflows. It provides an integrated, physics-aware framework that combines **Global Sensitivity Analysis (GSA)**, **Estimability Analysis (EA)**, **Parameter Estimation**, **Uncertainty Quantification**, **Model-Based Design of Experiments (MBDoE)**, and **cross validation** for dynamic systems.

MIDDoE unifies the complete identification pipeline from pre-experimental diagnostics through post-analysis reporting—suitable for experimentalists and engineers with limited programming expertise.

Quick Start: Installation and First Use
----------------------------------------

Install via pip:

.. code-block:: bash

   pip install middoe

MIDDoE requires Python ≥ 3.9 and automatically installs dependencies: NumPy, SciPy, Pandas, Matplotlib, and Pymoo.

Mathematical Framework
======================

General Nonlinear MIMO System
-----------------------------

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

Core Workflow Steps
===================

A complete MIDDoE identification workflow for Basic user follows this sequence:

System and model establishment
==============================

1. **System structure**
   Define the experimental and operational space in the ``system`` dictionary, including all time-variant inputs, time-invariant inputs, measured outputs, and their physical/operational constraints (bounds, sampling limits, dead-times, CVP structure).

2. **Candidate models**
   Implement candidate models as black-box simulators and register them in the ``models`` dictionary, specifying parameter vectors, feasible bounds, estimation masks (``mutation``), and the model interface type (e.g. ``'pys'``, ``'pym'``, function name, or ``'gpr'``).

3. **Global Sensitivity Analysis**
   Configure the ``gsa`` dictionary and call ``sensa()`` from ``middoe.sc_sensa`` to analyse the influence of parameters and/or controls on the outputs, providing a ranking that guides model reduction and design choices.

Data creation
=============

4. **Preliminary data**
   Either create a ``data.xlsx`` file in the project repository containing your experimental measurements, or define the ``insilicos`` dictionary (true model, true parameters, noise type) and generate synthetic data by calling ``expera()`` from ``middoe.krnl_expera``.

5. **MBDoE for discrimination**
   Set up the ``des_opt`` dictionary for model discrimination (``md_ob`` set to ``'HR'`` or ``'BFF'``) and run ``mbdoe_md()`` from ``middoe.des_md`` to design experiments that maximally separate the predictions of rival models.

6. **MBDoE for precision**
   Configure ``des_opt`` for parameter precision (``pp_ob`` set to ``'D'``, ``'A'``, ``'E'``, or ``'ME'``) and run ``mbdoe_pp()`` from ``middoe.des_pp`` to obtain experiments that minimise parameter uncertainty for the selected model(s).

7. **Append data**
   After executing the designed experiments (in the lab or in-silico via ``expera()``), update ``data.xlsx`` or your in-memory data structure so that all newly collected experiments are available for calibration.

Model identification
====================

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

Post analysis
=============

13. **Model validation**
    Perform cross-validation over the full dataset by calling ``validation()`` from ``middoe.iden_valida``, obtaining calibration/validation statistics (e.g. per-response and global \(R^2\), residual analysis) for the final selected model(s).

14. **Save results**
    Persist all identification and design results to disk by calling ``save_to_jac()`` from ``middoe.log_utils`` (e.g. with purposes such as ``"iden"`` or ``"sensa"`` for later reuse).

15. **Load results**
    Reload previously saved workflows and analysis outputs by calling ``load_from_jac()`` from ``middoe.log_utils``, enabling further inspection, reporting, or continuation of the workflow.

16. **Post-processing**
    Generate global reports and visualisations by calling ``run_postprocessing()`` from ``middoe.iden_utils`` on the stored ``round_data``, producing plots (parameter trajectories, confidence ellipsoids, p/t-tests, estimability evolution) and Excel summaries for documentation and decision-making.


Before Writing Code: Prepare Your Data and Model
=================================================

Files and Directory Structure
-----------------------------

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

**1. Prepare experimental data (data/experimental_data.xlsx)**

Each sheet in ``data.xlsx`` represents one batch/experiment and contains measured outputs (MES), time-invariant controls :math:`\mathbf{w}`, and time-variant controls :math:`\mathbf{u}(t)` in a single table.

data.xlsx structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Column naming convention
~~~~~~~~~~~~~~~~~~~~~~~~

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


**2. Define your mechanistic model (model.py)**

MIDDoE requires a standard model interface. Create a Python file with a function following this signature:

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

**Three model interface options:**

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

Configuration Dictionaries
===========================

The **system** and **models** dictionaries are the central hubs for all MIDDoE operations. They specify experimental design space, model structure, and parameter bounds.

System Dictionary: Defining Time, Controls, and Measurements
------------------------------------------------------------

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

Models Dictionary: Defining Candidate Models and Parameters
-----------------------------------------------------------

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

In-Silico Experimenter and Global Sensitivity Analysis
======================================================

For synthetic validation and preliminary exploration:

.. code-block:: python

   # In-silico experimenter (used for synthetic data generation)
   insilicos = {
       'tr_m': 'M',        # True model (ground truth)
       'theta': theta,     # True parameter vector
       'errt': 'abs',      # Error type:
                           # 'abs' = absolute noise
                           # 'rel' = relative (proportional to signal)
       'prels': {}         # Classic pre-defined designs (empty here)
   }

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

MBDoE Configuration and Optimisation
====================================

Design of Experiments for Parameter Precision (MBDoE-PP)
-------------------------------------------------------

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

Parameter Estimation and Uncertainty Analysis
---------------------------------------------

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

Step-by-Step Workflow: Pharmaceutical CS2 Example
=================================================

This section walks through a complete identification workflow with concrete code.

**Step 0: Prepare Files**

1. Place experimental data in ``data/experimental_data.xlsx``
2. Place your model function in ``models/my_model.py`` following the standard interface
3. Create ``middoe_workflow.py`` with the following steps

**Step 1: Load Data and Define Configuration**

.. code-block:: python

   import pandas as pd
   from pathlib import Path

   # Load experimental data
   data_file = Path('data/experimental_data.xlsx')
   # If multiple sheets (multiple experiments):
   experiments_data = {}
   for sheet_name in pd.ExcelFile(data_file).sheet_names:
       df = pd.read_excel(data_file, sheet_name=sheet_name)
       experiments_data[sheet_name] = df

   # Define system, models, insilicos (as shown in previous sections)
   # ...include all dictionaries from above...

**Step 2: Global Sensitivity Analysis (Optional but Recommended)**

.. code-block:: python

   from middoe.sc_sensa import sensa
   from middoe.log_utils import save_to_jac, load_from_jac, save_to_xlsx

   # Run Sobol analysis
   sobol_results = sensa(gsa, models, system)

   # Save results
   save_to_jac(sobol_results, purpose="sensa")

   # Export to Excel for inspection
   results = load_from_jac()
   sensa_data = results['sensa']
   save_to_xlsx(sensa_data)

   # Interpretation:
   # S_i^(1) = first-order Sobol index (direct effect of parameter i)
   # S_i^T = total-order Sobol index (all interactions involving i)
   # S_i^T >> S_i^(1) = parameter has strong interactions with others

**Step 3: MBDoE Design - Round 1**

.. code-block:: python

   from middoe.des_pp import mbdoe_pp

   # Design optimal experiment for parameter precision
   designs_r1 = mbdoe_pp(
       des_opt,
       system,
       models,
       round=1,
       num_parallel_runs=16  # Parallel optimisation on 16 CPU cores
   )

   # Output contains: optimal controls u(t), initial conditions w, sampling times

**Step 4: Generate Synthetic Data (or Conduct Experiment)**

.. code-block:: python

   from middoe.krnl_expera import expera

   # Generate in-silico data using the designed experiment
   expera(
       system,
       models,
       insilicos,
       designs_r1,
       expr=1,                      # Experiment ID
       swps=designs_r1['swps']       # Switching times from design
   )

   # This generates synthetic data saved as Excel file
   # In practice, you would conduct real experiments with these controls

**Step 5: Parameter Estimation**

.. code-block:: python

   from middoe.iden_parmest import parmest

   # Estimate parameters from data
   pe_results_r1 = parmest(
       system,
       models,
       iden_opt,
       case='round_1'
   )

   # Output: estimated parameters, objective values, convergence info

**Step 6: Uncertainty Quantification**

.. code-block:: python

   from middoe.iden_uncert import uncert

   # Quantify parameter uncertainty
   unc_results_r1 = uncert(
       pe_results_r1,
       system,
       models,
       iden_opt
   )

   results_dict_r1 = unc_results_r1['results']
   obs_r1 = unc_results_r1['obs']

   # Output includes:
   # - Confidence intervals (CI) for each parameter
   # - t-values (precision metric: higher is better, should exceed t_ref ≈ 2.0)
   # - Correlation matrix (high correlations indicate identifiability issues)

**Step 7: Estimability Analysis**

.. code-block:: python

   from middoe.sc_estima import estima

   # Rank parameters by estimability
   ranking, k_optimal, rCC_values, J_k_values, best_result = estima(
       results_dict_r1,
       system,
       models,
       iden_opt,
       round=1
   )

   # Output interpretation:
   # ranking = parameter indices sorted by estimability (best to worst)
   # k_optimal = recommended number of parameters to estimate
   # rCC = corrected critical ratio (lower is better)

   print(f"Parameter ranking by estimability: {ranking}")
   print(f"Recommend estimating top {k_optimal} parameters")

**Step 8: Save Round Results**

.. code-block:: python

   from middoe.log_utils import save_rounds

   round_data = {}
   save_rounds(
       round=1,
       results=results_dict_r1,
       case='preliminary',
       round_data=round_data,
       models=models,
       iden_opt=iden_opt,
       obs=obs_r1,
       system=system,
       ranking=ranking,
       k_optimal_value=k_optimal,
       rCC_values=rCC_values,
       J_k_values=J_k_values,
       best_uncert_result=best_result
   )

**Step 9: Sequential MBDoE Rounds (Repeat Steps 3–8)**

.. code-block:: python

   # Typically, run 3–4 iterative rounds
   for round_num in range(2, 5):
       print(f"\n========== ROUND {round_num} ==========")

       # Update mutation mask based on previous estimability
       # Fix parameters outside top-k from previous round
       models['mutation']['M'] = [i < k_optimal for i in range(len(theta_n))]

       # Design new experiment
       design = mbdoe_pp(des_opt, system, models, round_num, num_parallel_runs=16)

       # Generate data
       expera(system, models, insilicos, design, expr=round_num, swps=design['swps'])

       # Estimate, quantify, analyse
       pe_res = parmest(system, models, iden_opt, case=f'round_{round_num}')
       unc_res = uncert(pe_res, system, models, iden_opt)

       # Estimability
       rank, k_opt, rCC, J_k, best = estima(unc_res['results'], system, models, iden_opt, round_num)

       # Save
       save_rounds(
           round_num, unc_res['results'], f'round_{round_num}', round_data,
           models, iden_opt, unc_res['obs'], system,
           ranking=rank, k_optimal_value=k_opt, rCC_values=rCC,
           J_k_values=J_k, best_uncert_result=best
       )

**Step 10: Cross-Validation**

.. code-block:: python

   from middoe.iden_valida import validation

   # Leave-one-out cross-validation
   validres = validation(system, models, iden_opt, round_data)

   # Output: R² for calibration and validation (should be similar)

**Step 11: Save and Post-Process**

.. code-block:: python

   from middoe.log_utils import save_to_jac, load_from_jac
   from middoe.iden_utils import run_postprocessing

   # Save all round data
   save_to_jac(round_data, purpose="iden")

   # Load and post-process
   results = load_from_jac()
   iden = results['iden']

   # Generate comprehensive reports
   run_postprocessing(
       round_data=iden,
       solvers=['M'],
       selected_rounds=[1, 2, 3, 4],
       plot_global_p_and_t=True,           # Parameter evolution
       plot_confidence_spaces=True,        # Confidence ellipsoids
       plot_p_and_t_tests=True,            # t-value plots
       export_excel_reports=True,          # Excel summaries
       plot_estimability=True              # Estimability improvement
   )

Configuration Reference Tables
==============================

.. _table_param_est_methods:

Parameter Estimation Methods and Cost Functions
-----------------------------------------------

.. list-table::
   :widths: 15 25 25 35
   :header-rows: 1

   * - **Method**
     - **Optimiser**
     - **Type**
     - **When to use**
   * - LBFGSB
     - Limited-memory BFGS
     - Local, gradient-based
     - Fast, smooth problems; default choice
   * - SLSQP
     - Sequential Least Squares
     - Local, gradient-based, constrained
     - Constrained optimisation
   * - DE
     - Differential Evolution
     - Global, population-based
     - Non-convex, multi-modal; slower but robust
   * - NMS
     - Nelder–Mead Simplex
     - Local, derivative-free
     - No gradients available; robust to noise
   * - BFGS
     - Broyden–Fletcher–Goldfarb–Shanno
     - Local, gradient-based
     - Similar to LBFGSB but requires more memory
   * - TC
     - Trust-region Constrained
     - Local, gradient-based, constrained
     - Highly constrained; superior to SLSQP for tight bounds

.. list-table::
   :widths: 15 25 40
   :header-rows: 1

   * - **Cost Function**
     - **Formulation**
     - **When to use**
   * - LS (Least Squares)
     - :math:`\min \sum_t (y_r(t) - \hat{y}_r(t))^2`
     - No noise information available
   * - WLS (Weighted LS)
     - :math:`\min \sum_t \frac{(y_r(t) - \hat{y}_r(t))^2}{\sigma_r^2}`
     - Known, constant measurement uncertainty
   * - MLE (Max Likelihood)
     - :math:`\min -\ln L(\theta \mid \text{data})`
     - Heteroscedastic noise (varying uncertainty)
   * - CS (Chi-Square)
     - :math:`\min \sum_t \frac{(y_r(t) - \hat{y}_r(t))^2 - 2\sigma_r^2}{2\sigma_r^2}`
     - Assumes normal error distribution

.. _table_mbdoe_pp_criteria:

MBDoE-PP Optimality Criteria (Alphabetical)
--------------------------------------------

.. list-table::
   :widths: 20 30 30 20
   :header-rows: 1

   * - **Criterion**
     - **Mathematical Form**
     - **Interpretation**
     - **Advantage / Drawback**
   * - **D-optimality**
     - :math:`\min \det(\mathbf{V}(\boldsymbol{\theta}))`
     - Shrink confidence ellipsoid volume
     - ✓ Scale-invariant; ✗ uneven precision
   * - **A-optimality**
     - :math:`\min \text{tr}(\mathbf{V}(\boldsymbol{\theta}))`
     - Minimise average variance
     - ✓ Balanced precision; ✗ ignores correlations
   * - **E-optimality**
     - :math:`\min \lambda_{\max}(\mathbf{V}(\boldsymbol{\theta}))`
     - Minimise largest eigenvalue (worst-case)
     - ✓ Targets uncertain parameters; ✗ sensitive to ill-conditioning
   * - **ME-optimality**
     - :math:`\min \kappa(\mathbf{V}(\boldsymbol{\theta}))`
     - Minimise condition number
     - ✓ Uniform precision, robust; ✗ limited to small models

.. _table_mdoe_md_criteria:

MBDoE-MD Optimality Criteria (T-optimal)
----------------------------------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - **Criterion**
     - **Focus**
     - **Best for**
   * - **Hunter–Reiner (HR)**
     - Maximise divergence :math:`\max \sum_t (\hat{y}_{l}(t) - \hat{y}_{l'}(t))^T F_{l,l',t} (\hat{y}_{l}(t) - \hat{y}_{l'}(t))`
     - Large prediction differences; late-time discrimination
   * - **Buzzi–Ferraris–Forzatti (BFF)**
     - Weight divergence by error :math:`F_{l,l',t} = \Sigma_y^{-1} + W_{l,t} V_l W_l^T + W_{l',t} V_{l'} W_{l'}^T`
     - Uncertainty-weighted discrimination; early-time focus

Troubleshooting and Guidelines
==============================

Time Resolution and Mesh Independence
--------------------------------------

**Problem:** My results change when I adjust ``t_r``.

**Solution:** Perform a mesh-independency test:

.. code-block:: python

   import numpy as np

   t_r_values = [0.1, 0.05, 0.02, 0.01]
   results = []

   for t_r in t_r_values:
       system['t_r'] = t_r
       # Run MBDoE or parameter estimation
       design = mbdoe_pp(des_opt, system, models, round=1, num_parallel_runs=8)
       results.append(design)  # Store objective values
       print(f"t_r = {t_r}: obj = {design['obj_value']}")

   # If objective values stabilise → mesh-independent
   # If they still change → reduce t_r further

**Guideline:** Use :math:`\Delta t \leq \frac{\tau_{\text{fastest}}}{20}` where :math:`\tau_{\text{fastest}}` is the fastest process time constant (e.g., reaction half-life).

Finite Difference Perturbation
-------------------------------

**Problem:** "FDM sensitivities are noisy" or "convergence is poor".

**Solution:** Use adaptive perturbation with mesh-independency test:

.. code-block:: python

   iden_opt['eps'] = None  # Auto mesh-independency test
   iden_opt['sens_m'] = 'central'  # Use central differences (more accurate)

**Typical range:** :math:`10^{-5} \leq \varepsilon \leq 10^{-3}` in normalised parameter space.

Parameter Correlations and Identifiability
-------------------------------------------

**Problem:** "Parameter estimates have high uncertainty, wide CI, low t-values".

**Solution:** Use Estimability Analysis to identify which parameters are actually estimable:

.. code-block:: python

   from middoe.sc_estima import estima

   ranking, k_opt, rCC, J_k, best = estima(results_dict, system, models, iden_opt, round=1)

   # Fix parameters outside top-k
   for i in range(len(theta_n)):
       models['mutation']['M'][ranking[i]] = i < k_opt

   print(f"Fix parameters: {[ranking[i] for i in range(k_opt, len(ranking))]}")
   print(f"Estimate parameters: {[ranking[i] for i in range(k_opt)]}")

Then re-estimate with reduced parameter subset.

Optimisation Not Converging
----------------------------

**Problem:** "Parameter estimation or MBDoE design doesn't converge".

**Solution:** Try in sequence:

1. **Increase iterations:**

   .. code-block:: python

      iden_opt['maxit'] = 1000  # Double or more
      des_opt['itr']['maxpp'] = 200

2. **Enable multi-start:**

   .. code-block:: python

      iden_opt['ms'] = True  # Try multiple random starts

3. **Switch optimiser:**

   .. code-block:: python

      iden_opt['meth'] = 'DE'  # Global search (slower but robust)
      # or
      des_opt['meth'] = 'DEPS'  # Hybrid: DE + local refine

4. **Relax bounds:**

   .. code-block:: python

      # If bounds are very tight, loosen them slightly
      models['t_l']['M'] = [x * 0.5 for x in theta_mins]
      models['t_u']['M'] = [x * 2.0 for x in theta_maxs]

Exporting and Sharing Results
==============================

All outputs are saved automatically:

- **Internal format (.jac):** Preserves all metadata, use for reloading
- **Excel (.xlsx):** Human-readable summaries, plots, and reports
- **Plots:** PNG files in ``results/`` folder

Access saved data:

.. code-block:: python

   from middoe.log_utils import load_from_jac

   results = load_from_jac()
   iden = results['iden']
   sensa = results['sensa']

   # Access parameter estimates for round 1
   params_r1 = iden['round_1']['estimated_parameters']
   confidence_intervals = iden['round_1']['confidence_intervals']

Citation
--------

If you use MIDDoE in your research, please cite:

   Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
   MIDDoE: An MBDoE Python package for model identification, discrimination,
   and calibration.
   *Digital Chemical Engineering*, 17, 100276.
   https://doi.org/10.1016/j.dche.2025.100276

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   api

Resources
---------

- **GitHub:** https://github.com/zuhairblr/middoe
- **PyPI:** https://pypi.org/project/middoe/
- **Paper:** https://doi.org/10.1016/j.dche.2025.100276
- **Issues & Support:** https://github.com/zuhairblr/middoe/issues
- **Case Studies:** https://github.com/zuhairblr/middoe/tree/main/tests/paper

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`