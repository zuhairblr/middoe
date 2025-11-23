MIDDoE Documentation
====================

.. image:: https://img.shields.io/pypi/v/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/zuhairblr/middoe.svg
   :target: https://github.com/zuhairblr/middoe/blob/main/LICENSE
   :alt: License

**MIDDoE** (Model-based Identification, Discrimination, and Design of Experiments) is an open-source Python package for comprehensive model identification workflows in dynamic systems. Developed to address the fragmentation of existing MBDoE tools, MIDDoE integrates the complete pipeline from sensitivity analysis to experimental design, parameter estimation, and model validation within a unified, user-friendly framework.

Overview
--------

Mathematical modelling plays a critical role in the design, optimisation, and control of dynamic systems in the process industry. While mechanistic models offer strong explanatory and predictive power, their effectiveness depends on informed model selection and precise parameter calibration.

**Model-Based Design of Experiments (MBDoE)** provides a systematic framework for addressing these challenges by designing experiments that accelerate model discrimination and improve parameter precision. However, practical application has been constrained by fragmented digital tools that lack integration, making MBDoE implementation accessible only to expert users.

**MIDDoE solves this problem** by providing:

- **Complete MBDoE workflows** for both model discrimination (MBDoE-MD) and parameter precision (MBDoE-PP)
- **Integrated analysis tools** including Global Sensitivity Analysis, Estimability Analysis, parameter estimation, uncertainty quantification, and cross-validation
- **Physics-aware optimization** with support for physical constraints and non-convex design spaces
- **Flexible simulation backends** compatible with both built-in and external simulators (including gPROMS)
- **User-friendly interface** designed for both experimentalists and advanced users

Key Features
------------

**Comprehensive Workflow Support**

- Global Sensitivity Analysis (Sobol' indices with Jansen's estimator)
- Parameter Estimation (SLSQP, DE, NMS, BFGS, LMBFGS, TC optimizers)
- Uncertainty Quantification (Hessian-based, Jacobian-based, Bootstrap)
- Estimability Analysis (orthogonalisation procedure with corrected critical ratio)
- MBDoE for Model Discrimination (Hunter-Reiner, Buzzi-Ferraris-Forzatti T-optimality)
- MBDoE for Parameter Precision (D, A, E, ME-optimality criteria)
- Cross-Validation (leave-one-out for model generalization assessment)

**Advanced Experimental Design**

- Control Vector Parameterisation (CPF: piecewise-constant, LPF: piecewise-linear)
- Physical and operational constraints (sampling limits, switching constraints)
- Non-convex optimization with global-local strategies (DE, PS, DEPS)
- Parallel execution for computationally intensive tasks

**Flexible Architecture**

- Decoupled simulation from analysis (kernel-logic-client layers)
- Compatible with built-in DAE solvers and external simulators
- Standard Python interfaces for easy integration
- Modular design enabling customization at multiple levels

Installation
------------

Install MIDDoE via pip:

.. code-block:: bash

   pip install middoe

Requirements: Python 3.9+

Quick Start Examples
--------------------

**Example 1: Global Sensitivity Analysis**

.. code-block:: python

   from middoe import sc_sensa

   # Configure Global Sensitivity Analysis
   gsa = {
       'samp': 2**10,          # Sample size (1024 base samples)
       'par_s': True,          # Sensitivity w.r.t. parameters
       'var_s': True,          # Sensitivity w.r.t. control variables
       'par_d': False,         # Full parameter space (not damped)
       'var_d': 1.1,           # Damped variable space (±10% around nominal)
       'plt': True,            # Generate plots
       'multi': 0.7            # Use 70% of CPU cores for parallel execution
   }

   # Run Sobol' analysis
   results = sc_sensa.sensa(gsa, system, models)

   # Access first-order and total-order indices
   S1 = results['analysis']['M1']['CA'][0]['S1']  # First-order indices
   ST = results['analysis']['M1']['CA'][0]['ST']  # Total-order indices

**Example 2: Parameter Estimation with Uncertainty Quantification**

.. code-block:: python

   from middoe import iden_parmest, iden_uncert

   # Configure parameter estimation
   iden_opt = {
       'meth': 'SLSQP',       # Sequential Least Squares Programming
       'ob': 'WLS',           # Weighted Least Squares objective
       'ms': False,           # Single-start optimization
       'init': None,          # Use nominal initial values
       'sens_m': 'central',   # Central finite difference
       'var-cov': 'J',        # Jacobian-based uncertainty
       'eps': None,           # Automatic mesh-independency test
       'log': True            # Enable verbose logging
   }

   # Perform parameter estimation
   results_pe = iden_parmest.parmest(system, models, iden_opt)

   # Quantify uncertainty
   results_uncert = iden_uncert.uncert(results_pe, system, models, iden_opt)

   # Access results
   theta_est = results_uncert['results']['M1']['estimations']
   R2 = results_uncert['results']['M1']['R2_total']
   CI = results_uncert['results']['M1']['CI']  # 95% confidence intervals

   print(f"R² = {R2:.4f}")
   print(f"Estimated parameters: {theta_est}")
   print(f"95% CI: ±{CI}")

**Example 3: Estimability Analysis for Parameter Subset Selection**

.. code-block:: python

   from middoe import sc_estima

   # Configure estimability analysis
   iden_opt = {
       'meth': 'SLSQP',
       'var-cov': 'J',
       'log': False
   }

   # Perform estimability analysis
   rankings, k_opt, rCC_values, J_k = sc_estima.estima(
       result=results_uncert,  # From previous uncertainty analysis
       system=system,
       models=models,
       iden_opt=iden_opt,
       round=1
   )

   # Results
   print(f"Parameter ranking (most→least estimable): {rankings['M1']}")
   print(f"Optimal number of parameters: {k_opt['M1']}")
   print(f"Selected subset: {rankings['M1'][:k_opt['M1']]}")

**Example 4: MBDoE for Parameter Precision (MBDoE-PP)**

.. code-block:: python

   from middoe import des_pp

   # Configure MBDoE-PP with D-optimality
   des_opt = {
       'ppob': 'D',                # D-optimality criterion
       'meth': 'DEPS',             # DE + Pattern Search (global-local)
       'itr': {
           'maxpp': 5000,          # Maximum iterations
           'toldpp': 1e-8,         # Convergence tolerance
           'pps': 40               # Population size
       },
       'eps': 0.01,                # Finite difference perturbation
       'plt': True                 # Generate design plots
   }

   # Run MBDoE-PP
   design = des_pp.mbdoepp(des_opt, system, models, round=2)

   # Access optimal design
   print(f"Optimal time-invariant inputs: {design['tii']}")
   print(f"Optimal time-variant profiles: {design['tvi']}")
   print(f"Optimal sampling times: {design['St']}")
   print(f"D-optimality value: {design['pp_obj']:.4e}")

**Example 5: MBDoE for Model Discrimination (MBDoE-MD)**

.. code-block:: python

   from middoe import des_md

   # Configure MBDoE-MD with Hunter-Reiner criterion
   des_opt = {
       'mdob': 'HR',               # Hunter-Reiner T-optimality
       'meth': 'DEPS',
       'itr': {
           'maxmd': 5000,
           'toldmd': 1e-8,
           'pps': 50
       },
       'eps': 0.01,
       'plt': True
   }

   # Run MBDoE-MD with 4 competing models
   design = des_md.mbdoemd(des_opt, system, models, round=1, num_parallel_runs=4)

   print(f"Discriminative design controls: {design['tvi']}")
   print(f"HR discrimination value: {design['md_obj']:.4e}")

Workflow
--------

MIDDoE implements a systematic model identification workflow:

1. **Global Sensitivity Analysis** → Identify influential parameters and inputs
2. **Preliminary Experiments** → Collect initial data for model calibration
3. **Parameter Estimation** → Calibrate candidate models
4. **Estimability Analysis** → Select identifiable parameter subset
5. **MBDoE-MD** (if needed) → Design experiments for model discrimination
6. **MBDoE-PP** (if needed) → Design experiments for parameter precision
7. **Cross-Validation** → Assess model generalization capability
8. **Model Selection & Reporting** → Select final model with validated parameters

This iterative workflow ensures robust model identification with minimal experimental effort.

Case Studies
------------

MIDDoE has been successfully applied to:

- **Case Study 1 - Bioprocess**: Fed-batch fermentation for enzyme production with 4 competing models. Successful model discrimination using MBDoE-MD (HR and BFF criteria).

- **Case Study 2 - Pharmaceutical synthesis**: Batch reaction kinetics with 6 parameters. Sequential MBDoE-PP improved estimable parameters from 3 to 4, demonstrating the power of iterative experimental design.

See the `tests/paper directory <https://github.com/zuhairblr/middoe/tree/main/tests/paper>`_ for detailed case studies with complete code and data.

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
   :caption: User Guide

   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
=====

- **GitHub Repository**: https://github.com/zuhairblr/middoe
- **PyPI Package**: https://pypi.org/project/middoe/
- **Issue Tracker**: https://github.com/zuhairblr/middoe/issues

License
=======

MIDDoE is released under the MIT License.

Support
=======

For questions, issues, or contributions, open an issue on `GitHub <https://github.com/zuhairblr/middoe/issues>`_ or contact the development team at CAPE-Lab, University of Padova.
