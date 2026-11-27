Key ideas
=========

A five-minute introduction to the ideas behind MIDDoE. The formulas are in the :doc:`user_guide`.

The identification loop
-----------------------

A model is only useful if its parameters are known well enough. MIDDoE improves the model in
**rounds**:

.. code-block:: text

   1. Plan an experiment     (MBDoE: mbdoe_pp / mbdoe_md)
   2. Run it                 (in the lab, or in the virtual lab: expera)
   3. Fit the model          (parmest, then uncert)
   4. Check precision        good enough?  no  -> back to 1 (next round)
                                           yes -> 5. Validate (validation) and report

Before the first round, a few ordinary ("preliminary") experiments give a first estimate of the
parameters. Each later experiment is **designed by the model itself** to be as informative as
possible.

Glossary
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Term
     - Meaning
   * - Model
     - Equations that predict how your process behaves. In MIDDoE: a Python function
       ``solve_model`` (usually differential equations).
   * - Parameters (θ)
     - Unknown constants in the model (rate constants, activation energies, …). They are the
       quantities you want to determine.
   * - Inputs (controls)
     - What you choose in an experiment. **Time-variant** inputs ``u(t)`` change during the run
       (e.g. a temperature profile); **time-invariant** inputs ``w`` are fixed per run (e.g. the
       initial concentration).
   * - Outputs (responses, y)
     - What you measure, at chosen **sampling times**. Each measurement has an error (σ, ``unc``).
   * - Virtual lab (in-silico experiment)
     - A simulation of an experiment with known "true" parameters plus random noise. Use it to test a
       study before doing real experiments (``expera``).
   * - Parameter estimation
     - Finding the parameter values that make the model match the measured data best
       (``parmest``).
   * - Confidence interval
     - The range in which the true parameter value lies with 95 % probability (``uncert``).
   * - t-test
     - A precision check. A parameter **passes** if its t-value (estimate divided by its standard
       error) is larger than a reference value (about 2). A parameter that fails is not determined
       well by the data.
   * - Sensitivity analysis (GSA)
     - Which parameters actually influence the outputs? Parameters with no influence cannot be
       estimated and can be fixed (``sensa``, Sobol method).
   * - Estimability analysis
     - With the data you have now, which parameters can be estimated reliably, and how many? The
       rest are kept fixed for now (``estima``).
   * - MBDoE-PP (parameter precision)
     - Plans the experiment that shrinks the parameter uncertainty most (``mbdoe_pp``). The
       criteria D, A, E and ME are different ways of measuring "uncertainty"; **D** is the usual
       choice.
   * - MBDoE-MD (model discrimination)
     - With several rival models, plans the experiment where their predictions differ most, so the
       data can tell which one is right (``mbdoe_md``). **HR** looks at the difference alone;
       **BFF** also accounts for measurement and parameter uncertainty.
   * - P-test
     - After fitting rival models, gives each one a probability (in %) of being the best. Models with
       a low P are discarded.
   * - Cross-validation
     - Refit the model leaving one experiment out each time and predict that experiment. A high
       validation R² means the model also predicts data it has not seen (``validation``).
   * - Round
     - One pass through plan → run → fit → check. ``save_rounds`` stores each round for the final
       reports.

Which function for which step
-----------------------------

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Step
     - Function
     - Module
   * - Sensitivity analysis
     - ``sensa``
     - ``middoe.sc_sensa``
   * - Virtual-lab experiment
     - ``expera``
     - ``middoe.krnl_expera``
   * - Plan an experiment (precision)
     - ``mbdoe_pp``
     - ``middoe.des_pp``
   * - Plan an experiment (discrimination)
     - ``mbdoe_md``
     - ``middoe.des_md``
   * - Fit parameters
     - ``parmest``
     - ``middoe.iden_parmest``
   * - Precision, t-test, P-test
     - ``uncert``
     - ``middoe.iden_uncert``
   * - Estimability analysis
     - ``estima``
     - ``middoe.sc_estima``
   * - Cross-validation
     - ``validation``
     - ``middoe.iden_valida``
   * - Store rounds, save and load results
     - ``save_rounds``, ``save_to_jac``, ``load_from_jac``
     - ``middoe.log_utils``
   * - Final plots and Excel reports
     - ``run_postprocessing``
     - ``middoe.iden_utils``

The settings for all steps are given in six dictionaries: ``system`` (the experiment), ``models``
(the model and its parameters), ``insilicos`` (the virtual lab), ``iden_opt`` (fitting), ``des_opt``
(design) and ``gsa`` (sensitivity). All their options are listed in :ref:`config-dicts`.
