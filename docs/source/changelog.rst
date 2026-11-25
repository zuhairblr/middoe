Changelog
=========

0.2.0
-----

This release makes the code match the methods of the MIDDoE paper (Tabrizi et al., *Digital
Chemical Engineering* 17 (2025) 100276), fixes a number of bugs, and makes parameter estimation
more robust. **Results can differ from 0.1.x**; the most important changes in behaviour are listed
first.

Changes in behaviour
~~~~~~~~~~~~~~~~~~~~

- **Objectives as in the paper.** ``parmest`` minimises the true sums of eqs. S5–S7 with the
  measurement error of each point, and ``uncert`` reports the same values. ``'MLE'`` is fitted
  without its constant term (same optimum), but the reported value includes it.
- **``iden_opt['tol']`` is a relative tolerance** on the objective for every optimiser. Use a
  tight value such as ``1e-8``: a loose one (e.g. ``0.1``) stops the fit early and gives wrong
  estimates and confidence intervals.
- **Design priors are off by default** (``des_opt['priors'] = False``). When switched on, the prior
  covariance is converted to the design's scaled units correctly, and a model without a covariance
  gets no prior term (previously a fake ``1e-50·I`` dominated the design).
- **Sobol first-order indices** use the Jansen estimator, as in the paper.
- **'Chi' objective:** the relative error is divided by ``max(|y|, σ)`` instead of ``|y|``, so a
  measurement near zero no longer dominates the fit.
- **``case=None`` keeps existing ``mutation`` masks** in ``parmest`` (``estima`` relies on this).
- **Missing ``MES_E`` column:** ``uncert`` now uses the system ``'unc'`` value (as ``parmest``
  does) instead of σ = 1, and both warn when this happens.

Estimation and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``var-cov='H'`` works for every optimiser: the Hessian is taken of the real objective, in
  whitened coordinates, and scaled correctly for each objective.
- Bootstrap (``var-cov='B'``): pairs resampling; truncated-normal correction only at bounds that are
  actually hit; replicates whose objective is more than twice the reference fit are rejected, and
  the kept/rejected counts are printed and stored (``boot_counts``).
- The optimiser works in log-scaled parameters (where the bounds allow it), uses a gradient step
  suited to the scaled problem and a normalised objective. New options ``pscale``, ``grad``,
  ``grad_step`` and ``fnorm`` (see the user guide) restore the previous behaviour if needed.
- Multi-start keeps fixed parameters at the same values as a single start.
- Failed fits raise an error instead of returning an empty result; errors raised by your model
  stop the run with a ``SimulationError`` that keeps your model's message.
- NaN values in the data or in the model output no longer make ``parmest`` hang.
- Measurements are matched to the simulation time grid exactly; a time that is not on the grid
  raises an error instead of being silently shifted.
- Conditioning diagnostics in ``uncert`` no longer depend on parameter units.
- Estimability ranking (``estima``) uses scaled sensitivities (paper S10), so it no longer depends
  on parameter units. ``estima`` and ``validation`` no longer crash, and ``estima`` restores
  ``iden_opt['log']``.
- The degrees of freedom count only the estimated parameters.

Design of experiments
~~~~~~~~~~~~~~~~~~~~~

- A-optimality criterion corrected.
- BFF criterion (``mbdoe_md``) uses the designed sampling times and the estimated parameter
  covariance, as in the paper.
- The P-test follows the paper's intended formula.
- Dead time ``t_d`` and sampling-time constraints are applied in model-discrimination designs.
- Input bounds are validated (``min <= max``, ``max > 0``, numeric).
- A design that finds no feasible solution raises a clear error that lists what to check.

Experiments, data and models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Data generation and fitting use exactly the same piecewise-constant input profile.
- ``expera``: outputs with ``'meas': False`` work, re-running a round replaces its sheet, and
  time-invariant outputs are written with noise. New option ``sigma_floor`` keeps relative errors
  above zero where the output is zero.
- The model file and the data file are checked, with clear messages for missing or invalid
  entries.
- ``save_rounds`` accepts ``data=``, like ``parmest`` and ``uncert``.
- ``run_postprocessing(..., show=False)`` saves the figures without opening a window, so scripts
  no longer stop at every figure.

Compatibility
~~~~~~~~~~~~~

- The option names used in the paper are accepted as aliases: ``'G'``, ``'Ls'``, ``'Ln'``,
  ``'SQP'``, ``'NM'``, ``'LBFGSB'`` (methods), ``'CS'`` (Chi), ``'J'`` (``var-cov='M'``), and the
  older ``des_opt`` method names.
- ``numdifftools`` is now declared as a dependency.

Documentation
~~~~~~~~~~~~~

- New pages: :doc:`getting_started` (a complete study in a minute), :doc:`concepts` and
  :doc:`case_studies` (the paper's case studies, ready to run).
- The :doc:`api` starts with the functions you call, in workflow order, and how to read their
  results.
- Formulas in the docstrings are rendered as mathematics.
