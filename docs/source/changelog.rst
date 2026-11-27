Changelog
=========

0.2.1
-----

Documentation and package information only; the code is the same as in 0.2.0.

- New front page and a :doc:`tutorials` page: five notebooks on two real systems, each with an
  animation of what it does; they open in Google Colab.
- The package description, keywords, classifiers and links on PyPI now say what MIDDoE is for, and
  the README starts with what it does and when to use it.
- ``llms.txt``: a plain-text summary of MIDDoE at the root of the documentation site.
- The MIDDoE logo is served from the documentation itself.

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
  does) instead of σ = 1, and both warn when this happens. Empty ``MES_E`` cells are filled the
  same way (before, they stopped the fit with an unclear error).
- **Missing ``'unc'``:** ``expera`` stops with a clear message naming the output (before: a bare
  ``KeyError``), and a design warns that it assumes σ = 1 for that output.

Estimation and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``var-cov='H'`` works for every optimiser: the Hessian is taken of the real objective, in
  whitened coordinates, and scaled correctly for each objective.
- Bootstrap (``var-cov='B'``): pairs resampling; truncated-normal correction only at bounds that are
  actually hit; replicates whose objective is more than twice the reference fit are rejected, and
  the kept/rejected counts are printed and stored (``boot_counts``). Each replicate starts from the
  reference estimate plus 4 random starts and keeps the best fit (new option ``boot_start``,
  default ``'mixed'``; ``'multi'`` restores a full multi-start per replicate). On case study 2
  this finds the same replicate optima as the full multi-start about 4 times faster.
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
- ``expera`` shows a small figure of each new data set and its noise (``insilicos['plt']``,
  default on; saved in ``./data_plots/``).
- gPROMS models (``models['krt'] = 'gpr'``) are no longer loaded as a Python file before the
  simulation, which made every gPROMS run fail.
- ``des_opt['conv_rep']`` now defaults to ``False``; the per-iteration design metrics (FIM metrics and
  a curvature probe, about as many extra model evaluations as the optimisation itself) are computed
  only when it is ``True``. Designs are unchanged; MBDoE-PP runs about twice as fast by default.
- The model file and the data file are checked, with clear messages for missing or invalid
  entries.
- ``save_rounds`` accepts ``data=``, like ``parmest`` and ``uncert``.
- ``run_postprocessing(..., show=False)`` saves the figures without opening a window, so scripts
  no longer stop at every figure.

Plots
~~~~~

- Confidence ellipses are drawn at the 95 % joint level (they were 1σ, about 39 %, next to 95 %
  intervals), each parameter on its own axis range; correlation coefficients are shown, and the
  bootstrap replicates for ``var-cov='B'``.
- The parameter-estimate plot shows the physical estimates ± 95 % CI (it plotted normalised values
  against a "true value = 1" line); optional true values with ``run_postprocessing(true_theta=...)``.
- The t-test plot maps t-values to the right parameter when the estimated subset changes, the
  design-type shading works for the labels used with ``save_rounds``, and the P plot is skipped
  for a single model. Relative standard deviations replace absolute ones; a change in the number
  of estimated parameters is marked on the volume plot. Rounds with ``var-cov='B'`` are no longer
  left out of these plots.
- Fit plots show error bars (±2σ), a 95 % prediction band, weighted residuals with
  :math:`\sum r^2/(N-p)`, one panel per output and per input; a parity plot is added.
- Excel reports add σ, model values at the sampling times, residuals and weighted residuals, and a
  summary sheet; the text summary has a parameter table and the fit statistics.
- Design figures have one panel per output and input; with ``des_opt['plt']`` the best core's
  figure is displayed by the main process (displaying from the worker processes did not work and
  needed IPython).
- ``estima``, ``validation`` and the step-size test use the same style; ``iden_opt['plt_s']``
  controls their display. The validation summary no longer fails when all folds give the same
  value, and the MSE plot no longer labels its reference "R²".
- ``save_rounds`` stores copies of ``models``, ``iden_opt``, ``system`` and the masks, so later
  rounds cannot change earlier ones; a failing plot only gives a warning.
- Plotting no longer changes the global matplotlib settings; the optional
  ``models['theta_names']`` labels the parameters.

Compatibility
~~~~~~~~~~~~~

- The option names used in the paper are accepted as aliases: ``'G'``, ``'Ls'``, ``'Ln'``,
  ``'SQP'``, ``'NM'``, ``'LBFGSB'`` (methods), ``'CS'`` (Chi), ``'J'`` (``var-cov='M'``), and the
  older ``des_opt`` method names.
- ``numdifftools`` is now declared as a dependency.

Documentation
~~~~~~~~~~~~~

- An :doc:`onboarding guide <user_guide>` in five steps, from :doc:`getting_started` (a complete
  study in a minute) to the reference, with interactive figures; :doc:`concepts`; the whole page
  tree in the side menu.
- :doc:`tutorials`: the four case studies of the MIDDoE paper as notebooks, one folder per case
  with only the notebook and its model files. They and the tutorials open in Google Colab.
- The :doc:`api` starts with the functions you call, in workflow order, and how to read their
  results.
- Formulas in the docstrings are rendered as mathematics.
