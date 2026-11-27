.. _reading-results:

Read the results
================

.. include:: _ribbon.rst

.. rst-class:: md-lead

Six checks tell you whether the model is good enough. Then: where every file is, and what to do when
something looks wrong.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`check-circle;1.2em` t-test
      :class-card: md-card

      Every estimated parameter has **t > t**\ :sub:`ref` (about 2). Printed by ``save_rounds``;
      plot ``*_pt_tests.png``.

   .. grid-item-card:: :octicon:`meter;1.2em` Fit statistic
      :class-card: md-card

      :math:`\sum[(y-\hat y)/\sigma]^2/(N-p)` inside its printed 95 % range (≈ 1). Above: model or σ
      wrong. Below: σ too large.

   .. grid-item-card:: :octicon:`pulse;1.2em` Residuals
      :class-card: md-card

      Weighted residuals around 0, mostly within ±2, **no trend** in time (a trend = model error).

   .. grid-item-card:: :octicon:`git-compare;1.2em` Correlations
      :class-card: md-card

      Not close to ±1 (upper cells of the confidence plots). Near ±1: the pair is only determined
      together.

   .. grid-item-card:: :octicon:`trophy;1.2em` Model probability P
      :class-card: md-card

      With rival models: one model clearly ahead.

   .. grid-item-card:: :octicon:`graph;1.2em` Over the rounds
      :class-card: md-card

      Confidence intervals shrink towards the precision you need (``*_parameter_estimates.png``).

See the residual check at work: pick a case and watch the points and the fit statistic.

.. raw:: html

   <div class="mdw" data-widget="residuals"></div>

Where everything is written
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder / file
     - Content
   * - ``data.xlsx``
     - the measurements, one sheet per experiment
   * - ``data_plots/``
     - preview of each in-silico experiment: data, model and noise
   * - ``modelling/`` · ``confidence/``
     - per round: fit, residual and parity plots · 95 % confidence regions
   * - ``design/``
     - designed experiments: figure and Excel per core (``mbdoe_pp`` also a text summary)
   * - ``estimability/`` · ``validation/`` · ``meshindep_plots/``
     - rCC curves · cross-validation · finite-difference step test
   * - ``iden_results.jac``
     - all rounds; reload with ``load_from_jac()``
   * - ``post_processing/``
     - the final report: plots across rounds, Excel per round, a summary

.. _pitfalls:

Something went wrong?
---------------------

.. dropdown:: The script restarts itself or fails with a multiprocessing error
   :icon: alert

   Put the study inside ``def main(): ...`` and call it under ``if __name__ == '__main__':``, on
   every operating system. Notebooks do not need this.

.. dropdown:: Old experiments appear in my in-silico study
   :icon: alert

   ``expera`` adds sheets to an existing ``data.xlsx``. Delete it (and the output folders) before
   starting again; re-running round ``k`` replaces sheet ``k``.

.. dropdown:: "Success", but the estimates look wrong
   :icon: alert

   The tolerance is probably too loose: keep ``iden_opt['tol']`` at 1e-6 or tighter, and compare the
   objective values of two runs.

.. dropdown:: The confidence intervals are suspiciously narrow or wide
   :icon: alert

   They scale with σ. Check ``MES_E`` (or ``unc``) and the fit statistic: inside its range means σ
   and the model agree with the data.

.. dropdown:: A parameter is no longer estimated
   :icon: alert

   ``estima`` writes its selection into ``models['mutation']``, and the parameters it fixes take
   their starting-guess value. Set the mask back to all ``True`` before ``parmest`` to re-rank all
   parameters, or use ``iden_opt['model_update'] = False``.

.. dropdown:: No figures appear
   :icon: alert

   Figures are always saved; they are only displayed in a notebook or a window. In batch runs use
   ``run_postprocessing(..., show=False)`` and ``iden_opt['plt_s'] = False``.
