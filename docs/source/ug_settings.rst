.. _choosing-settings:

Choose the settings
===================

.. include:: _ribbon.rst

.. rst-class:: md-lead

Five choices cover most studies; everything else can keep its default. Pick a tab.

``iden_opt['meth']`` and ``iden_opt['ob']`` are required.

.. tab-set::

   .. tab-item:: :octicon:`pulse` Error & objective

      σ (the standard deviation of each measurement) weights the fit and sets the width of every
      confidence interval, so it must be realistic.

      .. list-table::
         :header-rows: 1
         :widths: 28 40 32

         * - Your measurement error
           - Give MIDDoE
           - ``iden_opt['ob']``
         * - same σ for every point (absolute)
           - σ in ``MES_E`` or ``unc``; in silico ``'errt': 'abs'``
           - ``'WLS'`` (``'MLE'`` gives the same estimates)
         * - proportional to the value (relative)
           - each point's σ in ``MES_E``; in silico ``'errt': 'rel'``
           - ``'WLS'``; ``'Chi'`` if you have no per-point σ
         * - unknown
           - an estimate from replicates; otherwise any positive placeholder (``unc = 1``)
           - ``'LS'`` with ``var-cov='H'``

      :octicon:`play` :ref:`See absolute vs relative error and the floor, interactively <ia-noise>`

   .. tab-item:: :octicon:`zap` Optimiser

      All methods reach the same answer when there is one minimum; they differ in cost and
      robustness.

      .. list-table::
         :header-rows: 1
         :widths: 46 30 24

         * - Situation
           - ``meth``
           - ``ms``
         * - smooth model, reasonable start (most cases)
           - ``'SLSQP'`` :bdg-success:`fast` or ``'LMBFGS'``
           - ``False``
         * - unsure about the start; several minima possible
           - ``'LMBFGS'``
           - ``True`` :bdg-info:`robust` (random starts inside the bounds, best kept)
         * - no idea of the values; cheap model
           - ``'DE'`` :bdg-warning:`slow`
           - —
         * - noisy or non-smooth model output
           - ``'NMS'``
           - optional

      .. admonition:: Keep ``tol`` tight
         :class: warning

         ``iden_opt['tol']`` between 1e-6 and 1e-9. A loose tolerance stops early and *still
         reports success*.

   .. tab-item:: :octicon:`shield-check` Uncertainty

      Set ``iden_opt['var-cov']`` **before** ``parmest``: ``'H'`` and ``'B'`` are computed there.

      .. list-table::
         :header-rows: 1
         :widths: 14 46 40

         * - ``var-cov``
           - Method
           - Use when
         * - ``'M'``
           - Fisher information, linearised :bdg-success:`fast`
           - σ is known: the usual choice
         * - ``'H'``
           - inverse Hessian of the objective :bdg-success:`fast`
           - ``'LS'``/``'Chi'`` objectives; cross-check of ``'M'``
         * - ``'B'``
           - bootstrap, ``nboot`` refits :bdg-warning:`slow`; the estimates are the mean of the refits
           - non-linear models, few data. ``nboot`` 20–50 to check, 100–500 to report

   .. tab-item:: :octicon:`goal` Design

      .. list-table::
         :header-rows: 1
         :widths: 20 80

         * - ``des_opt``
           - Designs the experiment that ...
         * - ``pp_ob='D'``
           - shrinks the volume of the joint confidence region (first choice)
         * - ``'A'`` / ``'E'`` / ``'ME'``
           - shrinks the average variance / the worst direction / the gap between best and worst
             direction
         * - ``md_ob='BFF'``
           - separates rival models, weighted by their uncertainty (recommended)
         * - ``md_ob='HR'``
           - separates rival models by the plain difference of their predictions
         * - ``meth``
           - ``'PS'`` pattern search :bdg-success:`fast` (default); ``'DE'`` or ``'DEPS'`` (DE, then
             pattern search) search more widely :bdg-warning:`slow`. ``num_parallel_runs=N`` starts
             N independent searches and keeps the best.

      :octicon:`play` Interactively: :ref:`what D, A and E do to the confidence region <ia-sampling>` ·
      :ref:`how a design separates rival models <ia-discrim>`

   .. tab-item:: :octicon:`filter` Parameters

      Start with all parameters free (``mutation`` all ``True``). If some have very wide intervals or
      correlations near ±1:

      - run ``estima`` to find the subset the data can support, or
      - design a better experiment with ``mbdoe_pp``.

      ``estima`` keeps its choice in ``models['mutation']``, and the parameters it fixes take their
      starting-guess value. Free them again before the next round if every round should re-rank all
      of them, or set ``iden_opt['model_update'] = False`` to leave ``models`` unchanged.

.. dropdown:: :octicon:`gear` Advanced keys (keep the defaults)

   ``pscale``, ``grad``, ``grad_step``, ``fnorm``, ``boot_start``, ``sens_m``, ``eps``: see the notes
   under :ref:`config-dicts`.
