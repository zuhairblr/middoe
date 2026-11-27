Package structure
=================

MIDDoE is a set of modules in five layers. Their names tell the layer: ``krnl_`` kernel, ``iden_``
identification, ``sc_`` screening of the parameters, ``des_`` design, ``log_`` storage. You call a
dozen functions; everything else is internal.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`cpu;1.2em` Kernel
      :class-card: md-card

      ``krnl_simula`` simulates any model (Python file, built-in, gPROMS); every other module uses it.
      ``krnl_expera`` is the virtual lab: ``expera``. ``krnl_models`` holds built-in test models.

   .. grid-item-card:: :octicon:`graph;1.2em` Identification
      :class-card: md-card

      ``iden_parmest``: ``parmest`` fits the parameters. ``iden_uncert``: ``uncert`` gives their
      precision. ``iden_valida``: ``validation``. ``iden_utils``: plots and reports.

   .. grid-item-card:: :octicon:`filter;1.2em` Screening
      :class-card: md-card

      ``sc_sensa``: ``sensa``, global (Sobol) sensitivity before any data. ``sc_estima``:
      ``estima``, which parameters the data can support.

   .. grid-item-card:: :octicon:`goal;1.2em` Design
      :class-card: md-card

      ``des_pp``: ``mbdoe_pp``, the experiment for precision. ``des_md``: ``mbdoe_md``, the
      experiment for discrimination. ``des_utils``: shared helpers.

   .. grid-item-card:: :octicon:`database;1.2em` Storage and reports
      :class-card: md-card

      ``log_utils``: ``save_rounds``, ``save_to_jac``, ``load_from_jac``, ``read_excel``.
      ``iden_utils``: ``run_postprocessing``.

How they connect
----------------

.. raw:: html

   <div class="mdw" data-widget="flow"></div>

.. dropdown:: :octicon:`table` The same as a table

   .. list-table::
      :header-rows: 1
      :widths: 30 34 36

      * - Function
        - Takes
        - Gives, writes
      * - ``sensa(gsa, models, system)``
        - the parameter bounds
        - Sobol indices; ``save_to_jac(..., purpose='sensa')``, ``save_to_xlsx`` store them
      * - ``expera(system, models, insilicos, design, expr, swps)``
        - a design, or ``insilicos['prels']``
        - sheet ``expr`` of ``data.xlsx``; ``data_plots/``
      * - ``parmest(system, models, iden_opt)``
        - ``data.xlsx``
        - the fit of every candidate model
      * - ``uncert(resultpr, system, models, iden_opt)``
        - the fit
        - confidence intervals, t-values, P, fit statistic
      * - ``estima(resultun, system, models, iden_opt, k)``
        - the uncertainty result
        - the parameter ranking and subset; updates ``models['mutation']``; ``estimability/``
      * - ``mbdoe_pp(des_opt, system, models, round)`` · ``mbdoe_md(...)``
        - the current estimates
        - the design (inputs, profiles, sampling times); ``design/``
      * - ``validation(system, models, iden_opt, round_data)``
        - all rounds
        - cross-validation; ``validation/``
      * - ``save_rounds(k, resultun, ...)``
        - one round
        - ``round_data``; ``modelling/``, ``confidence/``
      * - ``save_to_jac`` · ``load_from_jac``
        - ``round_data``
        - ``iden_results.jac``
      * - ``run_postprocessing(round_data, ...)``
        - all rounds
        - ``post_processing/``: plots and Excel across rounds

Every key of every dictionary is in the :ref:`reference <config-dicts>`; every argument of every
function in the :doc:`api`.
