Key ideas & package structure
=============================

.. rst-class:: md-lead

The ideas behind MIDDoE in five minutes. Play with them in :doc:`interactive figures
<ug_interactive>`; see how the package fits together in :doc:`package_structure`.

The identification loop
-----------------------

A model is useful only when its parameters are known well enough. MIDDoE gets there in **rounds**;
the first experiments are ordinary ("preliminary") ones, every later one is **designed by the model
itself** to be as informative as possible:

.. raw:: html

   <nav class="md-ribbon md-loop" aria-label="The identification loop">
     <a href="ug_setup.html#setup-data"><span class="md-num">1</span>Run an experiment: <code>expera</code> or the lab</a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-rounds"><span class="md-num">2</span>Fit: <code>parmest</code></a><span class="md-sep"></span>
     <a href="ug_results.html"><span class="md-num">3</span>Precise enough? <code>uncert</code></a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-design"><span class="md-num">4</span>Plan the next: <code>mbdoe_pp</code> / <code>mbdoe_md</code></a>
   </nav>

When step 3 says yes, validate (``validation``) and report. The animated version of this loop is in
the :doc:`onboarding guide <user_guide>`.

Glossary
--------

.. tab-set::

   .. tab-item:: :octicon:`beaker` Model & data

      .. rst-class:: md-gloss

      Model
         Equations that predict your process: a Python function ``solve_model``, usually ODEs.
      Parameters θ
         Unknown constants in the model (rate constants, activation energies…): what you want to
         determine.
      Inputs
         What you choose per experiment: **time-variant** ``u(t)`` (a temperature profile) or
         **time-invariant** ``w`` (an initial concentration).
      Outputs y
         What you measure, at chosen **sampling times**, each with an error σ (``unc``).
      Virtual lab
         A simulated experiment with known "true" parameters plus noise (``expera``): test a study
         before the lab.

   .. tab-item:: :octicon:`graph` Fitting & checking

      .. rst-class:: md-gloss

      Parameter estimation
         The parameter values that make the model match the data best (``parmest``).
      Confidence interval
         The range that holds the true value with 95 % probability (``uncert``).
      t-test
         Passes when t = estimate / standard error is above a reference (about 2).
      Sensitivity analysis
         Which parameters influence the outputs at all; the rest can be fixed (``sensa``, Sobol).
      Estimability analysis
         How many, and which, parameters the data you have can support (``estima``).
      P-test
         With rival models: each model's probability of being the right one, in %.
      Cross-validation
         Refit leaving one experiment out and predict it; a high validation R² means the model
         predicts unseen data (``validation``).

   .. tab-item:: :octicon:`goal` Design

      .. rst-class:: md-gloss

      MBDoE-PP
         Plans the experiment that shrinks the parameter uncertainty most (``mbdoe_pp``); criteria
         D (usual), A, E, ME.
      MBDoE-MD
         Plans the experiment where rival models disagree most (``mbdoe_md``); HR on the difference
         alone, BFF also weighing the uncertainty.
      Round
         One pass of run → fit → check → plan; ``save_rounds`` stores it for the reports.

The settings of every step are given in six dictionaries: ``system`` (the experiment), ``models``
(the model and its parameters), ``insilicos`` (the virtual lab), ``iden_opt`` (fitting), ``des_opt``
(design) and ``gsa`` (sensitivity). Which function belongs to which step, and what it reads and writes:
:doc:`package_structure`. Every key: :ref:`config-dicts`.

Go further
----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`play;1.3em` See the ideas at work
      :link: ug_interactive
      :link-type: doc
      :class-card: md-card

      Interactive figures: where to sample (D, A, E), separating rival models, measurement error.

   .. grid-item-card:: :octicon:`stack;1.3em` Package structure
      :link: package_structure
      :link-type: doc
      :class-card: md-card

      The modules, what each one does, and which dictionaries and files it reads and writes.

.. toctree::
   :hidden:

   ug_interactive
   package_structure
