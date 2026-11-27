.. _user-guide:

################
Onboarding guide
################

From installation to your own study, in five steps. Take them in order; each page ends with a button
to the next one.

.. grid:: 1
   :gutter: 2
   :class-container: md-path

   .. grid-item-card:: 1 · Get started
      :link: getting_started
      :link-type: doc
      :class-card: md-card md-step

      Install, then run a complete study in 15 minutes.

   .. grid-item-card:: 2 · Set up
      :link: ug_setup
      :link-type: doc
      :class-card: md-card md-step

      Your model file, the dictionaries, your data, the rounds.

   .. grid-item-card:: 3 · Settings
      :link: ug_settings
      :link-type: doc
      :class-card: md-card md-step

      The five choices that matter, and what to pick.

   .. grid-item-card:: 4 · Results
      :link: ug_results
      :link-type: doc
      :class-card: md-card md-step

      Is the model good enough? Where every file is.

   .. grid-item-card:: 5 · Reference
      :link: ug_reference
      :link-type: doc
      :class-card: md-card md-step

      Every key of every dictionary, when you need it.

One round of a study
====================

Everything in MIDDoE happens in **rounds**: each round adds an experiment and refits the model, until
the parameters are precise enough. Press play to watch three rounds, or click a step to see what it
does, the function you call and the idea behind it.

.. raw:: html

   <div class="mdw" data-widget="loop"></div>

.. button-ref:: getting_started
   :ref-type: doc
   :color: primary
   :shadow:
   :class: md-next

   Start with step 1: get started  →

.. _tutorials:

When you want more examples
===========================

Notebooks that go further than this guide. Each link opens it in Google Colab, ready to run in
the browser; the files are in ``tests/tutorials`` and ``tests/middoe_paper`` of the repository.

- **Tutorial 1** (`open in Colab <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_estimation_and_uncertainty/tutorial.ipynb>`__): every
  dictionary key, optimiser, objective and uncertainty method explained and compared on one model.
- **Tutorial 1, short versions**, one notebook per setting:
  `A <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/a_relative_fisher_fast/runner.ipynb>`__ relative noise, Fisher matrix, fast optimiser ·
  `B <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/b_relative_fisher_best/runner.ipynb>`__ with multi-start ·
  `B2 <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/b2_relative_bootstrap_best/runner.ipynb>`__ as B with the bootstrap ·
  `C <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/c_absolute_hessian_estimability/runner.ipynb>`__ absolute noise, Hessian, estimability ·
  `D <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/d_absolute_bootstrap_mle/runner.ipynb>`__ absolute noise, bootstrap, maximum likelihood.
- :doc:`case_studies`: the four case studies of the MIDDoE paper, step by step: model
  discrimination (HR and BFF), calibration from available data, and sequential design.

.. toctree::
   :hidden:

   getting_started
   ug_setup
   ug_settings
   ug_results
   ug_reference
