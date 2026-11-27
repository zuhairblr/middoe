.. _case-studies:

Paper case studies
==================

.. rst-class:: md-lead

The four case studies of the MIDDoE paper, as notebooks you can run: each cell is explained, every
setting follows the paper.

They are in `tests/middoe_paper <https://github.com/zuhairblr/middoe/tree/main/tests/middoe_paper>`_.
Each folder holds only the notebook (``runner.ipynb``) and its model file(s); everything a run writes
appears next to the notebook.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`git-compare;1.2em` 1 · Model discrimination
      :class-card: md-card

      Four rival kinetic models. A preliminary round, the P-test, one experiment designed with
      ``mbdoe_md`` to separate the survivors, and the final choice.
      Two folders: ``scenario_HR`` (Hunter–Reiner) and ``scenario_BFF`` (Buzzi-Ferraris–Forzatti).

      :octicon:`play` `HR · Colab <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_HR/runner.ipynb>`__ · `view <https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_HR/runner.ipynb>`__

      :octicon:`play` `BFF · Colab <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_BFF/runner.ipynb>`__ · `view <https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_BFF/runner.ipynb>`__

   .. grid-item-card:: :octicon:`database;1.2em` 2a · Calibration from available data
      :class-card: md-card

      Six parameters, four given experiments added one per round. Each round: fit, bootstrap
      uncertainty and estimability analysis; then cross-validation.
      Folder ``scenario_1_available_data``.

      :octicon:`play` `Open · Colab <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_1_available_data/runner.ipynb>`__ · `view <https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_1_available_data/runner.ipynb>`__

   .. grid-item-card:: :octicon:`goal;1.2em` 2b · Sequential design
      :class-card: md-card

      The same model with no data: Sobol sensitivity analysis, then four rounds of ``mbdoe_pp``
      (D-optimal, then E-optimal), each followed by fit and estimability analysis.
      Folder ``scenario_2_sequential_design``.

      :octicon:`play` `Open · Colab <https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_2_sequential_design/runner.ipynb>`__ · `view <https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_2_sequential_design/runner.ipynb>`__

   .. grid-item-card:: :octicon:`play;1.2em` How to run
      :class-card: md-card

      **In the browser:** press a *Colab* link, then run all cells; the first cell installs MIDDoE
      and fetches the model files. Nothing to install.

      **On your PC:** download the repository (GitHub, **Code** → *Download ZIP*), run
      ``pip install middoe notebook``, open ``runner.ipynb`` and run all cells.

.. admonition:: Run time
   :class: tip

   With the paper's settings each notebook takes **hours** on a many-core PC; most of it goes into
   the design (``num_parallel_runs=16``) and the 500 bootstrap refits. For a first look, lower
   ``num_parallel_runs``, ``maxmd`` / ``maxpp`` and ``nboot``: with 2 runs, 40 iterations and 5
   refits each notebook finishes in about 30–80 minutes. Run one notebook at a time; on Colab (2
   cores) always start with the lower values.
