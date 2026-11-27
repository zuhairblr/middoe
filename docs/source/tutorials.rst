.. _hands-on:

Hands-on tutorials
==================

.. rst-class:: md-lead

Five notebooks on two real systems, from a first fit to designing experiments. Each runs in the
browser (Google Colab) or on your PC, and every cell is explained.

.. raw:: html

   <nav class="md-ribbon" aria-label="Learning path">
     <a href="#tut-fit"><span class="md-num">1</span>Fit &amp; uncertainty</a><span class="md-sep"></span>
     <a href="#tut-recipes"><span class="md-num">2</span>Setting recipes</a><span class="md-sep"></span>
     <a href="#tut-discrim"><span class="md-num">3</span>Rival models</a><span class="md-sep"></span>
     <a href="#tut-calib"><span class="md-num">4</span>Calibrate from data</a><span class="md-sep"></span>
     <a href="#tut-design"><span class="md-num">5</span>Design experiments</a>
   </nav>

The animations sketch what each notebook does; click one to replay it.

.. _tut-fit:

1 · Yeast fermentation: fit a model and know how sure you are
-------------------------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="fit"></div>

   .. grid-item::
      :columns: 12 12 6 6

      A fed-batch baker's-yeast fermentation with Monod growth: 4 parameters, 2 measured outputs and
      3 inputs. Make in-silico data, fit the model, compute confidence intervals, and see how each
      dictionary key, optimiser, objective and uncertainty method changes the answer.

      :bdg-info:`start here` :bdg-light:`about 25 min` :bdg-primary-line:`expera` :bdg-primary-line:`parmest` :bdg-primary-line:`uncert` :bdg-primary-line:`save_rounds`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_estimation_and_uncertainty/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/01_estimation_and_uncertainty/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-recipes:

2 · Yeast fermentation: five setting recipes
--------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="recipes"></div>

   .. grid-item::
      :columns: 12 12 6 6

      The same study as five short notebooks, one setting each, to compare side by side:
      **A** relative error, Fisher matrix, fast optimiser · **B** with multi-start · **B2** as B with
      the bootstrap · **C** absolute error, Hessian, estimability analysis · **D** absolute error,
      bootstrap, maximum likelihood.

      :bdg-primary-line:`iden_opt` :bdg-primary-line:`var-cov` :bdg-primary-line:`estima`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/a_relative_fisher_fast/runner.ipynb
         :color: primary
         :outline:

         :octicon:`play` Recipe A

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/b_relative_fisher_best/runner.ipynb
         :color: primary
         :outline:

         :octicon:`play` Recipe B

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/b2_relative_bootstrap_best/runner.ipynb
         :color: primary
         :outline:

         :octicon:`play` Recipe B2

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/c_absolute_hessian_estimability/runner.ipynb
         :color: primary
         :outline:

         :octicon:`play` Recipe C

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_simple/d_absolute_bootstrap_mle/runner.ipynb
         :color: primary
         :outline:

         :octicon:`play` Recipe D

.. _tut-discrim:

3 · Yeast fermentation: which of four kinetic models is right?
--------------------------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="discrim"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Paper case study 1. Four rival growth models fit two preliminary experiments; the P-test ranks
      them, two are dropped, and **one designed experiment** separates the survivors. One notebook
      per criterion: Hunter-Reiner (HR) and Buzzi-Ferraris-Forzatti (BFF).

      :bdg-primary-line:`mbdoe_md` :bdg-primary-line:`P-test` :bdg-primary-line:`HR · BFF`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_HR/runner.ipynb
         :color: primary
         :shadow:

         :octicon:`play` HR in Colab

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_BFF/runner.ipynb
         :color: primary
         :shadow:

         :octicon:`play` BFF in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_1_model_discrimination/scenario_HR/runner.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-calib:

4 · Pharmaceutical batch reaction: calibrate from existing experiments
----------------------------------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="calib"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Paper case study 2, scenario 1. A reaction with an impurity, six parameters and four
      experiments that already exist. They are added one per round; the **estimability analysis**
      decides how many parameters the data can support, and only those are estimated. Then the model
      is cross-validated.

      :bdg-primary-line:`estima` :bdg-primary-line:`bootstrap` :bdg-primary-line:`validation`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_1_available_data/runner.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_1_available_data/runner.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-design:

5 · Pharmaceutical batch reaction: design the experiments
---------------------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="design"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Paper case study 2, scenario 2. No data yet: a **global sensitivity analysis** finds the
      parameters that matter, then four rounds of **model-based design** (D-optimal, then E-optimal)
      choose the temperature profiles and sampling times that pin the parameters down.

      :bdg-primary-line:`sensa` :bdg-primary-line:`mbdoe_pp` :bdg-primary-line:`D · E`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_2_sequential_design/runner.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/middoe_paper/case_study_2_model_calibration/scenario_2_sequential_design/runner.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook


Before you run
--------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`globe;1.2em` In the browser
      :class-card: md-card

      Press an *Open in Colab* button and run all cells. The first cell installs MIDDoE and fetches
      the model file(s); there is nothing to install.

   .. grid-item-card:: :octicon:`device-desktop;1.2em` On your PC
      :class-card: md-card

      Download the repository (GitHub, **Code**, *Download ZIP*), run ``pip install middoe notebook``,
      open the notebook from its folder and run all cells. Outputs appear next to it.

.. admonition:: Run time of tutorials 3 to 5 (the paper case studies)
   :class: tip

   With the paper's settings each takes **hours** on a many-core PC, mostly in the design
   (``num_parallel_runs=16``) and the 500 bootstrap refits. For a first run lower
   ``num_parallel_runs``, ``maxmd`` / ``maxpp`` and ``nboot``: with 2 runs, 40 iterations and 5 refits
   each finishes in about 30 to 80 minutes. On Colab (2 cores) always start that way.
