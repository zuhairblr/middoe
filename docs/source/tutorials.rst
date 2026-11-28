.. _hands-on:

Hands-on tutorials
==================

.. rst-class:: md-lead

Notebooks on real processes, a fed-batch yeast fermentation and a pharmaceutical batch reaction.
The **tutorials** teach one MIDDoE function each and compare its options; the **case studies**
reproduce the MIDDoE paper from start to finish. Every notebook runs in the browser (Google Colab)
or on your PC.

.. raw:: html

   <nav class="md-ribbon" aria-label="Learning path">
     <a href="#tg-est"><span class="md-num">1</span>Estimation</a><span class="md-sep"></span>
     <a href="#tg-sens"><span class="md-num">2</span>Sensitivity</a><span class="md-sep"></span>
     <a href="#tg-estima"><span class="md-num">3</span>Estimability</a><span class="md-sep"></span>
     <a href="#tg-pp"><span class="md-num">4</span>Design: precision</a><span class="md-sep"></span>
     <a href="#tg-md"><span class="md-num">5</span>Design: rival models</a><span class="md-sep"></span>
     <a href="#tg-val"><span class="md-num">6</span>Validation</a><span class="md-sep"></span>
     <a href="#case-studies"><span class="md-num">★</span>Case studies</a>
   </nav>

Each figure sketches what its notebook does: click it to replay, and use its buttons to compare the
options. Every new tutorial has a ``QUICK`` switch in its first cell: ``True`` checks the set-up in
minutes with tiny budgets (and only one design in the design tutorials), ``False`` gives the real
result.

Tutorials
---------

.. _tg-est:

1 · Parameter estimation and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit the model to data and say how precise every parameter is (``parmest``, ``uncert``).

.. _tut-fit:

1a · Yeast growth kinetics: fit a Monod model and know how sure you are
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

      :bdg-info:`start here` :bdg-primary-line:`expera` :bdg-primary-line:`parmest` :bdg-primary-line:`uncert` :bdg-primary-line:`save_rounds`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/01_estimation_and_uncertainty/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/01_estimation_and_uncertainty/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-recipes:

1b · Yeast growth kinetics: five estimation recipes side by side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="recipes"></div>

   .. grid-item::
      :columns: 12 12 6 6

      The same study as five short notebooks, one setting each: **A** relative error, Fisher matrix,
      fast optimiser · **B** with multi-start · **B2** as B with the bootstrap · **C** absolute error,
      Hessian, estimability analysis · **D** absolute error, bootstrap, maximum likelihood.

      :bdg-primary-line:`iden_opt` :bdg-primary-line:`var-cov` :bdg-primary-line:`ob` :bdg-primary-line:`meth`

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

.. _tg-sens:

2 · Global sensitivity analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Which parameters and which inputs move the outputs at all (``sensa``, Sobol indices).

.. _tut-sobol:

2a · Batch API synthesis: which kinetic constants shape the product and the impurity?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="sobol"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Sobol indices of the six Arrhenius parameters of a pharmaceutical batch reaction, over the full
      bounds and in a ±20 % box, at both ends of the temperature range and along the batch. Then fix
      the parameters that never matter.

      :bdg-primary-line:`sensa` :bdg-primary-line:`par_s` :bdg-primary-line:`par_d` :bdg-primary-line:`samp`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/02_sensitivity/a_kinetic_parameters/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/02_sensitivity/a_kinetic_parameters/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-sobolin:

2b · Batch API synthesis: temperature or feed? Sensitivity to the operating conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="sobolin"></div>

   .. grid-item::
      :columns: 12 12 6 6

      The same reaction seen from the operator's side: how much do the temperature profile and the
      initial concentrations move the product and the impurity, over the whole operating window and
      ±2 % around a recipe?

      :bdg-primary-line:`sensa` :bdg-primary-line:`var_s` :bdg-primary-line:`var_d`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/02_sensitivity/b_operating_conditions/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/02_sensitivity/b_operating_conditions/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tg-estima:

3 · Estimability analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

How many parameters the data can support, and which (``estima``).

.. _tut-estima:

3 · Batch API synthesis: how many of six Arrhenius parameters can four batches support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="estima"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Four existing batches are added one at a time. ``estima`` ranks the parameters by orthogonalising
      the sensitivity matrix and the rCC criterion picks how many to estimate; the rest are fixed. See
      the number grow with the data.

      :bdg-primary-line:`estima` :bdg-primary-line:`rCC` :bdg-primary-line:`model_update` :bdg-primary-line:`run_postprocessing`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/03_estimability/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/03_estimability/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tg-pp:

4 · Model-based design for parameter precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the inputs and sampling times of the next experiment (``mbdoe_pp``).

.. _tut-criteria:

4a · Fed-batch yeast: which feeding policy pins down the growth kinetics?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="criteria"></div>

   .. grid-item::
      :columns: 12 12 6 6

      One run designed with each classical criterion, **D**, **A**, **E** and **modified E**, after two
      preliminary runs. Compare the designs and the shape of the confidence region each one aims for,
      then run the D-optimal one and see the intervals shrink.

      :bdg-primary-line:`mbdoe_pp` :bdg-primary-line:`D · A · E · ME` :bdg-primary-line:`priors`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/a_criteria/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/a_criteria/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-illcond:

4b · Arrhenius kinetics with correlated parameters: ORB and SEM designs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="illcond"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Isothermal data leave rate constants and activation energies strongly correlated: an
      ill-conditioned Fisher matrix. Compare D with the **ORB** and **SEM** criteria, which target the
      weak directions, and watch the condition number fall.

      :bdg-primary-line:`mbdoe_pp` :bdg-primary-line:`ORB · SEM` :bdg-primary-line:`conditioning`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/b_ill_conditioned/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/b_ill_conditioned/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-profiles:

4c · Fed-batch yeast: shaping the feed. Steps, ramps, monotone feeds, fewer samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="profiles"></div>

   .. grid-item::
      :columns: 12 12 6 6

      What the optimiser may choose is set in ``system``: step or ramp profiles, free or monotone
      changes, constant inputs, the number and spacing of samples, forced sampling times. Design the
      same run under five rule sets and see what each rule costs in information.

      :bdg-primary-line:`cvp` :bdg-primary-line:`const` :bdg-primary-line:`stps` :bdg-primary-line:`sp · samp_f · offt`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/c_profiles_and_sampling/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/04_design_precision/c_profiles_and_sampling/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tg-md:

5 · Model-based design for model discrimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design the experiment that tells rival models apart (``mbdoe_md``, P-test).

.. _tut-hr:

5a · Monod or Contois? The fed-batch run that tells two growth laws apart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="hr"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Both growth laws fit the preliminary data. The **Hunter-Reiner** design finds the conditions where
      their predictions differ most; one such run moves the P-test from a coin toss to a clear answer.

      :bdg-primary-line:`mbdoe_md` :bdg-primary-line:`HR` :bdg-primary-line:`P-test`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/05_design_discrimination/a_hunter_reiner/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/05_design_discrimination/a_hunter_reiner/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tut-bff:

5b · Is there a maintenance term? Hunter-Reiner versus Buzzi-Ferraris-Forzatti
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="bff"></div>

   .. grid-item::
      :columns: 12 12 6 6

      With uncertain parameters, a large difference between predictions may only reflect that
      uncertainty. **BFF** weighs the difference by the prediction covariance, **HR** does not. Design
      with both and run the BFF one.

      :bdg-primary-line:`mbdoe_md` :bdg-primary-line:`HR · BFF` :bdg-primary-line:`uncertainty`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/05_design_discrimination/b_hr_vs_bff/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/05_design_discrimination/b_hr_vs_bff/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _tg-val:

6 · Validation
~~~~~~~~~~~~~~

Check that the model predicts what it has not seen (``validation``).

.. _tut-valid:

6 · Fed-batch yeast: does the model predict batches it has never seen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="valid"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Leave-one-out cross-validation on four batches for two models, with and without maintenance.
      Both fit; only the right one keeps its R² when it predicts the batch left out.

      :bdg-primary-line:`validation` :bdg-primary-line:`R²` :bdg-primary-line:`leave-one-out`

      .. button-link:: https://colab.research.google.com/github/zuhairblr/middoe/blob/main/tests/tutorials/06_validation/tutorial.ipynb
         :color: primary
         :shadow:

         :octicon:`play` Open in Colab

      .. button-link:: https://github.com/zuhairblr/middoe/blob/main/tests/tutorials/06_validation/tutorial.ipynb
         :color: secondary
         :outline:

         :octicon:`mark-github` View the notebook

.. _case-studies:

Case studies
------------

The case studies of the MIDDoE paper (Tabrizi et al., *Digital Chemical Engineering*, 2025), each a
complete study that chains the functions above. With the paper's settings they take hours; see the
tip at the end of the page.

.. _tut-discrim:

Case study 1 · Yeast fermentation: which of four kinetic models is right?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="discrim"></div>

   .. grid-item::
      :columns: 12 12 6 6

      Four rival growth models fit two preliminary experiments; the P-test ranks them, two are dropped,
      and **one designed experiment** separates the survivors. One notebook per criterion:
      Hunter-Reiner (HR) and Buzzi-Ferraris-Forzatti (BFF).

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

Case study 2.1 · Pharmaceutical batch reaction: calibrate from existing experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="calib"></div>

   .. grid-item::
      :columns: 12 12 6 6

      A reaction with an impurity, six parameters and four experiments that already exist. They are
      added one per round; the **estimability analysis** decides how many parameters the data can
      support, and only those are estimated. Then the model is cross-validated.

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

Case study 2.2 · Pharmaceutical batch reaction: design the experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: md-tut

   .. grid-item::
      :columns: 12 12 6 6

      .. raw:: html

         <div class="mdw mdw-tut" data-widget="tut" data-kind="design"></div>

   .. grid-item::
      :columns: 12 12 6 6

      No data yet: a **global sensitivity analysis** finds the parameters that matter, then four rounds
      of **model-based design** (D-optimal, then E-optimal) choose the temperature profiles and sampling
      times that pin the parameters down.

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

.. admonition:: Run time
   :class: tip

   The tutorials take minutes to an hour; the design tutorials are the slowest. The case studies, with the paper's
   settings, take **hours** on a many-core PC, mostly in the design (``num_parallel_runs=16``) and the
   500 bootstrap refits. For a first run lower ``num_parallel_runs``, ``maxmd`` / ``maxpp`` and
   ``nboot``: with 2 runs, 40 iterations and 5 refits each finishes in about 30 to 80 minutes. On
   Colab (2 cores) always start that way.
