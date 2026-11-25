Paper case studies
==================

The folder `tests/middoe_paper <https://github.com/zuhairblr/middoe/tree/main/tests/middoe_paper>`_
of the source repository reproduces the two case studies of the MIDDoE paper, step by step and with
comments that explain every option. Each scenario is a notebook (``runner.ipynb``) and the same code
as a script (``runner.py``). After every round they print the results next to the paper's values.

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Scenario
     - What it shows
   * - Case study 1, ``scenario_HR`` and ``scenario_BFF``
     - **Model discrimination.** Four rival models of a fermentation. Preliminary fit, P-test, one
       discriminating experiment designed with the Hunter–Reiner (HR) or Buzzi-Ferraris–Forzatti
       (BFF) criterion, and the final model choice (paper Tables 4–5).
   * - Case study 2, ``scenario_1_available_data``
     - **Calibration with existing data.** A pharmaceutical batch reaction with 6 parameters and 4
       given experiments: estimability analysis, round-by-round fit and cross-validation (Table 8).
   * - Case study 2, ``scenario_2_sequential_design``
     - **Sequential design.** Sensitivity analysis, then 4 rounds of D-optimal experiment design,
       fit and estimability analysis (Table 9).

How to run
----------

#. Download the repository from GitHub (green **Code** button → *Download ZIP*) and unzip it, or
   clone it with git.
#. Install MIDDoE (see :doc:`getting_started`) and Jupyter: ``pip install notebook``.
#. Open ``runner.ipynb`` of a scenario with Jupyter (``jupyter notebook``) and run all cells, or run
   the script from a terminal: ``python runner.py``.

Every runner has switches at the top. The most useful is ``QUICK``:

- ``QUICK = True`` uses tiny budgets and finishes in **minutes**. Good for learning the workflow;
  the numbers differ from the paper.
- ``QUICK = False`` (paper settings) takes **hours**: about 4 h for each case-study-1 scenario and
  2.5 h or more for each case-study-2 scenario on a 28-core PC. Most of the time goes into the
  experiment design (case study 1) and the 500 bootstrap repetitions of the uncertainty analysis
  (case study 2). Running several scenarios at once needs a lot of memory.

All outputs (``data.xlsx``, plots, Excel reports, ``run_summary.json``) are written to the
scenario's ``results/`` folder. ``compare_all.py`` prints the comparison with the paper for all
scenarios. The folder's ``README.md`` describes every file and switch.
