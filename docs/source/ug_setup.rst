Set up a study
==============

.. include:: _ribbon.rst

.. rst-class:: md-lead

Six steps from an empty folder to a report. Steps marked *optional* can wait.

Everything lives in one folder; MIDDoE reads and writes in the working folder:

.. code-block:: text

   my_study/
   ├── model.py      # your model                      (step 1)
   ├── run.py        # your study, or a Jupyter notebook (steps 2-5)
   └── data.xlsx     # your measurements; written by expera() in in-silico studies

.. _setup-model:

1 · Write the model :bdg-primary:`required`
-------------------------------------------

``model.py`` defines ``solve_model(t, y0, tii, tvi, theta)`` and returns the predicted outputs on the
time grid ``t``. The names must match the keys of ``system``.

.. code-block:: python

   def solve_model(t, y0, tii, tvi, theta):
       # t: time grid, tii: {name: value}, tvi: {name: array on t}, theta: list
       ...                     # solve your equations, e.g. scipy solve_ivp
       return {'tvo': {'y1': y1_on_t, 'y2': y2_on_t}, 'tio': {}}

.. dropdown:: :octicon:`plug` Other ways to connect a model

   Python functions (right-hand side only), built-in test models and gPROMS models also work; set
   ``models['krt']``. The full contract and a complete example: :ref:`model-file`.

.. _setup-dicts:

2 · Describe the experiment and the model :bdg-primary:`required`
-----------------------------------------------------------------

Four dictionaries hold everything. Open one to see what it contains; the
:ref:`reference <config-dicts>` explains every key.

.. tab-set::

   .. tab-item:: system

      The experiment: time span and grid, the inputs you can set with their ranges, the outputs you
      measure with their measurement error ``unc`` and number of samples.

      .. code-block:: python

         system = {
             't_s': [0, 40], 't_r': 0.16, 't_d': 0,   # time span, grid, dead time
             'tvi': {'u1': {'min': 0.05, 'max': 0.2,  # inputs u(t)
                            'cvp': 'CPF', 'stps': 5, 'const': 'rel',
                            'offl': 0.01, 'offt': 0.5}},
             'tii': {'y10': {'min': 1, 'max': 5.5}},  # fixed per experiment
             'tvo': {'y1': {'meas': True, 'unc': 0.08,  # measured outputs
                            'init': 0, 'sp': 10, 'samp_s': 1,
                            'samp_f': [], 'offt': 0.5}},
             'tio': {},
         }

   .. tab-item:: models

      The candidate model(s): how to run them, the starting guess, bounds, and which parameters to
      estimate.

      .. code-block:: python

         models = {
             'can_m': ['M'],                         # candidate model names
             'krt': {'M': 'pys'},                    # a Python file ...
             'src': {'M': 'model.py'},               # ... at this path
             'theta': {'M': [0.3, 0.3, 1.0, 0.1]},   # starting guess
             't_l': {'M': [0.1, 0.1, 0.4, 0.04]},    # lower bounds
             't_u': {'M': [0.4, 0.4, 1.3, 0.14]},    # upper bounds
             'mutation': {'M': [True] * 4},          # True = estimate, False = keep fixed
         }

   .. tab-item:: iden_opt

      How to estimate and how to compute the uncertainty. Not sure what to pick? See
      :doc:`ug_settings`.

      .. code-block:: python

         iden_opt = {
             'meth': 'SLSQP', 'ms': False,   # optimiser (required), multi-start
             'ob': 'WLS', 'tol': 1e-8,       # objective, tolerance
             'var-cov': 'M',                 # uncertainty: 'M' Fisher, 'H' Hessian, 'B' bootstrap
             'c_plt': True, 'f_plt': True,   # confidence and fit plots per round
             'plt_s': True,                  # also show them
         }

   .. tab-item:: insilicos

      Only for in-silico studies: the "true" model and parameters, the noise, and the preliminary
      experiments.

      .. code-block:: python

         insilicos = {
             'tr_m': 'M', 'theta': [0.25, 0.25, 0.88, 0.09],   # true model and parameters
             'errt': 'rel',                                    # noise: 'abs' or 'rel'
             'prels': {'1': {'u1': 0.05, 'y10': 5.5},          # preliminary experiments
                       '2': {'u1': 0.10, 'y10': 5.5}},
         }

   .. tab-item:: des_opt

      Only when MIDDoE designs the experiments: the design criterion and the optimiser.

      .. code-block:: python

         des_opt = {
             'pp_ob': 'D',            # precision: 'D', 'A', 'E', 'ME'
             'md_ob': 'BFF',          # discrimination: 'BFF', 'HR'
             'meth': 'PS', 'eps': 1e-5,
             'itr': {'pps': 100, 'maxpp': 100, 'tolpp': 1, 'maxmd': 5, 'tolmd': 1},
             'plt': True,
         }

.. _setup-designspace:

See the design space your ``system`` defines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``tvi``, ``tvo`` and ``t_d`` keys decide which experiments MIDDoE may design. Change them below and
watch the space change: the generated dictionary updates as you go, every handle obeys the same rules
as MIDDoE's design optimiser, and the number of decision variables the optimiser has to choose is
counted for you.

.. raw:: html

   <div class="mdw" data-widget="designspace"></div>

.. _setup-data:

3 · Get the data :bdg-primary:`required`
----------------------------------------

.. tab-set::

   .. tab-item:: :octicon:`beaker` Lab data

      Put each experiment in its own sheet of ``data.xlsx``, laid out as below (point at a
      header). Every rule: :ref:`data-file`.

      .. raw:: html

         <div class="mdw" data-widget="datafile"></div>

   .. tab-item:: :octicon:`cpu` In silico

      Let MIDDoE simulate the experiment with the true model and add noise:

      .. code-block:: python

         expera(system, models, insilicos, {}, expr=1)     # preliminary experiment '1' -> data.xlsx

      A small figure shows the new data, the true curve and the noise.

.. _setup-rounds:

4 · Run the rounds :bdg-primary:`required`
------------------------------------------

Each round: add an experiment, refit **all** data, quantify the uncertainty, store the round.

.. code-block:: python

   round_data = {}
   for k in (1, 2, 3):
       expera(system, models, insilicos, {}, expr=k)            # or add your lab data
       resultpr = parmest(system, models, iden_opt)             # estimate
       unc = uncert(resultpr, system, models, iden_opt)         # CI, t-values; updates models
       save_rounds(k, unc['results'], 'preliminary', round_data, models, iden_opt,
                   unc['obs'], system)                          # store + fit/confidence plots

In a script, keep the study under ``if __name__ == '__main__':`` (see :ref:`pitfalls`).

.. _setup-design:

5 · Let MIDDoE design the next experiment :bdg-secondary:`optional`
-------------------------------------------------------------------

After a round, MBDoE chooses the inputs and sampling times that make the next experiment most
informative, at the current estimates:

.. code-block:: python

   design = mbdoe_pp(des_opt, system, models, round=4)      # or mbdoe_md(...) for rival models
   expera(system, models, insilicos, design, expr=4, swps=design['swps'])   # becomes experiment 4

In the lab, run the designed experiment, add it to ``data.xlsx`` and run the next round. Complete
examples: :ref:`practical-workflow`.

.. _setup-report:

6 · Report :bdg-primary:`required`
----------------------------------

.. code-block:: python

   save_to_jac(round_data, purpose='iden')                       # all rounds -> iden_results.jac
   run_postprocessing(round_data, ['M'], [1, 2, 3], show=False)   # -> post_processing/
