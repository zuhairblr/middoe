API Reference
=============

MIDDoE has **about a dozen functions you call yourself**. Everything else is used internally. This
page lists those functions first, in the order of a study, and the complete module reference after
them.

New to MIDDoE? Read :doc:`getting_started` first. The dictionaries the functions take (``system``,
``models``, ``iden_opt``, ``des_opt``, ``insilicos``, ``gsa``) are explained in
:ref:`config-dicts`.

Imports
-------

Copy the lines you need:

.. code-block:: python

   from middoe.krnl_expera import expera              # virtual-lab experiments -> data.xlsx
   from middoe.sc_sensa import sensa                   # global sensitivity analysis (Sobol)
   from middoe.iden_parmest import parmest             # parameter estimation
   from middoe.iden_uncert import uncert               # confidence intervals, t-values, P-test
   from middoe.sc_estima import estima                 # estimability: which parameters to fit
   from middoe.des_pp import mbdoe_pp                  # design for parameter precision
   from middoe.des_md import mbdoe_md                  # design for model discrimination
   from middoe.iden_valida import validation           # leave-one-out cross-validation
   from middoe.log_utils import save_rounds, save_to_jac, load_from_jac, read_excel
   from middoe.iden_utils import run_postprocessing    # plots and Excel reports

Which function do I call?
-------------------------

.. list-table::
   :widths: 22 22 28 28
   :header-rows: 1

   * - Step
     - Function
     - You give
     - You get
   * - Run an experiment in the virtual lab
     - :func:`~middoe.krnl_expera.expera`
     - ``system``, ``models``, ``insilicos``, a design (or ``{}``), the experiment number
     - a new sheet in ``data.xlsx``
   * - Find the influential parameters
     - :func:`~middoe.sc_sensa.sensa`
     - ``gsa``, ``models``, ``system``
     - Sobol indices for every model and output
   * - Fit the parameters
     - :func:`~middoe.iden_parmest.parmest`
     - ``system``, ``models``, ``iden_opt`` (data from ``data.xlsx``)
     - the estimates for every candidate model
   * - Check how precise they are
     - :func:`~middoe.iden_uncert.uncert`
     - the ``parmest`` result, ``system``, ``models``, ``iden_opt``
     - confidence intervals, t-values, R², model probabilities P
   * - Choose which parameters to fit
     - :func:`~middoe.sc_estima.estima`
     - ``unc['results']`` (the ``'results'`` entry of the ``uncert`` output), ``system``,
       ``models``, ``iden_opt``, the round
     - a ranking and the best number of parameters to estimate
   * - Plan the next experiment (precision)
     - :func:`~middoe.des_pp.mbdoe_pp`
     - ``des_opt``, ``system``, ``models``, the round
     - the designed inputs and sampling times
   * - Plan the next experiment (discrimination)
     - :func:`~middoe.des_md.mbdoe_md`
     - ``des_opt``, ``system``, ``models``, the round
     - the designed inputs and sampling times
   * - Store a round
     - :func:`~middoe.log_utils.save_rounds`
     - the round number, ``unc['results']``, ``unc['obs']`` and the settings
     - an entry in ``round_data`` (used by the next three)
   * - Cross-validate
     - :func:`~middoe.iden_valida.validation`
     - ``system``, ``models``, ``iden_opt``, ``round_data``
     - prediction and validation R² for each left-out experiment
   * - Plots and reports
     - :func:`~middoe.iden_utils.run_postprocessing`
     - ``round_data``, the model names, the rounds
     - figures and Excel files in ``post_processing/``
   * - Save / reload everything
     - :func:`~middoe.log_utils.save_to_jac`, :func:`~middoe.log_utils.load_from_jac`
     - any result
     - a ``.jac`` file (pickle)

Reading the results
-------------------

The two results you will use most:

.. code-block:: python

   est = parmest(system, models, iden_opt)
   unc = uncert(est, system, models, iden_opt)

   r = unc['results']['M1']        # one dictionary per candidate model
   r['estimations']                # parameter estimates (all parameters, physical units)
   r['CI']                         # 95 % confidence half-widths (estimated parameters only)
   r['t_values']                   # t-values; compare with the reference t (about 2)
   r['V_matrix']                   # covariance matrix: estimated parameters only ('M'/'H');
                                   # full size with zero rows/columns for fixed ones ('B')
   r['R2_responses']               # R² per measured output
   r['P']                          # model probability in % (when several models compete)

``uncert`` also stores the new estimates and covariance in ``models`` (``models['theta']``,
``models['V_matrix']``), so the next design starts from them automatically.

Main functions
--------------

Experiments and data
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.krnl_expera.expera

.. autofunction:: middoe.log_utils.read_excel

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.sc_sensa.sensa

.. autofunction:: middoe.log_utils.save_to_xlsx

Parameter estimation and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.iden_parmest.parmest

.. autofunction:: middoe.iden_uncert.uncert

Estimability analysis
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.sc_estima.estima

Design of experiments
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.des_pp.mbdoe_pp

.. autofunction:: middoe.des_md.mbdoe_md

Validation, storage and reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: middoe.iden_valida.validation

.. autofunction:: middoe.log_utils.save_rounds

.. autofunction:: middoe.log_utils.save_to_jac

.. autofunction:: middoe.log_utils.load_from_jac

.. autofunction:: middoe.iden_utils.run_postprocessing

Simulation
~~~~~~~~~~

You rarely call this yourself: every other function uses it to simulate your model.

.. autofunction:: middoe.krnl_simula.simula

Complete module reference
-------------------------

The remaining public functions and classes of each module: helpers used by the main functions,
plotting routines and the built-in test models. Most users never need them.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Contents
   * - :mod:`middoe.iden_parmest`
     - Parameter estimation; bootstrap helpers
   * - :mod:`middoe.iden_uncert`
     - Uncertainty analysis
   * - :mod:`middoe.iden_valida`
     - Cross-validation helpers
   * - :mod:`middoe.iden_utils`
     - Plots and Excel reports
   * - :mod:`middoe.des_pp`
     - Design for parameter precision: D, A, E, ME, ORB, SEM criteria
   * - :mod:`middoe.des_md`
     - Design for model discrimination: HR, BFF criteria
   * - :mod:`middoe.des_utils`
     - Internal design helpers; the only public function is ``configure_logger``
   * - :mod:`middoe.sc_sensa`
     - Sobol sampling and indices
   * - :mod:`middoe.sc_estima`
     - Estimability ranking and subset selection
   * - :mod:`middoe.krnl_simula`
     - Simulation kernel and ODE-solver selection
   * - :mod:`middoe.krnl_expera`
     - Virtual-lab experiments
   * - :mod:`middoe.krnl_models`
     - Built-in test models
   * - :mod:`middoe.log_utils`
     - Reading data, storing rounds, saving and loading results

Parameter estimation
~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.iden_parmest
   :members:
   :exclude-members: parmest

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.iden_uncert
   :members:
   :exclude-members: uncert

Validation
~~~~~~~~~~

.. automodule:: middoe.iden_valida
   :members:
   :exclude-members: validation

Plots and reports
~~~~~~~~~~~~~~~~~

.. automodule:: middoe.iden_utils
   :members:
   :exclude-members: run_postprocessing

Design for parameter precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.des_pp
   :members:
   :exclude-members: mbdoe_pp

Design for model discrimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.des_md
   :members:
   :exclude-members: mbdoe_md

Design utilities
~~~~~~~~~~~~~~~~

.. automodule:: middoe.des_utils
   :members:

Global sensitivity analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.sc_sensa
   :members:
   :exclude-members: sensa

Estimability analysis
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.sc_estima
   :members:
   :exclude-members: estima

Simulation kernel
~~~~~~~~~~~~~~~~~

.. automodule:: middoe.krnl_simula
   :members:
   :exclude-members: simula

Virtual-lab experiments
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.krnl_expera
   :members:
   :exclude-members: expera

Built-in models
~~~~~~~~~~~~~~~

Right-hand-side functions available with ``models['krt'][name] = 'pym'``
(the model name must equal one of the function names below, e.g. ``'f05'``).
Only the ``f*`` functions are models usable with ``'pym'``; the ``thetadic*``
functions return parameter sets for them, not models.

.. automodule:: middoe.krnl_models
   :members:
   :undoc-members:

Data, rounds and files
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: middoe.log_utils
   :members:
   :exclude-members: fun_globalizer, read_excel, save_to_xlsx, save_rounds, save_to_jac, load_from_jac
