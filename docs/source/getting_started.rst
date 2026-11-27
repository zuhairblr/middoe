Getting started
===============

.. include:: _ribbon.rst

.. rst-class:: md-lead

From nothing to a finished study in about 15 minutes: install, copy two files, run one command.

1. Install Python and MIDDoE
----------------------------

Install Python 3.9–3.13 from https://www.python.org/downloads/, then, in a terminal:

.. tab-set::

   .. tab-item:: :fab:`windows` Windows

      Tick **"Add python.exe to PATH"** in the installer. Open a terminal (Start → ``cmd``) and run:

      .. code-block:: bash

         py -m pip install middoe

   .. tab-item:: :fab:`apple` macOS

      Open the *Terminal* app and run:

      .. code-block:: bash

         python3 -m pip install middoe

   .. tab-item:: :fab:`linux` Linux

      In your terminal:

      .. code-block:: bash

         python3 -m pip install middoe

2. The example
--------------

A batch reactor converts **A into B**. The reaction speeds up with temperature according to the
Arrhenius law, with two unknown parameters:

- ``k_ref``, the rate constant at 50 °C, and
- ``Ea``, the activation energy (how strongly the rate depends on temperature).

In each experiment you choose the **temperature profile** and the **starting amount of A**, and you
measure **A and B** at 6 times. Instead of a real lab, a *virtual lab* simulates the experiments with
known "true" parameters plus measurement noise, so you can check that MIDDoE finds them.

3. Create the two files
-----------------------

Make a new folder (e.g. ``my_first_middoe``) and save the two files below in it. Any text editor
works (Notepad, TextEdit, or a code editor such as VS Code). Use the copy button at the top right of
each box.

**model.py**: the process model. MIDDoE calls ``solve_model`` with the time grid, the starting
values, the chosen inputs and the parameters, and expects the predicted outputs back.

.. literalinclude:: examples/quickstart/model.py
   :language: python

**run.py**: the study. Sections 1–3 describe the experiment, the model and the virtual lab; the last
lines run the study.

.. literalinclude:: examples/quickstart/run.py
   :language: python

4. Run it
---------

In the terminal, go to the folder and run the study:

.. code-block:: bash

   cd path/to/my_first_middoe
   python run.py

It takes about a minute. MIDDoE prints its progress, then results like these (the noise is random,
so your numbers differ a little):

.. code-block:: text

   After 2 starting experiments
     k_ref  =     0.3099  +/- 0.0146  (t = 44.0)
     Ea     =  5.773e+04  +/- 5.8e+03  (t = 20.6)

   After 1 designed experiment
     k_ref  =     0.3074  +/- 0.00652  (t = 95.8)
     Ea     =  5.427e+04  +/- 2.28e+03  (t = 48.4)

5. Read the results
-------------------

- The **estimates** are close to the virtual lab's true values (``k_ref = 0.3``, ``Ea = 60000``).
- **+/-** is the 95 % confidence interval; **t** is the estimate divided by its standard error. Above
  about 2, the parameter is well determined.
- ``mbdoe_pp`` designed experiment 3: as much A as allowed, hot for about half an hour, then cold,
  sampled at the most informative times. That one experiment **halved the uncertainty**:

.. raw:: html

   <div class="mdw" data-widget="quickstart"></div>

Your folder now also contains:

- ``data.xlsx``: one sheet per experiment (inputs, sampling times, measurements). For real
  experiments you fill this file with your lab data instead of using the virtual lab; see
  :ref:`data-file`.
- ``design/``: a report of the designed experiment.

6. Adapt it to your process
---------------------------

#. **Model:** replace the equations in ``model.py`` with yours. Keep the function name and
   arguments, and return one array per measured output; see :ref:`model-file`.
#. **Experiment:** in ``system``, list your inputs with their ranges and your measured outputs with
   their measurement error (``unc``); see :ref:`config-dicts`.
#. **Parameters:** in ``models``, give a first guess and bounds for each unknown parameter.
#. **Real data:** remove the ``expera`` calls and put your measurements in ``data.xlsx``.

.. tip::

   Keep the ``if __name__ == '__main__':`` line in your scripts, on every operating system. MIDDoE
   runs some steps in parallel processes, and this line is what makes that work (notebooks do not
   need it).
