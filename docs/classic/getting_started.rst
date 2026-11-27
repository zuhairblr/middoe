Getting started
===============

This page takes you from nothing to a finished MIDDoE study in about 15 minutes. You do not need
programming experience: you will copy two files and run one command.

1. Install Python and MIDDoE
----------------------------

#. **Install Python** (version 3.9 to 3.13) from https://www.python.org/downloads/.
   On Windows, tick **"Add python.exe to PATH"** in the first installer window.
#. **Open a terminal.** On Windows: press Start, type ``cmd``, press Enter. On macOS: open the
   *Terminal* app. On Linux: open your usual terminal.
#. **Install MIDDoE** by typing this line and pressing Enter:

   .. code-block:: bash

      pip install middoe

   (If ``pip`` is not found, try ``python -m pip install middoe``, or ``py -m pip install middoe``
   on Windows.)

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

It takes about a minute. MIDDoE prints its progress, then results like these (the numbers vary a
little from run to run because the noise is random):

.. code-block:: text

   After 2 starting experiments
     k_ref  =     0.3099  +/- 0.0146  (t = 44.0)
     Ea     =  5.773e+04  +/- 5.8e+03  (t = 20.6)

   After 1 designed experiment
     k_ref  =     0.3074  +/- 0.00652  (t = 95.8)
     Ea     =  5.427e+04  +/- 2.28e+03  (t = 48.4)

5. Read the results
-------------------

- The **estimates** are close to the true values used by the virtual lab (``k_ref = 0.3``,
  ``Ea = 60000``).
- **+/-** is the 95 % confidence interval: the true value very likely lies within that range.
- **t** measures precision. A parameter is considered well determined when its t-value is above a
  reference value of about 2.
- **The designed experiment pays off.** ``mbdoe_pp`` planned experiment 3: start with as much A as
  allowed, run hot for about half an hour, then run cold, sampling at the most informative times.
  This single experiment roughly **halved the uncertainty** of both parameters, more than two
  ordinary experiments together.

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

   On Windows, keep the ``if __name__ == '__main__':`` line in your scripts. MIDDoE runs some steps
   in parallel, and this line is what makes that work.

Next steps
----------

- :doc:`concepts`: what each step does and why.
- :doc:`case_studies`: larger studies with several rival models, sensitivity and estimability
  analysis, and sequential designs.
- :doc:`user_guide` and :doc:`api`: every option in detail.
