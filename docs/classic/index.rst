.. image:: https://research.dii.unipd.it/capelab/wp-content/uploads/sites/36/2025/03/logo-Page-5.png
   :alt: CAPE-Lab / MIDDoE
   :align: center
   :width: 400

MIDDoE
======

**Model Identification, Discrimination and Design of Experiments**

.. image:: https://img.shields.io/pypi/v/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/zuhairblr/middoe.svg
   :target: https://github.com/zuhairblr/middoe/blob/main/LICENSE
   :alt: License

MIDDoE helps you turn a process model and a few experiments into a **trustworthy model**. It answers
three practical questions:

- **Which model is right?** If you have several candidate models, MIDDoE plans the experiment that
  best tells them apart (*model discrimination*).
- **What are its parameters, and how sure are we?** MIDDoE fits the unknown parameters to your data
  and tells you how precise each one is (*parameter estimation and uncertainty*).
- **Which experiment should I run next?** MIDDoE plans the experiment that reduces that uncertainty
  the most (*model-based design of experiments*, MBDoE).

You describe your process in a few plain Python dictionaries and call one function per step. No
knowledge of optimisation or statistics is needed to get started.

How a study looks
-----------------

A study is a loop of **rounds**. Each round plans an experiment, runs it, and refits the model:

.. code-block:: text

        preliminary experiments
                  |
                  v
     +--> fit the parameters ----------- parmest
     |            |
     |    how precise are they? -------- uncert      (confidence intervals, t-test, P-test)
     |            |
     |      good enough? -- yes --> validate and report   (validation, run_postprocessing)
     |            | no
     |    plan the next experiment ----- mbdoe_pp  (precision)  or  mbdoe_md  (which model)
     |            |
     +---- run it (lab, or virtual lab: expera)

In code, one round is four lines:

.. code-block:: python

   from middoe.iden_parmest import parmest
   from middoe.iden_uncert import uncert
   from middoe.des_pp import mbdoe_pp
   from middoe.krnl_expera import expera

   est = parmest(system, models, iden_opt)                    # fit to data.xlsx
   unc = uncert(est, system, models, iden_opt)                # CI, t-values, R2
   design = mbdoe_pp(des_opt, system, models, round=1)        # plan the next experiment
   expera(system, models, insilicos, design, expr=3, swps=design['swps'])  # run it (virtual lab)

``system`` describes the experiment (inputs, outputs, limits), ``models`` the candidate models and
their parameters, and ``iden_opt`` / ``des_opt`` the options. :doc:`getting_started` builds all of
them for a complete example you can run in a minute.

Where to start
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`getting_started`
     - Install MIDDoE and run a complete example in about 15 minutes. Start here.
   * - :doc:`concepts`
     - The ideas behind MIDDoE in plain words (5-minute read).
   * - :doc:`case_studies`
     - The two case studies of the MIDDoE paper, ready to run.
   * - :doc:`user_guide`
     - The full reference: theory, data file, model file, every option, complete workflow.
   * - :doc:`api`
     - The dozen functions you call, in workflow order, with every argument and return value.
   * - :doc:`changelog`
     - What changed in each release.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   concepts
   case_studies
   user_guide
   api
   changelog

.. note::

   MIDDoE implements the methods of the MIDDoE paper (Tabrizi et al., *Digital Chemical Engineering*
   17 (2025) 100276). It also offers features beyond that paper, e.g. the ORB and SEM design
   criteria, FIM spectral metrics and the MBDoE Excel/text reports; see the :doc:`api`.

Citation
--------

If you use MIDDoE in your research, please cite:

#. Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025). MIDDoE: An MBDoE Python
   package for model identification, discrimination, and calibration. *Digital Chemical
   Engineering*, 17, 100276. https://doi.org/10.1016/j.dche.2025.100276
#. Tabrizi, S.Z.B., Barbera, E., Da Silva, W.R.L., & Bezzo, F. (2025). A Python/Numpy-based package
   to support model discrimination and identification. In J.F.M. Van Impe et al. (Eds.), *Systems
   and Control Transactions*, 4, 1282–1287. https://doi.org/10.69997/sct.192104

Applied in, for example: Tabrizi, Z., Rodriguez, C., Barbera, E., Leal da Silva, W.R., & Bezzo, F.
(2025). Wet carbonation of industrial recycled concrete fines: experimental study and reaction
kinetic modeling. *Ind. Eng. Chem. Res.*, 64(45), 21412–21425. https://doi.org/10.1021/acs.iecr.5c02835

Resources
---------

- **Source code:** https://github.com/zuhairblr/middoe
- **PyPI:** https://pypi.org/project/middoe/
- **Questions and bug reports:** https://github.com/zuhairblr/middoe/issues

* :ref:`genindex`
* :ref:`search`
