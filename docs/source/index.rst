.. image:: _static/middoe-logo.png
   :alt: CAPE-Lab / MIDDoE
   :align: center
   :width: 320

MIDDoE
======

.. rst-class:: md-tagline

**Model Identification, Discrimination and Design of Experiments** in Python.

|pypi| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: PyPI version
.. |pyver| image:: https://img.shields.io/pypi/pyversions/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: Python versions
.. |license| image:: https://img.shields.io/github/license/zuhairblr/middoe.svg
   :target: https://github.com/zuhairblr/middoe/blob/main/LICENSE
   :alt: License

MIDDoE turns a process model and a few experiments into a **trustworthy model**. You describe your
process in a few plain Python dictionaries and call one function per step; no knowledge of
optimisation or statistics is needed to get started.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item::

      .. button-ref:: user_guide
         :ref-type: doc
         :color: primary
         :shadow:
         :expand:

         :octicon:`rocket` Onboarding guide

   .. grid-item::

      .. button-ref:: concepts
         :ref-type: doc
         :color: primary
         :outline:
         :expand:

         :octicon:`light-bulb` Key ideas & structure

   .. grid-item::

      .. button-ref:: api
         :ref-type: doc
         :color: secondary
         :outline:
         :expand:

         :octicon:`code` API reference

What MIDDoE answers
-------------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`git-compare;1.3em` Which model is right?
      :class-card: md-card

      Several candidate models? MIDDoE plans the experiment that best tells them apart and ranks
      them by probability (*model discrimination*).

   .. grid-item-card:: :octicon:`shield-check;1.3em` How sure are the parameters?
      :class-card: md-card

      MIDDoE fits the unknown parameters to your data and tells you how precise each one is:
      confidence intervals, t-tests, correlations (*estimation and uncertainty*).

   .. grid-item-card:: :octicon:`goal;1.3em` Which experiment next?
      :class-card: md-card

      MIDDoE plans the inputs and sampling times that reduce that uncertainty the most
      (*model-based design of experiments*, MBDoE).

How a study looks
-----------------

A study is a loop of **rounds**. Each round plans an experiment, runs it, and refits the model:

.. raw:: html

   <nav class="md-ribbon md-loop" aria-label="A MIDDoE round">
     <a href="ug_setup.html#setup-data"><span class="md-num">1</span>Run experiments</a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-rounds"><span class="md-num">2</span>Fit: <code>parmest</code></a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-rounds"><span class="md-num">3</span>Precision: <code>uncert</code></a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-design"><span class="md-num">4</span>Plan the next: <code>mbdoe_pp</code> / <code>mbdoe_md</code></a><span class="md-sep"></span>
     <a href="ug_setup.html#setup-report"><span class="md-num">5</span>Report: <code>run_postprocessing</code></a>
   </nav>

Repeat steps 1–4 until the parameters are precise enough (or one model clearly wins). In code, one
round is four lines:

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

Where to go
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.3em` Onboarding guide
      :link: user_guide
      :link-type: doc
      :class-card: md-card

      Install MIDDoE, run a first study, then set up your own: dictionaries, settings, results.
      **Start here.**

   .. grid-item-card:: :octicon:`light-bulb;1.3em` Key ideas & package structure
      :link: concepts
      :link-type: doc
      :class-card: md-card

      The ideas behind MIDDoE, interactive figures that show them at work, and how the package is
      organised.

   .. grid-item-card:: :octicon:`play;1.3em` Paper case studies
      :link: case_studies
      :link-type: doc
      :class-card: md-card

      The four case studies of the MIDDoE paper as notebooks; open them in Google Colab and run them
      in the browser.

   .. grid-item-card:: :octicon:`code;1.3em` API reference
      :link: api
      :link-type: doc
      :class-card: md-card

      The dozen functions you call, in workflow order, with every argument and return value.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide
   concepts
   case_studies
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

MBDoE in ill-conditioned systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the design criteria for ill-conditioned systems (ORB and SEM), please also cite:

#. Tabrizi, Z., Leal da Silva, W.R., & Bezzo, F. (2026). Two novel criteria for model-based design
   of experiments in ill-conditioned systems. *Computers & Chemical Engineering*, 214, 109806.
   https://doi.org/10.1016/j.compchemeng.2026.109806

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
