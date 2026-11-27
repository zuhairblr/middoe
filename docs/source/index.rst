:html_theme.sidebar_secondary.remove: true

MIDDoE
======

.. raw:: html

   <section class="md-hero">
     <div class="md-hero-text">
       <img class="md-hero-logo" src="_static/middoe-logo.png" alt="MIDDoE logo">
       <p class="md-hero-title">Model Identification, Discrimination and Design&nbsp;of&nbsp;Experiments</p>
       <p class="md-hero-sub">Turn a process model and a few experiments into a <b>trustworthy model</b>.
         Describe your process in plain Python dictionaries and call one function per step:
         MIDDoE fits the parameters, tells you how sure they are, picks the right model and
         designs the next experiment.</p>
       <div class="md-hero-cta">
         <a class="md-cta md-cta-primary" href="user_guide.html">Get started</a>
         <a class="md-cta" href="tutorials.html">Hands-on tutorials</a>
         <a class="md-cta md-cta-ghost" href="api.html">API</a>
       </div>
       <div class="md-install"><code>pip install middoe</code><button type="button" class="md-copy" data-copy="pip install middoe" aria-label="Copy the install command">Copy</button></div>
     </div>
     <div class="md-hero-art"><div class="mdw md-hero-anim" data-widget="hero"></div></div>
   </section>

   <div class="md-stats">
     <div><b>3</b><span>questions it answers</span></div>
     <div><b>12</b><span>functions, one per step</span></div>
     <div><b>6</b><span>dictionaries describe a study</span></div>
     <div><b>5</b><span>tutorials that run in the browser</span></div>
   </div>

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

What MIDDoE answers
-------------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`git-compare;1.6em` Which model is right?
      :class-card: md-card md-feature

      Several candidate models? MIDDoE designs the experiment that best tells them apart and ranks
      them by probability.

      :bdg-primary-line:`model discrimination`

   .. grid-item-card:: :octicon:`shield-check;1.6em` How sure are the parameters?
      :class-card: md-card md-feature

      MIDDoE fits the unknown parameters to your data and tells you how precise each one is:
      confidence intervals, t-tests, correlations.

      :bdg-primary-line:`estimation & uncertainty`

   .. grid-item-card:: :octicon:`goal;1.6em` Which experiment next?
      :class-card: md-card md-feature

      MIDDoE chooses the inputs and sampling times that shrink that uncertainty the most.

      :bdg-primary-line:`MBDoE`

One round in four lines
-----------------------

A study is a loop of **rounds**: plan an experiment, run it, refit, check. Repeat until the
parameters are precise enough or one model clearly wins.

.. code-block:: python

   est = parmest(system, models, iden_opt)                # fit the parameters
   unc = uncert(est, system, models, iden_opt)            # how sure are they?
   design = mbdoe_pp(des_opt, system, models, round=1)    # plan the next experiment
   expera(system, models, insilicos, design, expr=3)      # run it (here in silico)

``system`` describes the experiment, ``models`` the candidate models, ``iden_opt`` and ``des_opt``
the options. :doc:`getting_started` builds them all for a study you can run in a minute.

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

   .. grid-item-card:: :octicon:`play;1.3em` Hands-on tutorials
      :link: tutorials
      :link-type: doc
      :class-card: md-card

      Five notebooks on two real systems, from a first fit to designing experiments, including the
      paper's case studies. They run in the browser (Google Colab).

   .. grid-item-card:: :octicon:`light-bulb;1.3em` Key ideas & package structure
      :link: concepts
      :link-type: doc
      :class-card: md-card

      The ideas behind MIDDoE, interactive figures that show them at work, and how the package is
      organised.

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
   tutorials
   api
   changelog

Cite MIDDoE
-----------

MIDDoE implements the methods of the MIDDoE paper and goes beyond it, for example with the ORB and
SEM design criteria, FIM spectral metrics and the MBDoE Excel and text reports (see the :doc:`api`).
If you use it in your research, please cite:

.. tab-set::

   .. tab-item:: MIDDoE

      Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025). MIDDoE: An MBDoE Python
      package for model identification, discrimination, and calibration. *Digital Chemical
      Engineering*, 17, 100276. https://doi.org/10.1016/j.dche.2025.100276

   .. tab-item:: Conference paper

      Tabrizi, S.Z.B., Barbera, E., Da Silva, W.R.L., & Bezzo, F. (2025). A Python/Numpy-based
      package to support model discrimination and identification. In J.F.M. Van Impe et al. (Eds.),
      *Systems and Control Transactions*, 4, 1282–1287. https://doi.org/10.69997/sct.192104

   .. tab-item:: ORB & SEM criteria

      For the design criteria for ill-conditioned systems, please also cite: Tabrizi, Z., Leal da
      Silva, W.R., & Bezzo, F. (2026). Two novel criteria for model-based design of experiments in
      ill-conditioned systems. *Computers & Chemical Engineering*, 214, 109806.
      https://doi.org/10.1016/j.compchemeng.2026.109806

Applied in, for example: Tabrizi, Z., Rodriguez, C., Barbera, E., Leal da Silva, W.R., & Bezzo, F.
(2025). Wet carbonation of industrial recycled concrete fines: experimental study and reaction
kinetic modeling. *Ind. Eng. Chem. Res.*, 64(45), 21412–21425. https://doi.org/10.1021/acs.iecr.5c02835

**Source code** on `GitHub <https://github.com/zuhairblr/middoe>`__ · **package** on
`PyPI <https://pypi.org/project/middoe/>`__ · **questions and bug reports** in the
`issue tracker <https://github.com/zuhairblr/middoe/issues>`__ · :ref:`genindex`
