.. _interactive:

See the ideas at work
=====================


Three small experiments you can play with, each on a model simple enough to see everything at once.
The numbers are computed live with the same formulas MIDDoE uses; the models are toys, MIDDoE does
the same for your model with many parameters, inputs and outputs.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: :octicon:`location;1.2em` Where to sample
      :link: ia-sampling
      :link-type: ref
      :class-card: md-card

      Why sampling times decide how precise the parameters are, and what D, A, E and ME optimise.

   .. grid-item-card:: :octicon:`git-compare;1.2em` Rival models
      :link: ia-discrim
      :link-type: ref
      :class-card: md-card

      Why an experiment that both models fit says nothing, and how to design one that separates them.

   .. grid-item-card:: :octicon:`pulse;1.2em` Measurement error
      :link: ia-noise
      :link-type: ref
      :class-card: md-card

      Absolute or relative error, and why a relative error near zero needs a floor.

.. _ia-sampling:

Where to sample: information and the design criteria
----------------------------------------------------

The model is :math:`y = A\,(1-e^{-kt})`. A sample at time :math:`t` tells you about a parameter in
proportion to the **sensitivity** :math:`\partial y/\partial	heta` there (lower strip). The samples
together give the Fisher information :math:`\mathbf{M}=\sum_i \mathbf{s}_i\mathbf{s}_i^{\mathsf T}/\sigma^2`,
and the 95 % joint confidence region of the parameters is the ellipse
:math:`\delta	heta^{\mathsf T}\mathbf{M}\,\delta	heta \le \chi^2_{0.95}(2)` (right, in % of each
parameter, as MIDDoE scales them). Its semi-axes are :math:`\sqrt{\chi^2/\lambda_i(\mathbf{M})}`, so
every design criterion acts on one feature of this ellipse:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - ``pp_ob``
     - optimises
     - which means, for the region
   * - ``'D'``
     - max :math:`\det\mathbf{M}`
     - smallest **area** (volume), whatever the shape
   * - ``'A'``
     - min :math:`\mathrm{tr}\,\mathbf{M}^{-1}`
     - smallest **box** around it (the projections, :math:`a^2+b^2`)
   * - ``'E'``
     - max :math:`\lambda_{\min}(\mathbf{M})`
     - shortest **longest axis**, the worst-determined direction
   * - ``'ME'``
     - min :math:`\mathrm{cond}\,\mathbf{M}`
     - **roundest** region (:math:`a/b` closest to 1), not necessarily small

The scale of the region is fixed (choose it under the plot), so what you see shrinking really shrinks;
the dashed ellipse is the starting design, for comparison.

.. raw:: html

   <div class="mdw" data-widget="sampling"></div>

.. rubric:: Try this

- Put every sample after t = 6 (set the scale to ±120): A is well known but the ellipse is long in the
  k direction, since late samples carry almost nothing about k.
- Bunch them all near t = 1: the ellipse gets thin, tilted and much larger. A and k are then strongly
  correlated (\|ρ\| close to 1): the data fix a combination of them, not each one.
- Switch the minimum gap off and press **D-optimal**: the samples go to just two times, about
  t = 1.9 (twice) and t = 10, one near the peak of the k-sensitivity and one at the end. With two
  parameters, two well-chosen times carry all the information, so repeating them beats spreading out.
  MIDDoE's ``offt`` keeps samples apart, as the switch does.
- Then press **A**, **E** and **ME** in turn, and hover each button to see what it acts on. D gives the
  smallest area (about 350 %²); A and E move the early samples to about t = 1.4–1.5 and shorten the
  long axis slightly; ME moves them to about t = 1.1 and makes the region the roundest
  (:math:`a/b` 3.3 instead of 3.9) but *larger* (about 400 %²). The same data, four different ideas of
  "best".

In MIDDoE: ``des_opt['pp_ob']`` chooses the criterion (``'D'``, ``'A'``, ``'E'``, ``'ME'``);
``system['tvo'][y]['sp']`` sets the number of samples and ``'offt'`` the minimum gap. The
sensitivities and :math:`\mathbf{M}` are computed by finite differences on your model. See
:ref:`choosing-settings` (Design tab).

.. _ia-discrim:

Rival models: design an experiment that separates them
------------------------------------------------------

Two kinetic models fit the first experiment (initial concentration :math:`c_0 = 1`) almost equally
well: first order, whose conversion does not depend on :math:`c_0`, and second order, whose does.
Data at :math:`c_0 = 1` cannot tell them apart. Model discrimination designs the experiment where
their predictions differ most compared with the noise, here measured by the Hunter-Reiner criterion
:math:`\mathrm{HR}=\sum_i (\hat y_{1,i}-\hat y_{2,i})^2/\sigma^2`.

.. raw:: html

   <div class="mdw" data-widget="discrim"></div>

.. rubric:: Try this

- Move :math:`c_0` away from 1 in both directions: the second-order prediction moves, the first-order
  one does not, and the gap (vertical bars at the samples) grows.
- Increase σ: the same gap becomes harder to detect; the verdict depends on the gap *relative to* σ.
- Press **Design to discriminate**: it picks :math:`c_0` and the sampling times that maximise HR.

The verdict compares the lack of fit expected for the wrong model, :math:`n + \mathrm{HR}`, with the
95 % limit of :math:`\chi^2(n)`; it ignores that the wrong model would be refitted, so it is an
optimistic guide. In MIDDoE: ``mbdoe_md`` with ``des_opt['md_ob'] = 'HR'``, or ``'BFF'``, which also
accounts for the parameter uncertainty of each model (recommended). After the experiment,
``save_rounds`` reports each model's probability P.

.. _ia-noise:

Measurement error: absolute, relative, and the floor
----------------------------------------------------

σ weights every point in the fit (:math:`1/\sigma^2`, lower strip) and sets the width of every
confidence interval. With an **absolute** error every point counts the same. With a **relative**
error, :math:`\sigma = \mathrm{unc}\cdot|y|`: small values are trusted more, and a value of zero would
be trusted infinitely, so MIDDoE floors σ at ``sigma_floor``.

.. raw:: html

   <div class="mdw" data-widget="noise"></div>

.. rubric:: Try this

- With ``errt = 'rel'`` and the default floor 1e-6, look at the weight of the point at t = 0 where
  y = 0: it carries almost all the weight, so the fit is pinned to that one point.
- Raise ``sigma_floor``: at 1e-2 that point's share drops from 100 % to about a third, at 1e-1 the floor
  exceeds every ``unc·|y|``, so all weights are equal. A sensible floor is the smallest σ your instrument really achieves.
- Switch to ``'abs'``: the band has constant width and every point weighs the same.

In MIDDoE: ``system['tvo'][y]['unc']`` and ``errt`` (``insilicos['errt']`` for in-silico data);
the floor is ``insilicos['sigma_floor']`` for the generated ``MES_E`` and ``system['sigma_floor']`` for
the design. For your own data, ``MES_E`` holds each point's σ. See :ref:`choosing-settings`
(Error & objective tab).
