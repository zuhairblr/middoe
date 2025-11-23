MIDDoE Documentation
====================

**MIDDoE** (Model-based Identification, Discrimination, and Design of Experiments)
is an open-source Python package for model identification of dynamic lumped models.

.. image:: https://badge.fury.io/py/middoe.svg
   :target: https://pypi.org/project/middoe/
   :alt: PyPI version

Installation
------------

.. code-block:: bash

   pip install middoe

Quick Example
-------------

.. code-block:: python

   from middoe import parmest, uncert

   # Run parameter estimation
   results = parmest(system, models, iden_opt, data=data)
   results = uncert(results, system, models, iden_opt)

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
