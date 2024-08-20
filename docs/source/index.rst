.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart/root


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: CLI Reference

    graph-pes-train/root

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: API Reference

    models/root
    building-blocks/root
    data
    losses

.. toctree::
    :maxdepth: 2
    :caption: Tools
    :hidden:

    md/root
    analysis
    examples


.. toctree::
    :maxdepth: 2
    :caption: About
    :hidden:

    tech-stack
    development


.. image:: _static/logo-light.svg
    :align: center
    :alt: graph-pes logo
    :width: 90%
    :target: .
    :class: only-light

.. image:: _static/logo-dark.svg
    :align: center
    :alt: graph-pes logo
    :width: 90%
    :target: .
    :class: only-dark

#########
graph-pes
#########

.. raw:: html
    :file: hide-title.html

``graph-pes`` is a framework built to accelerate the development of machine-learned potential energy surface (PES) models that act on graph representations of atomic structures.

Use ``graph-pes`` to easily do the following:

#. experiment with new model architectures by inheriting from our :class:`~graph_pes.core.ConservativePESModel` or :class:`~graph_pes.models.scaling.AutoScaledPESModel` base classes
#. train your own or existing (e.g. SchNet, NequIP, PaiNN, MACE, etc.) models. Easily configure distributed training, learning rate scheduling, weights and biases logging, and other features using our ``graph-pes-train`` command line interface, or use our data-loading pipeline within your own training loop
#. run molecular dynamics simulations via LAMMPS (or ASE) using any :class:`~graph_pes.core.ConservativePESModel` and the ``pair_style graph_pes`` LAMMPS command


Installation
============

Install ``graph-pes`` from PyPI using pip (installs all dependencies):

.. code-block:: bash

    pip install graph-pes

.. tip::

    We recommend installing ``graph-pes`` in a new environment, e.g. using `conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_:

    .. code-block:: bash

        conda create -n graph-pes python=3.10 -y
        conda activate graph-pes


Quickstart
==========

See the menu-bar on the left for a complete API and usage guide, or jump straight in with the following quickstart guides:

* :doc:`train a model from the command line <quickstart/train-a-model>`
* implement your own model in xxx lines, and train it
* implement a custom training loop
* load a trained model into a python notebook for analysis
* run MD simulations using LAMMPS
* fine-tune a model
* sweep?
