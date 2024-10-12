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
    data
    fitting/root
    building-blocks/root

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

    theory
    tech-stack
    development


.. image:: _static/logo-text.svg
    :align: center
    :alt: graph-pes logo
    :width: 90%
    :target: .


#########
graph-pes
#########

.. raw:: html
    :file: hide-title.html


``graph-pes`` provides the following functionality:

1. a set of :doc:`command line tools <graph-pes-train/root>` for training state-of-the-art ML-PESs. We've re-implemented several popular models, including :class:`~graph_pes.models.NequIP`, :class:`~graph_pes.models.PaiNN`, :class:`~graph_pes.models.MACE` and :class:`~graph_pes.models.TensorNet`. Easily train any of these from a unified interface, and also configure distributed training, learning rate scheduling, `Weights & Biases <https://wandb.ai>`__ logging, and other useful features.

2. a `LAMMPS <https://docs.lammps.org/Manual.html>`__ ``pair_style graph_pes`` plugin for using any :class:`~graph_pes.core.GraphPESModel` to drive MD simulations, together with a set of tools to easily build LAMMPS executables for this purpose.

3. a mature, well-documented and completely hackable codebase containing:
    * a data pipeline for turning :class:`ase.Atoms` into :class:`~graph_pes.graphs.AtomicGraph`
    * a wide selection of ``TorchScript``- and periodic-boundary-condition compatible functions for common operations on :class:`~graph_pes.graphs.AtomicGraph` objects, including batching, edge-trimming, neighbour selection and summation etc.
    * a set of base classes (:class:`~graph_pes.core.GraphPESModel`, :class:`~graph_pes.models.PairPotential`, :class:`~graph_pes.models.AdditionModel`, etc.) for easy experimentation with new model architectures: inherit from these classes, implement the relevant energy prediction method, and ``graph-pes`` will handle force and stress predictions automatically. Any new architectures you create are completely compatible with ``graph-pes-train`` and ``pair_style graph_pes``
    * a set of common building blocks for constructing new models, including :class:`~graph_pes.models.distances.DistanceExpansion`, :class:`~graph_pes.models.distances.Envelope`, :class:`~graph_pes.nn.PerElementParameter` and :class:`~graph_pes.nn.MLP`


Quick-start
===========

Use our :doc:`command line tools <graph-pes-train/root>` to:

* :doc:`train a model from scratch <quickstart/train-a-model>`
* :doc:`fine-tune an existing model <quickstart/model-fine-tuning>`
* :doc:`run an MD simulation in LAMMPs <md/lammps>` 

Or get your hands dirty with the code by:

* :doc:`analysing your model in a python notebook <quickstart/model-analysis>`
* :doc:`implementing your own model <quickstart/implement-a-model>`, train it from the command line and use it to drive MD.
* using our mature data pipeline to :doc:`implement your own training loop <quickstart/custom-training-loop>`


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

