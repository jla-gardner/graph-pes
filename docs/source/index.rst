.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart/root


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: CLI Reference

    cli/graph-pes-train
    cli/graph-pes-id

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: API Reference

    data/root
    models/root
    fitting/root
    building-blocks/root
    utils

.. toctree::
    :maxdepth: 2
    :caption: Tools
    :hidden:

    tools/ase
    tools/lammps
    tools/analysis


.. toctree::
    :maxdepth: 2
    :caption: About
    :hidden:

    theory
    development


.. image:: _static/logo-text.svg
    :align: center
    :alt: graph-pes logo
    :width: 70%
    :target: .


#########
graph-pes
#########

.. raw:: html
    :file: hide-title.html

``graph-pes`` is a set of tools designed to accelerate the development of machine-learned potential energy surfaces (ML-PESs) that act on graph representations of atomic structures. 

A 3-in-1 toolset:
=================

.. dropdown:: ``graph_pes``: a Python library of components for training ML-PESs

   This includes:

   - a data pipeline for turning :class:`ase.Atoms` objects into :class:`~graph_pes.AtomicGraph` objects
   - a set of operations for common tasks on :class:`~graph_pes.AtomicGraph` objects, including :func:`edge-trimming <graph_pes.atomic_graph.trim_edges>`, :func:`neighbour indexing <graph_pes.atomic_graph.index_over_neighbours>`, :func:`summations <graph_pes.atomic_graph.sum_over_neighbours>`, and :func:`batching <graph_pes.atomic_graph.to_batch>`,
   - a base class for all model of the potential energy surface (:class:`~graph_pes.GraphPESModel`) that automatically infers force and stress predictions if these are not provided by the model's implementation
   - reference implementations of popular models, including :class:`~graph_pes.models.NequIP`, :class:`~graph_pes.models.PaiNN`, :class:`~graph_pes.models.MACE` and :class:`~graph_pes.models.TensorNet`
   
   ``graph_pes`` is written in vanilla ``PyTorch`` that is 100% compatible with ``TorchScript`` compilation for use within LAMMPS

.. dropdown:: ``graph-pes-train``: a command line tool for training ML-PESs

   - get up and running quickly with sensible defaults to train new models from scratch
   - fine-tune existing models on new datasets
   - easily configure advanced features such as :ref:`distributed training <multi-GPU training>`, :ref:`learning rate scheduling <learning rate scheduler>`, :ref:`stochastic weight averaging <stochastic weight averaging>`, and logging to `Weights & Biases <https://wandb.ai>`__

.. dropdown:: ``pair_style graph_pes``: a LAMMPS pair style for GPU-accelerated MD

   - use this to drive GPU-accelerated molecular dynamics (MD) simulations with any model that inherits from :class:`~graph_pes.GraphPESModel` (i.e. both ones we've implemented and also your own)
   - we've included helper scripts to automate the `LAMMPS <https://docs.lammps.org/Manual.html>`__ build process for you

Quick-start
===========

.. code-block:: bash

    pip install graph-pes
    wget wget https://tinyurl.com/graph-pes-quickstart-cgap17
    graph-pes-train quickstart-cgap17.yaml

Alternatively, open any of these notebooks to get started. Install ``graph-pes`` to follow along locally, or run the code in the cloud using `Google Colab <https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb>`__.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card::
        :text-align: center

        .. button-link:: quickstart/quickstart.html#Train-a-model
            :click-parent:

            Train a model
    
    .. grid-item-card::
        :text-align: center

        .. button-link:: quickstart/quickstart.html#Model-analysis
            :click-parent:

            Analyse a model

    .. grid-item-card::
        :text-align: center

        .. button-link:: quickstart/quickstart.html#Fine-tuning
            :click-parent:

            Fine-tune a model

    .. grid-item-card::
        :text-align: center

        .. button-ref:: tools/lammps
            :ref-type: doc
            :click-parent:

            Run LAMMPS MD

    .. grid-item-card::
        :text-align: center

        .. button-ref:: quickstart/implement-a-model
            :ref-type: doc
            :click-parent:

            Implement your own model

    .. grid-item-card::
        :text-align: center

        .. button-ref:: quickstart/custom-training-loop
            :ref-type: doc
            :click-parent:

            Implement your own training loop


Installation
============

We recommend installing ``graph-pes`` in a new environment, e.g. using `conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_:

.. code-block:: bash

    conda create -n graph-pes python=3.10 -y
    conda activate graph-pes

Install ``graph-pes`` from PyPI using pip (installs all dependencies):

.. code-block:: bash

    pip install graph-pes



