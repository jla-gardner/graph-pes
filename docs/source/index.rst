.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart/root


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: CLI Reference

    cli/graph-pes-train/root
    cli/graph-pes-resume
    cli/graph-pes-test
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
    :caption: Interfaces
    :hidden:

    interfaces/mace
    interfaces/mattersim
    interfaces/orb
    
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

**Date:** |today| - **Author:** `John Gardner <https://jla-gardner.github.io>`__ - **Version:** |release|

``graph-pes`` is a package designed to accelerate the development of machine-learned potential energy surfaces (ML-PESs) that act on graph representations of atomic structures. 


The core component of ``graph-pes`` is the :class:`~graph_pes.GraphPESModel`. 
You can take **any** model that inherits from this class and:

* train and/or fine-tune it on your own data using the ``graph-pes-train`` command line tool
* use it to drive MD simulations via :doc:`LAMMPS <tools/lammps>` or :doc:`ASE <tools/ase>`

We provide many :class:`~graph_pes.GraphPESModel`\ s, including:

* re-implementations of popular architectures, including :class:`~graph_pes.models.NequIP`, :class:`~graph_pes.models.PaiNN`, :class:`~graph_pes.models.MACE` and :class:`~graph_pes.models.TensorNet`
* wrappers for other popular ML-PES frameworks, including :doc:`mace-torch <interfaces/mace>`, :doc:`mattersim <interfaces/mattersim>`, and :doc:`orb-models <interfaces/orb>`, that convert their models into ``graph-pes`` compatible :class:`~graph_pes.GraphPESModel` instances

Use ``graph-pes`` to train models from scratch, experiment with new architectures, write architecture-agnostic validation pipelines, and try out different foundation models with minimal code changes.


**Useful links**:

.. grid:: 1 2 3 3
    :gutter: 3



    .. grid-item-card:: 🔥 Train
        :link: quickstart/quickstart
        :link-type: doc
        :text-align: center

        Train an existing architecture from scratch

    .. grid-item-card:: 🔍 Analyse
        :link: https://jla-gardner.github.io/graph-pes/quickstart/quickstart.html#Model-analysis
        :text-align: center

        Analyse a trained model

    .. grid-item-card:: 🔧 Fine-tune
        :link: quickstart/fine-tuning
        :link-type: doc
        :text-align: center

        Fine-tune a foundation model on your data

    .. grid-item-card:: 🎓 Learn
        :link: theory
        :link-type: doc
        :text-align: center

        How does ``graph-pes`` work under the hood?

    .. grid-item-card:: 🔨 Build
        :link: quickstart/implement-a-model
        :link-type: doc
        :text-align: center

        Implement your own ML-PES architecture



**Installation:**

Install ``graph-pes`` using pip. We recommend doing this in a new environment (e.g. using conda):

.. code-block:: bash

    conda create -n graph-pes python=3.10 -y
    conda activate graph-pes
    pip install graph-pes

Please see the `GitHub repository <https://github.com/jla-gardner/graph-pes>`__ for the source code and to report issues.
