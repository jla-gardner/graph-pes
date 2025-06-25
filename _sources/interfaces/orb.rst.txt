``orb-models``
==============


``graph-pes`` supports the conversion of arbitrary ``orb-models`` models to :class:`~graph_pes.GraphPESModel` objects via the :class:`~graph_pes.interfaces._orb.OrbWrapper` class.

Use the :func:`~graph_pes.interfaces.orb_model` function to load a pre-trained ``orb-models`` model and convert it into a :class:`~graph_pes.GraphPESModel`. You can then use this model in the same way as any other :class:`~graph_pes.GraphPESModel`, for instance by :doc:`fine-tuning it <../quickstart/fine-tuning>` or using it to run MD via 
:doc:`torch-sim <../tools/torch-sim>`,
:doc:`ASE <../tools/ase>` or :doc:`LAMMPS <../tools/lammps>`:

.. code-block:: python

   from graph_pes.interfaces import orb_model
   from graph_pes import GraphPESModel

   model = orb_model()
   assert isinstance(model, GraphPESModel)

   # do stuff ...


You can also reference the :func:`~graph_pes.interfaces.orb_model` function in your training configs for :doc:`graph-pes-train <../cli/graph-pes-train/root>`:

.. code-block:: yaml

    model:
        +orb_model:
            name: orb-v3-direct-20-omat



If you use any ``orb-models`` models in your work, please visit the `orb-models <https://github.com/orbital-materials/orb-models>`_ repository and cite the following:

.. code-block:: bibtex

    @misc{rhodes2025orbv3atomisticsimulationscale,
        title={Orb-v3: atomistic simulation at scale}, 
        author={
            Benjamin Rhodes and Sander Vandenhaute and Vaidotas Šimkus 
            and James Gin and Jonathan Godwin and Tim Duignan and Mark Neumann
        },
        year={2025},
        eprint={2504.06231},
        archivePrefix={arXiv},
        primaryClass={cond-mat.mtrl-sci},
        url={https://arxiv.org/abs/2504.06231}, 
    }

    @misc{neumann2024orbfastscalableneural,
        title={Orb: A Fast, Scalable Neural Network Potential}, 
        author={
            Mark Neumann and James Gin and Benjamin Rhodes 
            and Steven Bennett and Zhiyi Li and Hitarth Choubisa 
            and Arthur Hussey and Jonathan Godwin
        },
        year={2024},
        eprint={2410.22570},
        archivePrefix={arXiv},
        primaryClass={cond-mat.mtrl-sci},
        url={https://arxiv.org/abs/2410.22570}, 
    }


Installation
------------

To install ``graph-pes`` with support for ``orb-models`` models, you need to install
the `orb-models <https://github.com/orbital-materials/orb-models>`_ package alongside ``graph-pes``. We recommend doing this in a new environment:

.. code-block:: bash

   conda create -n graph-pes-orb python=3.10
   conda activate graph-pes-orb
   pip install graph-pes orb-models


Interface
---------

.. autofunction:: graph_pes.interfaces.orb_model

.. autoclass:: graph_pes.interfaces._orb.OrbWrapper
   :members: orb_model