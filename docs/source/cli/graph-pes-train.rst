.. _cli-reference:

``graph-pes-train``
===================

``graph-pes-train`` is a command line tool for training graph-based potential energy surface models using `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/index.html>`__:

.. code-block:: console

    $ graph-pes-train -h
    usage: graph-pes-train [-h] [args [args ...]]

    Train a GraphPES model using PyTorch Lightning.

    positional arguments:
    args        Config files and command line specifications. 
                Config files should be YAML (.yaml/.yml) files. 
                Command line specifications should be in the form 
                nested^key=value. Final config is built up from 
                these items in a left to right manner, with later 
                items taking precedence over earlier ones in the 
                case of conflicts.

    optional arguments:
    -h, --help  show this help message and exit

    Copyright 2023-24, John Gardner

Usage
-----

Train from a config file:

.. code-block:: bash

    $ graph-pes-train config.yaml 

Train from a config file, overriding a specific option:

.. code-block:: bash

    $ graph-pes-train config.yaml fitting^trainer_kwargs^max_epochs=10

Train from multiple config files:

.. code-block:: bash

    $ graph-pes-train config-1.yaml config-2.yaml


Example configs:
----------------

.. _minimal config:
.. dropdown:: A minimal config: ``minimal.yaml``

    .. literalinclude:: ../../../configs/minimal.yaml
        :language: yaml
        :caption: minimal.yaml

    This config can be so small because ``graph-pes-train`` loads in 
    default values from ``defaults.yaml`` and overwrites them with any 
    values you specify in the config file:

    .. literalinclude:: ../../../src/graph_pes/config/defaults.yaml
        :language: yaml
        :caption: defaults.yaml


Complete ``config`` API
-----------------------

Configuration for the ``graph-pes-train`` command line tool is represented as a nested dictionary. The values of this dictionary are sourced from three places:

1. the default values defined in `defaults.yaml <https://github.com/jla-gardner/graph-pes/blob/main/src/graph_pes/config/defaults.yaml>`_
2. values you define in the config file/s you pass to ``graph-pes-train``:  ``<config-1.yaml> <config-2.yaml> ...``
3. additional command line arguments you pass to ``graph-pes-train``: ``<nested^key=value> <nested^key=value> ...``

The final nested configuration dictionary is built up from these values in a left to right manner, with later items taking precedence over earlier ones in the case of conflicts.

For example, to train a :class:`~graph_pes.models.SchNet` model with 2 layers, you would run:

.. code-block:: bash

    graph-pes-train minimal.yaml model^graph_pes.model.SchNet^layers=2


will train a model :class:`~graph_pes.models.SchNet` model with **2** layers (rather than the 3 specified in :ref:`minimal.yaml <minimal config>`).


Under-the-hood, ``graph-pes-train`` uses `dacite <https://github.com/konradhalas/dacite/tree/master/>`_ to convert this nested configuration dictionary into a series of nested dataclasses, the structure of which is defined by the :class:`~graph_pes.config.Config` dataclass, and subsequent children.

.. note::

    Several configuration options expect you to point towards some data
    that can be turned into a Python object. The format for this is as follows. To point to:

    * an object defined in some module, simplify specify the **fully qualified name**,
      e.g. ``my_module.my_object``.
    * a function that you want to use the return value of, append ``()`` to the name, e.g.
      ``my_module.my_function()``. 
    * the return value of some function/class constructor for which you also want to provide arguments, use a nested dictionary structure like so:

      .. code-block:: yaml

            graph_pes.models.SchNet:
                cutoff: 5.0
                n_layers: 3

      hence another way to point to the return value of some function with no arguments is:

      .. code-block:: yaml

            my_module.my_function: {}

    Use this format to point to any Python object, defined in ``graph-pes`` or otherwise. To prevent malicious execution of
    arbitrary code, ``graph-pes-train`` will only attempt to import and/or execute code that you have marked as safe using
    a ``,`` seperated list of module names exported to the ``GRAPH_PES_ALLOW_IMPORT`` environment variable. The above therefore requires:
    ``GRAPH_PES_ALLOW_IMPORT=my_module graph-pes-train <config.yaml>``.

.. autoclass:: graph_pes.config.Config()
    :members:
    :exclude-members: hash
