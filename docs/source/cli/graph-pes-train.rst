TODO: write this

.. _cli-reference:

``graph-pes-train``
===================

``graph-pes-train`` is a command line tool for training graph-based potential energy surface models:

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



Configuration
-------------

Configuration for the ``graph-pes-train`` command line tool is represented as a nested dictionary. The values of this dictionary are sourced from three places:

1. the default values defined in `defaults.yaml <https://github.com/jla-gardner/graph-pes/blob/main/src/graph_pes/config/defaults.yaml>`_
2. values you define in the config file/s you pass to ``graph-pes-train <config-1.yaml> <config-2.yaml> ...``
3. additional command line arguments you pass to ``graph-pes-train``

Command line arguments overwrite values in your config file, which again overwrite the defaults.
Hence:

.. code-block:: bash

    graph-pes-train minimal.yaml model^graph_pes.model.SchNet^layers=2


will train a model :class:`~graph_pes.models.SchNet` model with **2** layers (rather than the 3 specified in :ref:`minimal.yaml <minimal config>`).


Under-the-hood, ``graph-pes-train`` uses `dacite <https://github.com/konradhalas/dacite/tree/master/>`_ to convert the configuration dictionary into a series of nested dataclasses, the structure of which is defined in :class:`~graph_pes.config.Config`. 

All available configuration options are documented below. For example config files, see the bottom of this page.

.. autoclass:: graph_pes.config.Config()
    :members:
    :exclude-members: hash

.. autoclass:: graph_pes.config.spec.FittingConfig()
    :members:
    :inherited-members:

.. autoclass:: graph_pes.config.spec.SWAConfig()
    :members:

.. autoclass:: graph_pes.config.spec.GeneralConfig()
    :members:

.. autoclass:: graph_pes.config.spec.LossSpec()
    :members:


Defaults
--------

The default configuration is defined in `defaults.yaml <https://github.com/jla-gardner/graph-pes/blob/main/src/graph_pes/config/defaults.yaml>`_:

.. literalinclude:: ../../../src/graph_pes/config/defaults.yaml
    :language: yaml
    :caption: defaults.yaml


Examples
--------

.. _minimal config:

Minimal
+++++++

Train ...

.. literalinclude:: ../../../configs/minimal.yaml
    :language: yaml
    :caption: minimal.yaml