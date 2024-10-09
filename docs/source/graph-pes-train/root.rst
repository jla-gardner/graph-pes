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

.. toctree::
    :hidden:
    :maxdepth: 2

    configuration