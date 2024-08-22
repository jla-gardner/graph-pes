.. _cli-reference:

``graph-pes-train``
===================

``graph-pes-train`` is a command line tool for training graph-based potential energy surface models:

.. code-block:: console

    $ graph-pes-train -h
    usage: graph-pes-train [-h] [--config CONFIG] [overrides [overrides ...]]

    Train a GraphPES model from a configuration file using PyTorch Lightning.

    positional arguments:
    overrides        Config overrides in the form nested^key=value, separated by spaces, e.g.
                     fitting^loader_kwargs^batch_size=32.

    optional arguments:
    -h, --help       show this help message and exit
    --config CONFIG  Path to the configuration file. This argument can be used multiple times, with later files taking
                     precedence over earlier ones in the case of conflicts. If no config files are provided, the script
                     will auto-generate.

    Example usage: graph-pes-train --config config1.yaml --config config2.yaml fitting^loader_kwargs^batch_size=32


.. toctree::
    :hidden:
    :maxdepth: 2

    configuration