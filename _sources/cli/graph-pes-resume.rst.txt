``graph-pes-resume``
====================

``graph-pes-resume`` is a command line tool for resuming training runs that have been interrupted:


.. code-block:: console

    $ graph-pes-resume -h
    usage: graph-pes-resume [-h] train_directory

    Resume a `graph-pes-train` training run.

    positional arguments:
    train_directory  Path to the training directory. 
                     For instance, `graph-pes-results/abdcefg_hijklmn`

    optional arguments:
    -h, --help       show this help message and exit

    Copyright 2023-35, John Gardner


Usage
-----

.. code-block:: bash

    $ graph-pes-resume graph-pes-results/abdcefg_hijklmn
