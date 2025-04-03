``graph-pes-test``
==================

Use the ``graph-pes-test`` command to test a trained model. 
Testing functionality is already baked into ``graph-pes-train``, but this command
allows you more fine-grained control over the testing process.


Usage
-----

Simplest possible usage - test the model at ``path/to/model.pth`` on the datasets
found in ``path/to/model.pt/../training-config.yaml``.

.. code-block:: bash

    graph-pes-test model_path=path/to/model.pth


Alternatively, to test on new data, pass a path to a new config file that specifies
a :class:`~graph_pes.config.testing.TestingConfig` object:

.. code-block:: yaml

    graph-pes-test test-config.yaml model_path=path/to/model.pth

Where ``test-config.yaml`` contains e.g.:

.. code-block:: yaml

    data:
        dimers:
            +file_dataset:
                path: path/to/dimers.xyz
                cutoff: 5.0
        amorphous:
            +file_dataset:
                path: path/to/amorphous.xyz
                cutoff: 5.0

    accelerator: gpu

    loader_kwargs:
        batch_size: 64
        num_workers: 4

Complete usage:

.. code-block:: bash

    graph-pes-test -h

    usage: graph-pes-test [-h] [args ...]

    Test a GraphPES model using PyTorch Lightning.

    positional arguments:
    args        Config files and command line specifications. 
                Config files should be YAML (.yaml/.yml) files. 
                Command line specifications should be in the form 
                my/nested/key=value. Final config is built up from 
                these items in a left to right manner, with later 
                items taking precedence over earlier ones in the 
                case of conflicts. The data2objects package is used 
                to resolve references and create objects directly 
                from the config dictionary.

    optional arguments:
    -h, --help  show this help message and exit


Config
------

.. autoclass:: graph_pes.config.testing.TestingConfig()
    :members:
