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
                nested/key=value. Final config is built up from 
                these items in a left to right manner, with later 
                items taking precedence over earlier ones in the 
                case of conflicts. The data2objects package is used 
                to resolve references and create objects directly 
                from the config dictionary.

    optional arguments:
    -h, --help  show this help message and exit

    Copyright 2023-24, John Gardner

``graph-pes-train``:

1. loads in your data, model, loss function, etc. from the config files you pass to it.
2. pre-fits the model on the training data, if you have specified the ``pre_fit_model`` flag.
3. trains the model using the training and validation data.
4. saves the best model for later use, as well as "deploying it" for `use in LAMMPS <../tools/lammps.html>`__
5. tests the best model on the training and validation data, and any other test data you have specified.



Usage
-----

Train from a config file:

.. code-block:: bash

    $ graph-pes-train config.yaml 

Train from a config file, overriding a specific option:

.. code-block:: bash

    $ graph-pes-train config.yaml fitting/trainer_kwargs/max_epochs=10

Train from multiple config files:

.. code-block:: bash

    $ graph-pes-train config-1.yaml config-2.yaml


.. _multi-GPU training:

Multi-GPU training:
+++++++++++++++++++

The ``graph-pes-train`` command supports multi-GPU out of the box, relying on PyTorch Lightning's native support for distributed training.
By default, ``graph-pes-train`` will attempt to use all available GPUs. You can override this by exporting the ``CUDA_VISIBLE_DEVICES`` environment variable:

.. code-block:: bash

    $ export CUDA_VISIBLE_DEVICES=0,1
    $ graph-pes-train config.yaml

Weights and Biases
+++++++++++++++++++

By default, ``graph-pes-train`` will try to use `Weights and Biases <https://wandb.ai/site>`__ for logging (see the ``wandb`` section of the config below).

When you run ``graph-pes-train`` interactively, you will be prompted to log in to your W&B account before training begins.
To avoid this interactive step, you can do one of the following:

1. set the ``wandb: null`` flag in your config file to all disable logging.
2. run ``wandb login`` before running ``graph-pes-train`` to log in to your W&B account permanently on your machine.
3. set the ``WANDB_API_KEY`` environment variable to your W&B API key in your shell and directly before running ``graph-pes-train``.

.. warning::
    
    If you submit a job to some job scheduler (e.g. SLURM, PBS, etc.), without taking one of the 3 steps above, your job will hang forever while ``graph-pes-train`` waits for you to log in to your W&B account.

Config files
------------

Configuration for the ``graph-pes-train`` command line tool is represented as a nested dictionary. The values of this dictionary are sourced from three places:

1. the default values defined in `training-defaults.yaml <https://github.com/jla-gardner/graph-pes/blob/main/src/graph_pes/config/training-defaults.yaml>`_
2. values you define in the config file/s you pass to ``graph-pes-train``:  ``<config-1.yaml> <config-2.yaml> ...``
3. additional command line arguments you pass to ``graph-pes-train``: ``<nested/key=value> <nested/key=value> ...``

The final nested configuration dictionary is built up from these values in a left to right manner, with later items taking precedence over earlier ones in the case of conflicts.

The structure of config files used by ``graph-pes-train`` is that of a ``.yaml`` file containing a nested dictionary of items:

.. _minimal config:

.. literalinclude:: ../../../configs/minimal.yaml
    :caption: ``minimal.yaml``
    :language: yaml

For a full list of available options in these config files, see :ref:`the below API reference <complete config API>` and :ref:`the kitchen sink config example <kitchen sink config>`.

.. note::
    
    Under-the-hood, ``graph-pes-train`` turns the final nested config dictionary into a :class:`~graph_pes.config.training.TrainingConfig` object via a 3 step process:

    1. all reference strings (of the form ``=/absolute/path/to/object`` or ``=../relative/path/to/object``) are replaced with the corresponding value.
       For example:

       .. code-block:: yaml

           a: {b: "=/c"}  # absolute reference
           c: 2
           d: {e: "=../c"}  # relative reference

       will be transformed into:

       .. code-block:: yaml

           a: {b: 2}
           c: 2
           d: {e: 2}
    
    2. all "special" keys starting with a ``+`` are interpreted as references to Python objects. These are imported and replaced with the actual 
       object they point to using the `data2objects <https://github.com/jla-gardner/data2objects>`__ package - see there for more details.
       You can use this format to point to any Python object, defined in ``graph-pes`` or otherwise! To point to your own custom classes/objects/functions,
       use the fully-qualified name, e.g. ``+my_module.MyModelClass``. By default, ``graph-pes-train`` will look for objects within the following modules:

       .. code-block:: python

           graph_pes
           graph_pes.models
           graph_pes.training
           graph_pes.training.opt
           graph_pes.training.loss
           graph_pes.data

       and hence ``+NequIP`` is shorthand for ``+graph_pes.models.NequIP``.
       Ending the name with ``()`` will call the function/class constructor with no arguments. Pointing the key to a nested dictionary will pass those values as keyword arguments to the constructor. Hence, above, ``+PerAtomEnergyLoss()`` will resolve to a ``~graph_pes.training.loss.PerAtomEnergyLoss`` object, while the :class:`~graph_pes.models.SchNet` model will be constructed with the keyword arguments specified in the config.
    
    3. the resulting dictionary of python objects is then converted, using `dacite <https://github.com/konradhalas/dacite/tree/master/>`__, into a final nested :class:`~graph_pes.config.training.TrainingConfig` object.

Example configs
+++++++++++++++

The minimal config required to use ``graph-pes-train`` provides a model, some data and a loss function, as can be seen :ref:`above <minimal config>`.
This config can be so small because ``graph-pes-train`` loads in 
default values from ``training-defaults.yaml`` and overwrites them with any 
values you specify in the config file (see below):

.. dropdown:: ``training-defaults.yaml``

    .. literalinclude:: ../../../src/graph_pes/config/training-defaults.yaml
        :language: yaml

Note the use of the :attr:`~graph_pes.config.training.TrainingConfig.misc` section of the config to store constants, which can then be referenced multiple times elsewhere in the config - this allows you to easily scan over these constants in an elegant manner:

.. code-block:: bash

    $ for cutoff in 3.5 4.5 5.5; do
    >     graph-pes-train realistic.yaml misc/CUTOFF=$cutoff
    > done

A more realistic configuration might look like this:

.. dropdown:: ``realistic.yaml``

    We train a :class:`~graph_pes.models.MACE` model (accounting for core repulsions and arbitraty energy offsets) on the `C-GAP-20U dataset <https://jla-gardner.github.io/load-atoms/datasets/C-GAP-20U.html>`__:

    .. literalinclude:: ../../../configs/realistic.yaml
        :language: yaml


Finally, here is a `"kitchen sink"` config that attempts to specify every possible option:

.. _kitchen sink config:
.. dropdown:: ``kitchen-sink.yaml``

    .. literalinclude:: ../../../configs/kitchen-sink.yaml
        :language: yaml


For more information as to the structure of these config files, and the various options available, read on:


.. _complete config API:

Complete ``config`` API
-----------------------

.. autoclass:: graph_pes.config.training.TrainingConfig()
    :members:




