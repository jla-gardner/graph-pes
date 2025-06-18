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
                case of conflicts.

    optional arguments:
    -h, --help  show this help message and exit

    Copyright 2023-25, John Gardner


.. toctree::
    :maxdepth: 1
    :hidden:

    the-basics
    complete-docs
    examples

For a hands-on introduction, try our `quickstart Colab notebook <https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb>`__. Alternatively, you can learn about how to use ``graph-pes-train`` from :doc:`the basics guide <the-basics>`, :doc:`the complete configuration documentation <complete-docs>` or :doc:`a set of examples <examples>`.

\

There are a few important things to note when using ``graph-pes-train`` in special situations:


.. _multi-GPU training:

Multi-GPU training:
-------------------

The ``graph-pes-train`` command supports multi-GPU out of the box, relying on PyTorch Lightning's native support for distributed training.
**By default, ``graph-pes-train`` will attempt to use all available GPUs.** You can override this by exporting the ``CUDA_VISIBLE_DEVICES`` environment variable:

.. code-block:: bash

    $ export CUDA_VISIBLE_DEVICES=0,1
    $ graph-pes-train config.yaml


If you are running ``graph-pes-train`` on a SLURM-managed cluster, you can use the ``srun`` command to run the training job.
If you are requesting 4 GPUs, use a config similar to this:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --tasks-per-node=4
    #SBATCH --gpus-per-node=4
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=256gb
    #SBATCH ... (more config options relevant to your job)

    srun graph-pes-train config.yaml fitting/trainer_kwargs/devices=4


Non-interactive jobs
--------------------

In cases were you are running ``graph-pes-train`` in a non-interactive session (e.g. from a script or scheduled job) and where you wish to make use of the `Weights and Biases <https://wandb.ai/site>`__ logging functionality, you will need to take one of the following steps:

1. run ``wandb login`` in an interactive session beforehand - this will cache your credentials to ``~/.netrc``
2. set the ``WANDB_API_KEY`` environment variable to your W&B API key directly before running ``graph-pes-train``

Failing to do this will result in ``graph-pes-train`` hanging forever while waiting for you to log in to your W&B account.

Alternatively, you can set the ``wandb: null`` flag in your config file to disable W&B logging.


Compute clusters
----------------

If you are running ``graph-pes-train`` on a compute cluster as a scheduled job, ensure that you:

* use a ``"logged"`` progress bar so that you can monitor the progress of your training run directly from the jobs outputs
* correctly set the ``CUDA_VISIBLE_DEVICES`` environment variable so that ``graph-pes-train`` makes use of all the GPUs you have requested (and no others) (see above)
* consider copying across your data to the worker nodes, and running ``graph-pes-train`` from there rather than on the head node
    - ``graph-pes-train`` writes checkpoints semi-frequently to disk, and this may cause issues/throttle the clusters network.
    - if you are using a disk-backed dataset (for instance reading from an ``<ase>.db`` file), each data point access will require an I/O operation, and reading from local file storage on the worker nodes will be many times faster than over the network.
