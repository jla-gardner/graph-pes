Train a model
=============

``graph-pes-train`` provides a unified interface to train any :class:`~graph_pes.core.GraphPESModel`, including those packaged within ``graph_pes.models`` [todo link] and those defined by you, the user [TODO: link].

.. seealso::

    For more information on the ``graph-pes-train`` command, and the plethora of options available for specification in your ``config.yaml`` see the :ref:`CLI reference <cli-reference>`.

.. tip::

    Pre-requisites:

    1. Ensure you have installed graph-pes. TODO: link
    2. Check that you have ``graph-pes-train`` available in your shell by running ``graph-pes-train -h`` (might need to restart)
    3. Copy the contents of ``train.yaml`` below into your current directory (e.g. by ...) TODO

To get started, or to test your ``graph-pes`` installation, run the following:

.. code-block:: bash

    graph-pes-train --config train.yaml

where the contents of ``train.yaml`` specifies:

* the model architecture you want to instantiate (here :class:`~graph_pes.models.PaiNN`) 
* the data you want to train on (here the `QM7 <https://jla-gardner.github.io/load-atoms/datasets/QM7.html>`_ dataset downloaded internally using `load-atoms <https://jla-gardner.github.io/load-atoms/>`_) 
* the loss function you want to use (here a simple :class:`~graph_pes.training.loss.PerAtomEnergyLoss`) 
* ... and any other training hyperparameters you care about:

.. literalinclude:: quickstart-config.yaml
    :language: yaml
    :caption: train.yaml


Outputs
-------

When this script first runs, you may well be prompted to login to `Weights and Biases <https://wandb.ai/site>`_ (WandB) to log your training runs. This is optional, but recommended for tracking your experiments.



