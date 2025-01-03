``graph-pes-id``
================

``graph-pes-id`` is a command line tool for generating a random ID:

.. code-block:: console

    $ graph-pes-id
    brxcu7_p3s018

A common use case for this is to create a series of experiments associated with a single id:

.. code-block:: console

    $ # generate a random id
    $ ID=$(graph-pes-id)

    $ # pre-train a model
    $ graph-pes-train pre-train.yaml \
        general/root_dir=results \
        general/run_id=$ID-pre-train
    ...
    
    $ # fine-tune the model:
    $ # we know where the model weights are and so 
    $ # fine-tuning is easy: we just load the weights
    $ graph-pes-train fine-tune.yaml \
        general/root_dir=results \
        model/+load_model/path=results/$ID-pre-train/model.pt \
        general/run_id=$ID-fine-tune
    ...
