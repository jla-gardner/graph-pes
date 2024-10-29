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

    $ # train a pre-training model
    $ graph-pes-train pre-train.yaml \
        general^root_dir=results \
        general^run_id=$ID-pre-train
    ...
    
    $ # we know where the model weights are - 
    $ # fine-tuning is easy: we just load them in
    $ graph-pes-train fine-tune.yaml \
        general^root_dir=results \
        model^graph_pes.models.load_model^path=results/$ID-pre-train/model.pt \
        general^run_id=$ID-fine-tune
    ...
