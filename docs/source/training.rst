#########
Training
#########

.. Training a graph-based PES model is not trivial. graph-pes attempts to provide useful
.. defaults for training, but since all models are subclasses of :class:`torch.nn.Module`,
.. you are free to roll your own training procedure.

.. An Overview
.. -----------

.. Training a :class:`GraphPESModel <graph_pes.core.GraphPESModel>` involves three preparatory steps:

.. 1. Define and initialize the model (see models)
.. 2. Loading the training data (see loading atomic graphs)
.. 3. Defining the loss function (see loss functions)

.. Within the graph-pes framework, this can be as simple as:

.. .. literalinclude:: ./training_setup.py
..     :language: python
..     :lines: 1-22

.. Once these are defined, by default a two-step training procedure is used in 
.. :meth:`train_model <graph_pes.training.train_model>`:

.. 4. Fit summary statistics to the training data. These are used:
..     * to initialize internal model transformations, such as scale and shifts of local energy estimates
..     * within several available loss functions, e.g. ... such that losses live in a reasonable range, and initial gradients and updates are well behaved
.. 5. Train the model using the loss function and training data

.. This is all handled internally by :meth:`train_model <graph_pes.training.train_model>`, for instance:

.. .. code:
..     :language: python

..     train_model(
..         model,
..         train,
..         val,
..         losses=loss_components,
..         max_epochs=100,
..         accelerator="auto",
..     )

.. Roughly, the following steps are taken:

.. .. literalinclude:: ./training_setup.py
..     :language: python
..     :lines: 24-30


