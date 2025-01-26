The basics
==========

.. _pre-fit-model:

Under the hood, the ``graph-pes-train`` command performs the following steps:

1. **loads in your data, model, loss function, etc.** This happens before anything else so that if you run into errors, you can quickly identify the source of the problem.
2. **"pre-fits" the model on the training data.** (optional) Under-the-hood, any :class:`torch.nn.Module` components of your model that define a ``pre_fit`` method will be passed the training data for them to make any adjustments/calculations before training commences (see :meth:`~graph_pes.GraphPESModel.pre_fit_all_components` for details). This is useful for e.g. estimating :class:`energy scales <graph_pes.models.components.scaling.LocalEnergiesScaler>` and :class:`offsets <graph_pes.models.offsets.EnergyOffset>` from the training data. 
3. **trains the model** using a PyTorch Lightning trainer.
4. **saves the best model** for later use, as well as "deploying it" for `use in LAMMPS <../tools/lammps.html>`__
5. **tests the best model** on the training and validation data, together with any other test data you have specified.

To control the behaviour of these steps, you need to pass ``graph-pes-train`` a nested dictionary of configuration options. These options are sourced from three places:

1. the default values defined in `training-defaults.yaml <https://github.com/jla-gardner/graph-pes/blob/main/src/graph_pes/config/training-defaults.yaml>`_
2. values you define in the config file/s you pass to ``graph-pes-train``:  ``<config-1.yaml> <config-2.yaml> ...``
3. additional command line arguments you pass to ``graph-pes-train``: ``<nested/key=value> <nested/key=value> ...``

The following ``.yaml`` config file contains the bare minimum information you need to specify in order to train a model:

.. _minimal config:

.. literalinclude:: ../../../../configs/minimal.yaml
    :caption: ``minimal.yaml``
    :language: yaml

To use this config file, while overriding the default ``CUTOFF`` value, you would run:

.. code-block:: bash

    graph-pes-train minimal.yaml CUTOFF=3.5

.. dropdown:: the order of arguments matters!

    The final nested configuration dictionary used by ``graph-pes-train`` starts as the default values. 
    Reading from left to right, the values you pass to ``graph-pes-train`` will override these defaults.

    Hence above, the ``CUTOFF`` value is set to ``3.5`` in the final configuration dictionary, overriding the value of ``5.0`` specified in ``minimal.yaml``.

    If instead you used: ``graph-pes-train CUTOFF=3.5 minimal.yaml`` then the ``CUTOFF`` value would be set to ``3.5`` in an intermediate step, but would ultimately be overridden by the ``5.0`` in ``minimal.yaml``.

.. dropdown:: a note on syntax

    You may have noticed two special syntaxes in the config file above: ``=/CUTOFF`` and ``+SchNet``\ /\ ``+load_atoms_dataset``.

    Under-the-hood, ``graph-pes-train`` turns the final nested config dictionary into a ``TrainingConfig`` object via a 3 step process:

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

       and hence ``+SchNet`` is shorthand for ``+graph_pes.models.SchNet``.
       Ending the name with ``()`` will call the function/class constructor with no arguments. Pointing the key to a nested dictionary will pass those values as keyword arguments to the constructor. Hence, above, ``+PerAtomEnergyLoss()`` will resolve to a ``~graph_pes.training.loss.PerAtomEnergyLoss`` object, while the :class:`~graph_pes.models.SchNet` model will be constructed with the keyword arguments specified in the config.
    
    3. the resulting dictionary of python objects is then converted, using `dacite <https://github.com/konradhalas/dacite/tree/master/>`__, into a final nested ``TrainingConfig`` object.