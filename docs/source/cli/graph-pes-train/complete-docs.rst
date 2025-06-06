Config options
==============

``graph-pes-train`` is configured using a nested dictionary of options.
The top-level keys that we look for are: ``model``, ``data``, ``loss``, ``fitting``, ``general`` and ``wandb``.

You are free to add any additional top-level keys to your config files for your own purposes. This can be useful for easily referencing constants or repeated values using the ``=`` `reference syntax <https://github.com/jla-gardner/data2objects>`__.

.. code-block:: yaml

    # define a constant...
    CUTOFF: 10.0

    # ... and reference it later
    model:
        +SchNet:
            cutoff: =/CUTOFF

You will also notice the ``+`` syntax used throughout. Under-the-hood, we use the `data2objects <https://github.com/jla-gardner/data2objects>`__ library to parse these config files, and this syntax is used to automatically instantiate objects. 

You can use this syntax to reference arbitrary python functions, classes and objects:

.. code-block:: yaml

    # call your own functions/class constructors 
    # with the ``+`` syntax and key word arguments
    key: 
        +my_module.my_function:  
            foo: 1
            bar: 2

    # syntactic sugar for calling a function
    # with no arguments
    key: +torch.nn.ReLU()

    # reference arbitrary objects
    # (note the lack of any key word arguments or parentheses)
    key: +my_module.my_object


By default, we will look for any objects in the ``graph_pes`` namespace, and hence ``+SchNet`` is shorthand for ``graph_pes.models.SchNet`` etc.

``model``
---------

To specify the model to train, you need to point to something that instantiates a :class:`~graph_pes.GraphPESModel`:

.. code-block:: yaml

    # point to the in-built Lennard-Jones model
    model:
        +LennardJones:
            sigma: 0.1
            epsilon: 1.0
    
    # or point to a custom model
    model: +my_model.SpecialModel()

...or pass a dictionary mapping custom names to :class:`~graph_pes.GraphPESModel` objects:

.. code-block:: yaml

    model:
        offset:
            +FixedOffset: { H: -123.4, C: -456.7 }
        many-body: +SchNet()

The latter approach will be used to instantiate an :class:`~graph_pes.models.AdditionModel`, in this case with :class:`~graph_pes.models.FixedOffset` and
:class:`~graph_pes.models.SchNet` components. This is a useful approach
for dealing with arbitrary offset energies.

You can fine-tune an existing model by pointing ``graph-pes-train`` to an existing model:

.. code-block:: yaml

    model:
        +load_model:
            path: path/to/model.pt

You could also load in parts of a model if e.g. you are fine-tuning on a different level of theory with different offsets:

.. code-block:: yaml

    model:
        offset: +LearnableOffset()
        force_field: 
            +load_model_component:
                path: path/to/model.pt
                key: many-body

See `the fine-tuning guide <https://jla-gardner.github.io/graph-pes/quickstart/quickstart.html#Fine-tuning>`__, 
:func:`~graph_pes.models.load_model`, and :func:`~graph_pes.models.load_model_component` for more details.

``data``
--------

There are various ways to specify the data you wish to use.

The simplest is to point to a dictionary showing where your training, validation and (optionally) test data are located:

.. code-block:: yaml

    data:
        train: data/train.xyz
        valid: data/valid.xyz
        test: data/test.xyz  # results logged to "test/test/<metric_name>"

Note that the ``test`` key can point either to a single dataset as above, or to a dictionary of several named test sets:

.. code-block:: yaml

    data:
        train: data/train.xyz
        valid: data/valid.xyz
        test:
            bulk: data/bulk-test.xyz  # results logged to "test/bulk/<metric_name>"
            slab: data/slab-test.xyz  # results logged to "test/slab/<metric_name>"

Under the hood, these file-paths are passed to the :func:`~graph_pes.data.file_dataset` function, together with the ``cutoff`` of the model you are training.

You can achieve more-fine grained control by instead providing a dictionary of keys to pass to the :func:`~graph_pes.data.file_dataset` function:

.. code-block:: yaml

    data:
        train:
            # take a random sample of 1000 graphs to train on
            path: data/train.xyz
            n: 1000
            shuffle: true
            seed: 42
        valid:
            # use the first 100 graphs in the validation set
            path: data/valid.db
            n: 100
            shuffle: false

.. note::

    The files can be any plain-text file that can be read by :func:`ase.io.read`, e.g. an ``.xyz`` file, or a ``.db`` file containing a SQLite database of :class:`ase.Atoms` objects that is readable as an `ASE database <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`__.

Alternatively, you are able to point to any python function that returns a :class:`~graph_pes.data.GraphDataset` instance:

.. code-block:: yaml

    data:
        train: +my_module.my_training_set()
        valid: 
            +file_dataset:
                path: data/valid.xyz
                cutoff: 5.0

Finally, you can also just point the ``data`` key directly to a :class:`~graph_pes.data.DatasetCollection` instance:

.. code-block:: yaml

    data: +my_module.my_dataset_collection()

This is exactly what the :func:`~graph_pes.data.load_atoms_dataset` function does:

.. code-block:: yaml

    data:
        +load_atoms_dataset:
            id: QM9
            cutoff: 5.0
            n_train: 10000
            n_val: 1000
            property_map:
                energy: U0

``loss``
--------

This config section should either point to something that instantiates a single
:class:`graph_pes.training.loss.Loss` object...

.. code-block:: yaml
    
    # basic per-atom energy loss
    loss: +PerAtomEnergyLoss()

    # or more fine-grained control
    loss:
        +PropertyLoss:
            property: stress
            metric: MAE  # defaults to RMSE if not specified

...or specify a list of :class:`~graph_pes.training.loss.Loss` instances...

.. code-block:: yaml

    loss:
        # specify a loss with several components:
        - +PerAtomEnergyLoss()  # defaults to weight 1.0
        - +PropertyLoss:
            property: forces
            metric: MSE
            weight: 10.0

...or point to your own custom loss implementation, either in isolation:

.. code-block:: yaml

    loss: 
        +my.module.CustomLoss: { alpha: 0.5 }

...or in conjunction with other components:

.. code-block:: yaml

    loss:
        - +PerAtomEnergyLoss()
        - +my.module.CustomLoss: { alpha: 0.5 }


If you want to sweep over a loss component weight via the command line, you can use a
dictionary mapping arbitrary strings to loss instances like so:

.. code-block:: yaml

    loss:
        energy: +PerAtomEnergyLoss()
        forces:
            +ForceRMSE:
                weight: 5.0

allowing you to run a command such as:

.. code-block:: bash

    for weight in 0.1 0.5 1.0; do
        graph-pes-train config.yaml loss/forces/+ForceRMSE/weight=$weight
    done


``fitting``
-----------

The ``fitting`` section of the config is used to specify various hyperparameters and behaviours of the training process.

Optimizer
+++++++++

Configure the optimizer used to train the model, either by providing a dictionary of keyword arguments to the :class:`~graph_pes.training.opt.Optimizer` constructor:

.. code-block:: yaml

    fitting:        
        optimizer:
            # these are the default values
            name: Adam
            lr: 3e-3
            weight_decay: 0.0
            amsgrad: false

or by pointing to something that instantiates a :class:`~graph_pes.training.opt.Optimizer`, for instance using your own code:

.. code-block:: yaml

    fitting:
        optimizer: +my.module.MagicOptimizer()


.. _learning rate scheduler:

Learning rate scheduler
+++++++++++++++++++++++

Configure the learning rate scheduler to use to train the model by specifying a dictionary of keyword arguments to the :class:`~graph_pes.training.opt.LRScheduler` constructor:

.. code-block:: yaml

    fitting:
        scheduler:
            name: ReduceLROnPlateau
            factor: 0.5
            patience: 10

By default, no learning rate scheduler is used if you don't specify one, or if you specify ``null``:

.. code-block:: yaml

    fitting:
        scheduler: null

If you want to use a learning rate warm up, you can do so by specifying the number of training steps over which to warm up the learning rate:

.. code-block:: yaml

    fitting:
        lr_warmup_steps: 1000

This is compatible with specifying any other learning rate scheduler in the ``scheduler`` field: once the warmup is complete, the original scheduler is restored and used.
By default, no warmup is used.

Model pre-fitting
++++++++++++++++++

To turn off :ref:`pre-fitting of the model <pre-fit-model>`, override the ``pre_fit_model`` field (default is ``true``):

.. code-block:: yaml

    fitting:
        pre_fit_model: false

To set the maximum number of graphs to use for :ref:`pre-fitting <pre-fit-model>`, override the ``max_n_pre_fit`` field (default is ``5_000``). These graphs will be randomly sampled from the training data. To use all the training data, set this to ``null``:

.. code-block:: yaml

    fitting:
        max_n_pre_fit: 1000

Reference energies
++++++++++++++++++

If you know the reference energies for your dataset, you can construct an :class:`~graph_pes.models.AdditionModel` with a :class:`~graph_pes.models.FixedOffset` component (see e.g. `here <https://jla-gardner.github.io/graph-pes/cli/graph-pes-train/examples.html#realistic-config>`__).

If, however, you want `graph-pes` to make an educated guess at the mean energy per element in your training data, you can pass the following option:

.. code-block:: yaml

    fitting:
        auto_fit_reference_energies: true

See the :doc:`fine-tuning guide </quickstart/fine-tuning>` for more details.

Early stopping
+++++++++++++++

Turn on early stopping by setting the ``early_stopping`` field to a dictionary with keys corresponding to the :class:`~graph_pes.config.training.EarlyStoppingConfig` class.

.. autoclass:: graph_pes.config.training.EarlyStoppingConfig()
    :members:


The minimal required config for early stopping:

.. code-block:: yaml

    fitting:
        early_stopping: 
            patience: 10


Achieve more fine-grained control:

.. code-block:: yaml

    fitting:
        early_stopping:
            monitor: valid/metrics/forces_rmse  # early stop on forces...
            patience: 10  # ... after 10 checks with no improvement
            min_delta: 0.01  # ... with a minimum change of 0.01 in the rmse

Data loaders
++++++++++++

Data loaders are responsible for sampling batches of data from the dataset. We use :class:`~graph_pes.data.loader.GraphDataLoader` instances to do this. These inherit from the PyTorch :class:`~torch.utils.data.DataLoader` class, and hence you can pass any key word arguments to the underlying loader by setting the ``loader_kwargs`` field:

.. code-block:: yaml

    fitting:
        loader_kwargs:
            seed: 42
            batch_size: 32
            persistent_workers: true
            num_workers: 4

See the `PyTorch documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ for details.

We reccommend using several, persistent workers, since loading data can be a bottleneck, either due to expensive read operations from disk, or due to the time taken to convert the underlying data into :class:`~graph_pes.AtomicGraph` objects (calculating neighbour lists etc.).

Caution: setting the ``shuffle`` field here will have no effect: we always shuffle the training data, and keep the validation and testing data in order.

.. _swa:

Stochastic weight averaging
+++++++++++++++++++++++++++

Configure stochastic weight averaging (SWA) by specifying fields from the :class:`~graph_pes.config.training.SWAConfig` class, e.g.:

.. code-block:: yaml

    fitting:
        swa:
            lr: 1e-3
            start: 0.8
            anneal_epochs: 10


.. autoclass:: graph_pes.config.training.SWAConfig()
    :members:


.. _callbacks:

Callbacks
+++++++++

PyTorch Lightning callbacks are a convenient way to add additional functionality to the training process. 
We implement several useful callbacks in ``graph_pes.training.callbacks`` (e.g. :class:`graph_pes.training.callbacks.OffsetLogger`). Use the ``callbacks`` field to define a list of these, or any other :class:`~pytorch_lightning.callbacks.Callback` objects, that you wish to use:

.. code-block:: yaml

    fitting:
        callbacks:
            - +graph_pes.training.callbacks.OffsetLogger()
            - +my_module.my_callback: { foo: 1, bar: 2 }

PyTorch Lightning Trainer
+++++++++++++++++++++++++

You are free to configure the PyTorch Lightning trainer as you see fit using the ``trainer_kwargs`` field - these keyword arguments will be passed directly to the :class:`~pytorch_lightning.Trainer` constructor. By default, we train for 100 epochs on the best device available (and disable model summaries):

.. code-block:: yaml

    fitting:
        trainer_kwargs:
            max_epochs: 100
            accelerator: auto
            enable_model_summary: false

You can use this functionality to configure any other PyTorch Lightning trainer options, including...

* :ref:`gradient clipping <gradient-clipping>`
* :ref:`validation frequency <validation-frequency>`


.. _gradient-clipping:

Gradient clipping
+++++++++++++++++

Use the ``trainer_kwargs`` field to configure gradient clipping, e.g.:

.. code-block:: yaml

    fitting:
        trainer_kwargs:
            gradient_clip_val: 1.0
            gradient_clip_algorithm: "norm"

.. _validation-frequency:

Validation frequency
++++++++++++++++++++

Use the ``trainer_kwargs`` field to configure validation frequency. For instance, to validate at 10\%, 20\%, 30\% etc. through the training dataset:

.. code-block:: yaml

    fitting:
        trainer_kwargs:
            val_check_interval: 0.1


See the `PyTorch Lightning documentation <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`__ for details.

``wandb``
---------

Disable weights & biases logging:

.. code-block:: yaml
    
        wandb: null

Otherwise, provide a dictionary of
overrides to pass to lightning's `WandbLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`__

.. code-block:: yaml

    wandb:
        project: my_project
        entity: my_entity
        tags: [my_tag]


``general``
-----------

Other miscellaneous configuration options are defined here:

Random seed
+++++++++++

Set the global random seed for reproducibility by setting this to an integer value (by default it is ``42``). This is used to set the random seed for the ``torch``, ``numpy`` and ``random`` modules.

.. code-block:: yaml

    general:
        seed: 42


Output location
+++++++++++++++

The outputs from a training run (model weights, logs etc.) are stored in ``./<root_dir>/<run_id>`` (relative to the current working directory when you run ``graph-pes-train``). By default, we use:

.. code-block:: yaml

    general:
        root_dir: graph-pes-results
        run_id: null  # a random run ID will be generated

You are free to specify any other root directory, and any run ID. If the same run ID is specified for multiple runs, we add numbers to the end of the run ID to make it unique (i.e. ``my_run``, ``my_run_1``, ``my_run_2``, etc.):

.. code-block:: yaml

    general:
        root_dir: my_results
        run_id: my_run


Logging verbosity
+++++++++++++++++

Set the logging verbosity for the training run by setting this to a string value (by default it is ``"INFO"``).

.. code-block:: yaml

    general:
        log_level: DEBUG

Progress bar
++++++++++++

Set the progress bar style to use by setting this to either:

* ``"rich"``: use the `RichProgressBar <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichProgressBar.html>`__ implemented in PyTorch Lightning to display a progress bar. This will not be displayed in any logs.
* ``"logged"``: prints the validation metrics to the console at the end of each validation check.

.. code-block:: yaml

    general:
        progress: logged

Torch options
+++++++++++++

Configure common PyTorch options by setting the ``general.torch`` field to a dictionary of values from the :class:`~graph_pes.config.shared.TorchConfig` class, e.g.:

.. code-block:: yaml

    general:
        torch:
            dtype: float32
            float32_matmul_precision: high

.. autoclass:: graph_pes.config.shared.TorchConfig()
    :members:
