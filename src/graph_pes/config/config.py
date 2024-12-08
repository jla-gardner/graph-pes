# ruff: noqa: UP006, UP007
# ^^ NB: dacite parsing requires the old type hint syntax in
#        order to be compatible with all versions of Python that
#         we are targeting (3.8+)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import dacite
import data2objects
import yaml
from pytorch_lightning import Callback

from graph_pes.data.datasets import FittingData, GraphDataset
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.training.loss import Loss, TotalLoss, WeightedLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.util import VerboseSWACallback
from graph_pes.utils.misc import nested_merge_all


@dataclass
class FittingOptions:
    """Options for the fitting process."""

    pre_fit_model: bool
    """
    Whether to pre-fit the model before training. See
    :meth:`graph_pes.GraphPESModel.pre_fit_all_components` for details.
    """

    max_n_pre_fit: Union[int, None]
    """
    The maximum number of graphs to use for pre-fitting.
    Set to ``None`` to use all the available training data.
    """

    early_stopping_patience: Union[int, None]
    """
    The number of epochs to wait for improvement in the total validation loss
    before stopping training. Set to ``None`` to disable early stopping.
    """

    loader_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the underlying 
    :class:`torch.utils.data.DataLoader`.

    See the `PyTorch documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ for details.

    Example
    -------
    .. code-block:: yaml

        fitting:    
            loader_kwargs:
                shuffle: true
                seed: 42
                batch_size: 32
                persistent_workers: true
                num_workers: 4
    """  # noqa: E501


@dataclass
class SWAConfig:
    """
    Configuration for Stochastic Weight Averaging.

    Internally, this is handled by `this PyTorch Lightning callback
    <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.StochasticWeightAveraging.html>`__.
    """

    lr: float
    """
    The learning rate to use during the SWA phase. If not specified,
    the learning rate from the end of the training phase will be used.
    """

    start: Union[int, float] = 0.8
    """
    The epoch at which to start SWA. If a float, it will be interpreted
    as a fraction of the total number of epochs.
    """

    anneal_epochs: int = 10
    """
    The number of epochs over which to linearly anneal the learning rate
    to zero.
    """

    strategy: Literal["linear", "cosine"] = "linear"
    """The strategy to use for annealing the learning rate."""

    def instantiate_lightning_callback(self):
        return VerboseSWACallback(
            swa_lrs=self.lr,
            swa_epoch_start=self.start,
            annealing_epochs=self.anneal_epochs,
            annealing_strategy=self.strategy,
        )


@dataclass
class FittingConfig(FittingOptions):
    """Configuration for the fitting process."""

    optimizer: Optimizer
    """
    Specification for the optimizer. Point to something that instantiates a
    :class:`~graph_pes.training.opt.Optimizer`.

    Examples
    --------
    The default (see :func:`~graph_pes.training.opt.Optimizer` for details):

    .. code-block:: yaml

        fitting:        
            optimizer:
                +Optimizer:
                    name: Adam
                    lr: 3e-3
                    weight_decay: 0.0
                    amsgrad: false

    Or a custom one:
    
    .. code-block:: yaml

        fitting:
            optimizer: +my.module.MagicOptimizer()
    """

    scheduler: Union[LRScheduler, None]
    """
    .. _learning rate scheduler:
    
    Specification for the learning rate scheduler. Optional.
    Default is to have no learning rate schedule (``None``).

    Examples
    --------
    .. code-block:: yaml
    
        fitting:
            scheduler:
                +LRScheduler:
                    name: ReduceLROnPlateau
                    factor: 0.5
                    patience: 10
    """

    swa: Union[SWAConfig, None]
    """
    Optional, defaults to ``None``.

    .. dropdown:: ``swa`` options

        .. _stochastic weight averaging:

        .. autoclass:: graph_pes.config.config.SWAConfig()
            :members:
    """

    trainer_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the `PyTorch Lightning Trainer 
    <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`__ .
    
    Example
    -------
    .. code-block:: yaml
        
        fitting:
            trainer:
                max_epochs: 10000
                accelerator: gpu
                accumulate_grad_batches: 4
    """

    callbacks: List[Callback]
    """
    List of PyTorch Lightning :class:`~pytorch_lightning.Callback` or
    :class:`~graph_pes.training.callbacks.GraphPESCallback` instances.
    """


@dataclass
class GeneralConfig:
    """General configuration for a training run."""

    seed: int
    """The global random seed for reproducibility."""

    root_dir: str
    """
    The root directory for this run. 
    
    Results will be stored in ``<root_dir>/<run_id>``, where ``run_id``
    is one of:
    
    * the user-specified ``run_id`` string
    * a random string generated by ``graph-pes``
    """

    run_id: Union[str, None]
    """A unique identifier for this run."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    """The logging level for the logger."""

    progress: Literal["rich", "logged"]
    """The progress bar style to use."""

    torch: TorchConfig
    """
    Configuration for PyTorch.

    .. dropdown:: ``torch`` options

        .. autoclass:: graph_pes.config.config.TorchConfig()
            :members:
    """


@dataclass
class TorchConfig:
    """Configuration for PyTorch."""

    dtype: Literal["float16", "float32", "float64"]
    """
    The dtype to use for all model parameters and graph properties.
    Defaults is ``"float32"``.
    """

    float32_matmul_precision: Literal["highest", "high", "medium"]
    """
    The precision to use internally for float32 matrix multiplications. Refer to the
    `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html>`__
    for details.

    Defaults to ``"high"`` to favour accelerated learning over numerical
    exactness for matmuls.
    """  # noqa: E501


@dataclass
class Config:
    """
    A schema for a configuration file to train a
    :class:`~graph_pes.GraphPESModel`.
    """

    model: Union[GraphPESModel, Dict[str, GraphPESModel]]
    """        
    The model to train.

    .. dropdown:: ``model`` options

        Either point to something that instantiates a
        :class:`~graph_pes.GraphPESModel` object...:

        .. code-block:: yaml

            # basic Lennard-Jones model
            model:
                +LennardJones:
                    sigma: 0.1
                    epsilon: 1.0
            
            # or some custom model
            model: +my_model.SpecialModel()
        
        ...or pass a dictionary mapping custom names to
        :class:`~graph_pes.GraphPESModel` objects:

        .. code-block:: yaml
        
            model:
                offset:
                    +FixedOffset: {H: -123.4, C: -456.7}
                many-body: +SchNet()

        The latter will instantiate an :class:`~graph_pes.models.AdditionModel`
        with :class:`~graph_pes.models.FixedOffset` and
        :class:`~graph_pes.models.SchNet` components. This is a useful approach
        for dealing with arbitrary offset energies.
    """

    data: Union[FittingData, Dict[Literal["train", "valid"], GraphDataset]]
    """
    The data to train on.

    .. dropdown:: ``data`` options
    
        Point to something that can create a 
        :class:`~graph_pes.data.FittingData` instance, or a dictionary mapping 
        ``"train"`` and ``"valid"`` keys to 
        :class:`~graph_pes.data.GraphDataset` instances:

        Load custom data from a function with no arguments:

        .. code-block:: yaml
            
            data: my_module.my_fitting_data()

        Point to :func:`graph_pes.data.load_atoms_dataset` with arguments:

        .. code-block:: yaml

            data:
                +load_atoms_dataset:
                    id: QM9
                    cutoff: 5.0
                    n_train: 10000
                    n_val: 1000
                    property_map:
                        energy: U0

        Point to separate train and validation datasets, taking a random
        1,000 structures from the training file to train from, and all
        structures from the validation file:

        .. code-block:: yaml

            data:
                train:
                    +file_dataset:
                        path: training_data.xyz
                        cutoff: 5.0
                        n: 1000
                        shuffle: true
                        seed: 42
                valid:
                    +file_dataset:
                        path: validation_data.xyz
                        cutoff: 5.0
    """

    loss: Union[Loss, WeightedLoss, TotalLoss, List[Union[WeightedLoss, Loss]]]
    """
    The loss function to use.

    .. dropdown:: ``loss`` options

        This config should either point to something that instantiates a
        :class:`graph_pes.training.loss.Loss` object...

        .. code-block:: yaml
            
            # basic per-atom energy loss
            loss: +PerAtomEnergyLoss()

            # or more fine-grained control
            loss:
                +PropertyLoss:
                    property: energy
                    metric: MAE  # defaults to RMSE if not specified

        ...or specify a list of :class:`~graph_pes.training.loss.WeightedLoss`
        and/or :class:`~graph_pes.training.loss.Loss` instances:

        .. code-block:: yaml
        
            loss:
                # specify a loss with several sub-losses:
                - +PerAtomEnergyLoss()  # defaults to weight 1.0
                - +WeightedLoss:
                    component: +PropertyLoss: { property: forces, metric: MSE }
                    weight: 10.0

        ...or point to your own custom loss implementation, either in isolation:

        .. code-block:: yaml

            loss: 
                +my.module.CustomLoss: { alpha: 0.5 }

        ...or as just another component of a 
        :class:`~graph_pes.training.loss.TotalLoss`:

        .. code-block:: yaml

            loss:
                - +PerAtomEnergyLoss()
                - +my.module.CustomLoss: { alpha: 0.5 }
    """

    fitting: FittingConfig
    """
    Extended configuration for the fitting process.

    .. dropdown:: ``fitting`` options

        .. autoclass:: graph_pes.config.config.FittingConfig()
            :members:
            :inherited-members:
    """

    general: GeneralConfig
    """
    General configuration for a training run.

    .. dropdown:: ``general`` options

        .. autoclass:: graph_pes.config.config.GeneralConfig()
            :members:
    """

    wandb: Union[Dict[str, Any], None]
    """
    Configure Weights and Biases logging.

    .. dropdown:: ``wandb`` options

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
                
    """  # noqa: E501

    misc: Dict[str, Any]
    """
    Miscellaneous configuration - unused by ``graph-pes``, but useful for
    passing through configuration for external tools, or for defining
    constants used in several places in the config:

    .. code-block:: yaml
    
        misc:
            CUTOFF: 5.0

        model:
            +my_module.create_model:
                cutoff: '!~/misc/CUTOFF'
    
        data:
            +my_module.create_data:
                cutoff: '!~/misc/CUTOFF'
    """

    ### Methods ###

    @classmethod
    def from_raw_config_dicts(
        cls, *data_dicts: dict[str, Any]
    ) -> tuple[dict[str, Any], Config]:
        """
        Get the final, merged, reference-replaced config dictionary,
        and corresponding Config instance
        """

        final_dict = nested_merge_all(*data_dicts)
        # special, ugly handling of the fitting/optimizer field
        if final_dict["fitting"]["optimizer"] is None:
            final_dict["fitting"]["optimizer"] = yaml.safe_load(
                """
                +Optimizer:
                    name: Adam
                    lr: 0.001
                """
            )
        final_dict: dict = data2objects.fill_referenced_parts(final_dict)  # type: ignore

        import graph_pes
        import graph_pes.data
        import graph_pes.models
        import graph_pes.training
        import graph_pes.training.callbacks
        import graph_pes.training.loss
        import graph_pes.training.opt

        object_dict = data2objects.from_dict(
            final_dict,
            modules=[
                graph_pes,
                graph_pes.models,
                graph_pes.training,
                graph_pes.training.opt,
                graph_pes.training.loss,
                graph_pes.data,
                graph_pes.training.callbacks,
            ],
        )
        try:
            return final_dict, dacite.from_dict(
                data_class=cls,
                data=object_dict,
                config=dacite.Config(strict=True),
            )
        except Exception as e:
            raise ValueError(
                "Your configuration file could not be successfully parsed. "
                "Please check that it is formatted correctly. For examples, "
                "please see https://jla-gardner.github.io/graph-pes/cli/graph-pes-train.html"
            ) from e

    def get_model(self) -> GraphPESModel:
        if isinstance(self.model, GraphPESModel):
            return self.model
        elif isinstance(self.model, dict):
            if not all(
                isinstance(m, GraphPESModel) for m in self.model.values()
            ):
                raise ValueError(
                    "Expected all values in the model dictionary to be "
                    "GraphPESModel instances."
                )
            return AdditionModel(**self.model)
        raise ValueError(
            "Expected to be able to parse a GraphPESModel or a "
            "dictionary of named GraphPESModels from the model config, "
            f"but got something else: {self.model}"
        )

    def get_data(self) -> FittingData:
        if isinstance(self.data, FittingData):
            return self.data
        elif isinstance(self.data, dict):
            return FittingData(**self.data)

        raise ValueError(
            "Expected to be able to parse a FittingData instance or a "
            "dictionary mapping 'train' and 'valid' keys to GraphDataset "
            "instances from the data config, but got something else: "
            f"{self.data}"
        )

    def get_loss(self) -> TotalLoss:
        if isinstance(self.loss, Loss):
            return TotalLoss([self.loss])
        elif isinstance(self.loss, TotalLoss):
            return self.loss
        elif isinstance(self.loss, list):
            return TotalLoss(self.loss)
        raise ValueError(
            "Expected to be able to parse a Loss, TotalLoss, or a list of "
            "WeightedLoss instances from the loss config, but got something "
            f"else: {self.loss}"
        )
