# ruff: noqa: UP006, UP007
# ^^ NB: dacite parsing requires the old type hint syntax in
#        order to be compatible with all versions of Python that
#         we are targeting (3.8+)
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Literal, Union

import dacite
import yaml
from pytorch_lightning import Callback

from graph_pes.data.datasets import FittingData
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.util import VerboseSWACallback

from .utils import create_from_data, create_from_string


@dataclass
class LossSpec:
    """
    Specification for a component of a
    :class:`~graph_pes.training.loss.TotalLoss`.
    """

    component: Union[str, Dict[str, Any]]
    """Point to a :class:`~graph_pes.training.loss.Loss` instance."""

    weight: Union[int, float] = 1.0
    """The weight of this loss component."""


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

    optimizer: Union[str, Dict[str, Any]]
    """
    Specification for the optimizer. Point to something that instantiates a
    :class:`~graph_pes.training.opt.Optimizer`.

    Examples
    --------
    The default (see :func:`~graph_pes.training.opt.Optimizer` for details):

    .. code-block:: yaml

        fitting:        
            optimizer:
                graph_pes.training.opt.Optimizer:
                    name: Adam
                    lr: 3e-3
                    weight_decay: 0.0
                    amsgrad: false

    Or a custom one:
    
    .. code-block:: yaml

        fitting:
            optimizer: my.module.MagicOptimizer()
    """

    scheduler: Union[str, Dict[str, Any], None]
    """
    .. _learning rate scheduler:
    
    Specification for the learning rate scheduler. Optional.
    Default is to have no learning rate schedule (``None``).

    Examples
    --------
    .. code-block:: yaml
    
        fitting:
            scheduler:
                graph_pes.training.opt.LRScheduler:
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

    callbacks: List[Dict[str, Any]]
    """
    List of dictionaries, each of which points to a 
    :class:`~pytorch_lightning.Callback` or 
    :class:`~graph_pes.training.callbacks.GraphPESCallback` instance.
    """

    ### Methods ###

    def instantiate_optimizer(self) -> Optimizer:
        return create_from_data(self.optimizer)

    def instantiate_scheduler(self) -> LRScheduler | None:
        if self.scheduler is None:
            return None
        return create_from_data(self.scheduler)

    def instantiate_callbacks(self) -> List[Callback]:
        return [create_from_data(c) for c in self.callbacks]


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

    model: Union[str, Dict[str, Any]]
    """        
    The model to train.

    .. dropdown:: ``model`` options

        Either point to something that instantiates a
        :class:`~graph_pes.GraphPESModel`...:

        .. code-block:: yaml

            # basic Lennard-Jones model
            model:
                graph_pes.models.LennardJones:
                    sigma: 0.1
                    epsilon: 1.0
            
            # or some custom model
            model: my_model.SpecialModel()
        
        ...or pass a dictionary mapping custom names to
        :class:`~graph_pes.GraphPESModel` instances:

        .. code-block:: yaml
        
            model:
                offset:
                    graph_pes.models.FixedOffset: {H: -123.4, C: -456.7}
                many-body: graph_pes.models.SchNet()

        The latter will instantiate an :class:`~graph_pes.models.AdditionModel`
        with :class:`~graph_pes.models.FixedOffset` and
        :class:`~graph_pes.models.SchNet` components. This is a useful approach
        for dealing with arbitrary offset energies.
    """

    data: Union[str, Dict[str, Any]]
    """
    The data to train on.

    .. dropdown:: ``data`` options
    
        Point to one of the following:

        - a callable that returns a :class:`~graph_pes.data.FittingData` 
          instance
        - a dictionary mapping ``"train"`` and ``"valid"`` keys to callables 
          that return :class:`~graph_pes.data.GraphDataset` instances

        Load custom data from a function with no arguments:

        .. code-block:: yaml
            
            data: my_module.my_fitting_data()

        Point to :func:`graph_pes.data.load_atoms_dataset` with arguments:

        .. code-block:: yaml

            data:
                graph_pes.data.load_atoms_dataset:
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
                    graph_pes.data.file_dataset:
                        path: training_data.xyz
                        cutoff: 5.0
                        n: 1000
                        shuffle: true
                        seed: 42
                valid:
                    graph_pes.data.file_dataset:
                        path: validation_data.xyz
                        cutoff: 5.0
    """

    loss: Union[str, Dict[str, Any], List[LossSpec]]
    """
    The loss function to use.

    .. dropdown:: ``loss`` options

        This config should either point to something that instantiates a
        :class:`~graph_pes.training.loss.Loss`...

        .. code-block:: yaml
            
            # basic per-atom energy loss
            loss: graph_pes.training.loss.PerAtomEnergyLoss()

            # or more fine-grained control
            loss:
                graph_pes.training.loss.Loss:
                    property: energy
                    metric: graph_pes.training.loss.RMSE()

        ...or specify a list of :class:`~graph_pes.config.config.LossSpec`
        instances, each of which points to a 
        :class:`~graph_pes.training.loss.Loss` instance and specifies a weight:

        .. code-block:: yaml
        
            loss:
                - component: graph_pes.training.loss.Loss:
                      property: energy
                      metric: graph_pes.training.loss.RMSE()
                  weight: 1.0
                - component: graph_pes.training.loss.Loss:
                      property: forces
                      metric: graph_pes.training.loss.MAE()
                weight: 10.0
        
        .. autoclass:: graph_pes.config.config.LossSpec()
            :members:
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
    passing through configuration for external tools.
    """

    ### Methods ###

    def to_nested_dict(self) -> Dict[str, Any]:
        def _maybe_as_dict(obj):
            if isinstance(obj, list):
                return [_maybe_as_dict(v) for v in obj]
            elif not hasattr(obj, "__dict__"):
                return obj
            return {k: _maybe_as_dict(v) for k, v in obj.__dict__.items()}

        return _maybe_as_dict(self)  # type: ignore

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        try:
            return dacite.from_dict(
                data_class=cls,
                data=data,
                config=dacite.Config(strict=True),
            )
        except Exception as e:
            raise ValueError(
                "Your configuration file could not be successfully parsed. "
                "Please check that it is formatted correctly. For examples, "
                "please see https://jla-gardner.github.io/graph-pes/cli/graph-pes-train.html"
            ) from e

    def hash(self) -> str:
        """
        Get a unique identifier for this configuration.

        Returns
        -------
        str
            The SHA-256 hash of the configuration.
        """
        return sha256(str(self.to_nested_dict()).encode()).hexdigest()

    def __repr__(self):
        # yaml dump the nested config, complete with defaults,
        # as it would appear in a config.yaml file
        return yaml.dump(self.to_nested_dict(), indent=3, sort_keys=False)

    def instantiate_model(self) -> GraphPESModel:
        obj = create_from_data(self.model)
        if isinstance(obj, GraphPESModel):
            return obj

        elif isinstance(obj, dict):
            all_string_keys = all(isinstance(k, str) for k in obj)
            all_model_values = all(
                isinstance(v, GraphPESModel) for v in obj.values()
            )

            if not all_string_keys or not all_model_values:
                raise ValueError(
                    "Expected a dictionary of named GraphPESModels, "
                    f"but got {obj}."
                )

            try:
                return AdditionModel(**obj)
            except Exception as e:
                raise ValueError(
                    f"Parsed a dictionary, {obj}, from the model config, "
                    "but could not instantiate an AdditionModel from it."
                ) from e

        raise ValueError(
            "Expected to be able to parse a GraphPESModel or a "
            "dictionary of named GraphPESModels from the model config, "
            f"but got something else: {obj}"
        )

    def instantiate_data(self) -> FittingData:
        result: Any = None

        if isinstance(self.data, str):
            result = create_from_string(self.data)

        if isinstance(self.data, dict):
            if len(self.data) == 1:
                result = create_from_data(self.data)
            elif len(self.data) == 2:
                assert self.data.keys() == {"train", "valid"}
                result = FittingData(
                    train=create_from_data(self.data["train"]),
                    valid=create_from_data(self.data["valid"]),
                )

        if result is None:
            raise ValueError(
                "Unexpected data specification. "
                "Please provide a callable or a dictionary containing "
                "a single key (the fully qualified name of some callable) "
                "or two keys ('train' and 'valid') mapping to callables."
            )

        if not isinstance(result, FittingData):
            raise ValueError(
                "Expected to parse a FittingData instance from the data "
                f"config, but got {result}."
            )

        return result

    def instantiate_loss(self) -> TotalLoss:
        if isinstance(self.loss, (str, dict)):
            loss = create_from_data(self.loss)
            if isinstance(loss, Loss):
                return TotalLoss([loss])
            elif isinstance(loss, TotalLoss):
                return loss
            else:
                raise ValueError(
                    "Expected to parse a Loss or TotalLoss instance from the "
                    "loss config, but got something else: {loss}"
                )

        if not all(isinstance(l, LossSpec) for l in self.loss):
            raise ValueError(
                "Expected a list of LossSpec instances from the loss config, "
                f"but got {self.loss}."
            )

        weights = [l.weight for l in self.loss]
        losses = [create_from_data(l.component) for l in self.loss]

        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError(
                "Expected a list of weights from the loss config, "
                f"but got {weights}."
            )

        if not all(isinstance(l, Loss) for l in losses):
            raise ValueError(
                "Expected a list of Loss instances from the loss config, "
                f"but got {losses}."
            )

        return TotalLoss(losses, weights)
