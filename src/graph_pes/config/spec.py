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

from graph_pes.data.datasets import FittingData
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.addition import AdditionModel
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.ptl_utils import VerboseSWACallback

from .utils import create_from_data, create_from_string


@dataclass
class LossSpec:
    component: Union[str, Dict[str, Any]]
    weight: Union[int, float] = 1.0


@dataclass
class FittingOptions:
    pre_fit_model: bool
    """Whether to pre-fit the model before training."""

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

    trainer_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the PTL trainer.
    
    See their docs. # TODO
    
    Example
    -------
    .. code-block:: yaml
    
        trainer:
            max_epochs: 100
            gpus: 1
            check_val_every_n_epoch: 5
    """

    loader_kwargs: Dict[str, Any]
    """
    Key-word arguments to pass to the underlying 
    :class:`torch.utils.data.DataLoader`.

    See their docs. # TODO
    """


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
    """Configuration for the fitting process:"""

    optimizer: Union[str, Dict[str, Any]]
    """
    Specification for the optimizer.

    ``graph-pes`` provides a few common optimisers, but you can also 
    roll your own.
    
    Point to something that instantiates a graph_pes optimiser:

    Examples
    --------
    The default (see :func:`~graph_pes.training.opt.Optimizer` for details):

    .. code-block:: yaml
    
        optimizer:
            graph_pes.training.opt.Optimizer:
                name: Adam
                lr: 3e-3
                weight_decay: 0.0
                amsgrad: false

    Or a custom one:
    
    .. code-block:: yaml
    
        optimizer: my.module.MagicOptimizer()
    """

    scheduler: Union[str, Dict[str, Any], None]
    """
    Specification for the learning rate scheduler. Optional.

    TODO: more schedules/flexibility

    Examples
    --------
    .. code-block:: yaml
    
        scheduler:
            graph_pes.training.opt.LRScheduler:
                name: ReduceLROnPlateau
                factor: 0.5
                patience: 10
    """

    swa: Union[SWAConfig, None]
    """Configuration for Stochastic Weight Averaging. Optional."""

    ### Methods ###

    def instantiate_optimizer(self) -> Optimizer:
        return create_from_data(self.optimizer)

    def instantiate_scheduler(self) -> LRScheduler | None:
        if self.scheduler is None:
            return None
        return create_from_data(self.scheduler)


@dataclass
class GeneralConfig:
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


@dataclass
class Config:
    """
    A schema for a configuration file to train a
    :class:`~graph_pes.core.GraphPESModel`.

    While parsing your configuration file, we will attempt to import
    any class, object or function that you specify via a fully qualified
    name. This allows you to point both to classes and functions that
    ``graph-pes`` provides, as well as your own custom code.

    .. note::
        To point to an object, simplify specify the **fully qualified name**,
        e.g. ``my_module.my_object``.

        If you want to use the return value of a
        function with no arguments, append ``()`` to the name, e.g.
        ``my_module.my_function()``.

        To point to a class or function with arguments, use a nested dictionary
        structure like so:

    .. code-block:: yaml

        graph_pes.models.SchNet:
            cutoff: 5.0
            n_layers: 3
    """

    model: Union[str, Dict[str, Any]]
    """
    Specification for the model.

    Examples
    --------
    To specify a single model with parameters:

    .. code-block:: yaml
    
        model:
            graph_pes.models.LennardJones:
                sigma: 0.1
                epsilon: 1.0
    
    or, if no parameters are needed:

    .. code-block:: yaml
    
        model: my_model.SpecialModel()
    
    To specify multiple components of an 
    :class:`~graph_pes.models.AdditionModel`, create a list of specications as 
    above, using whatever unique names you please:

    .. code-block:: yaml
    
        model:
            offset:
                graph_pes.models.FixedOffset: {H: -123.4, C: -456.7}
            many-body: graph_pes.models.SchNet()
    """

    data: Union[str, Dict[str, Any]]
    """
    Specification for the data. 
    
    Point to one of the following:

    - a callable that returns a :class:`~graph_pes.data.dataset.FittingData` 
      instance
    - a dictionary mapping ``"train"`` and ``"valid"`` keys to callables that
      return :class:`~graph_pes.data.dataset.LabelledGraphDataset` instances

    Examples
    --------
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
    Specification for the loss function. This can be a single loss function
    or a list of loss functions with weights.

    Examples
    --------
    To specify a single loss function:

    .. code-block:: yaml
    
        loss: graph_pes.training.loss.PerAtomEnergyLoss()

    or with parameters:

    .. code-block:: yaml
        
            loss:
                graph_pes.training.loss.Loss:
                    property_key: energy
                    metric: graph_pes.training.loss.RMSE()

    To specify multiple loss functions with weights:

    .. code-block:: yaml
    
        loss:
            - component: graph_pes.training.loss.Loss:
                property_key: energy
                metric: graph_pes.training.loss.RMSE()
              weight: 1.0
            - component: graph_pes.training.loss.Loss:
                property_key: forces
                metric: graph_pes.training.loss.MAE()
              weight: 10.0
    """

    fitting: FittingConfig
    """see :class:`~graph_pes.config.spec.FittingConfig`"""

    general: GeneralConfig
    """Miscellaneous configuration options."""

    wandb: Union[Dict[str, Any], None]
    """
    Configuration for Weights & Biases logging.
    
    If ``None``, logging is disabled. Otherwise, provide a dictionary of
    overrides to pass to lightning's `WandbLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`__.

    Examples
    --------
    Custom project, entity and tags:

    .. code-block:: yaml
    
        wandb:
            project: my_project
            entity: my_entity
            tags: [my_tag]

    Disable weights & biases logging:

    .. code-block:: yaml
        
            wandb: null
    """  # noqa: E501

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
                "please see ..."  # TODO
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
        # TODO: simplify
        if isinstance(self.loss, (str, dict)):
            loss = create_from_data(self.loss)
            if isinstance(loss, Loss):
                return TotalLoss([loss])
            elif isinstance(loss, TotalLoss):
                return loss
            else:
                raise ValueError("# TODO")

        else:
            if not all(isinstance(l, LossSpec) for l in self.loss):
                raise ValueError("# TODO")

            weights = [l.weight for l in self.loss]
            losses = [create_from_data(l.component) for l in self.loss]

            if not all(isinstance(w, (int, float)) for w in weights):
                raise ValueError("# TODO")

            if not all(isinstance(l, Loss) for l in losses):
                raise ValueError("# TODO")

            return TotalLoss(losses, weights)
