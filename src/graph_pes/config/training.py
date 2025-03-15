# ruff: noqa: UP006, UP007
# ^^ NB: dacite parsing requires the old type hint syntax in
#        order to be compatible with all versions of Python that
#        we are targeting (3.9+)
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import yaml
from pytorch_lightning import Callback

from graph_pes.config.shared import TorchConfig
from graph_pes.data.datasets import DatasetCollection
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.callbacks import VerboseSWACallback
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer


@dataclass
class EarlyStoppingConfig:
    patience: int
    """
    The number of validation checks with no improvement before stopping.
    """

    min_delta: float = 0.0
    """
    The minimum change in the monitored quantity to qualify as an improvement.
    """

    monitor: str = "valid/loss/total"
    """The quantity to monitor."""


@dataclass
class FittingOptions:
    """Options for the fitting process."""

    pre_fit_model: bool
    max_n_pre_fit: Union[int, None]
    early_stopping: Union[EarlyStoppingConfig, None]
    loader_kwargs: Dict[str, Any]
    early_stopping_patience: Union[int, None]
    """
    DEPRECATED: use the `early_stopping` config option instead.
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

    strategy: Literal["linear", "cos"] = "linear"
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

    trainer_kwargs: Dict[str, Any]
    optimizer: Optimizer = Optimizer(name="AdamW", lr=1e-3, amsgrad=False)
    scheduler: Union[LRScheduler, None] = None
    swa: Union[SWAConfig, None] = None
    callbacks: List[Callback] = field(default_factory=list)


@dataclass
class GeneralConfig:
    """General configuration for a training run."""

    seed: int
    root_dir: str
    run_id: Union[str, None]
    torch: TorchConfig
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    progress: Literal["rich", "logged"] = "rich"


# TODO:
# - move get_model to utils, call it parse_model
# - turn loss into dict with human readable names + move parse_loss to utils


@dataclass
class TrainingConfig:
    """
    A schema for a configuration file to train a
    :class:`~graph_pes.GraphPESModel`.
    """

    model: Union[GraphPESModel, Dict[str, GraphPESModel]]
    data: DatasetCollection
    loss: Union[Loss, TotalLoss, Dict[str, Loss], List[Loss]]
    fitting: FittingConfig
    general: GeneralConfig
    wandb: Union[Dict[str, Any], None]

    ### Methods ###

    def get_data(self) -> DatasetCollection:
        if isinstance(self.data, DatasetCollection):
            return self.data
        elif isinstance(self.data, dict):
            return DatasetCollection(**self.data)

        raise ValueError(
            "Expected to be able to parse a DatasetCollection instance or a "
            "dictionary mapping 'train' and 'valid' keys to GraphDataset "
            "instances from the data config, but got something else: "
            f"{self.data}"
        )

    @classmethod
    def defaults(cls) -> dict:
        with open(Path(__file__).parent / "training-defaults.yaml") as f:
            return yaml.safe_load(f)
