from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pytorch_lightning.loggers import CSVLogger, Logger

from graph_pes.config.shared import TorchConfig
from graph_pes.data import GraphDataset
from graph_pes.training.callbacks import WandbLogger


@dataclass
class TestingConfig:
    """Configuration for testing a GraphPES model."""

    model_path: str
    """The path to the ``model.pt`` file."""

    data: Union[GraphDataset, dict[str, GraphDataset]]  # noqa: UP007
    """
    Either:

    - a single :class:`~graph_pes.data.GraphDataset`. Results will be logged as
      ``"<prefix>/<metric>"``.
    - a mapping from names to datasets. Results will be logged as
      ``"<prefix>/<dataset-name>/<metric>"``, allowing for testing on multiple 
      datasets.
    """

    loader_kwargs: dict[str, Any]
    """
    Keyword arguments to pass to the 
    :class:`~graph_pes.data.loader.GraphDataLoader`.

    Defaults to:

    .. code-block:: yaml

        loader_kwargs:
            batch_size: 2
            num_workers: 0
    
    You should tune this to make testing faster.
    """

    torch: TorchConfig
    """The torch configuration to use for testing."""

    logger: Union[Literal["auto", "csv"], dict[str, Any]] = "auto"  # noqa: UP007
    """
    The logger to use for logging the test metrics.

    If ``"auto"``, we will attempt to find the training config
    from ``<model_path>/../train-config.yaml``, and use the logger
    from that config.

    If ``"csv"``, we will use a CSVLogger.

    If a dictionary, we will instantiate a new 
    :class:`~graph_pes.training.callbacks.WandbLogger` with the provided 
    arguments.
    """

    accelerator: str = "auto"
    """The accelerator to use for testing."""

    prefix: str = "testing"
    """The prefix to use for logging. Individual metrics will be logged as
    ``<prefix>/<dataset_name>/<metric>``.
    """

    def get_logger(self) -> Logger:
        root_dir = Path(self.model_path).parent
        if self.logger == "csv":
            return CSVLogger(save_dir=root_dir, name="")
        elif isinstance(self.logger, dict):
            return WandbLogger(
                output_dir=root_dir,
                log_epoch=False,
                **self.logger,
            )

        if not self.logger == "auto":
            raise ValueError(f"Invalid logger: {self.logger}")

        train_config_path = root_dir / "train-config.yaml"
        if not train_config_path.exists():
            raise ValueError(
                f"Could not find training config at {train_config_path}. "
                "Please specify a logger explicitly."
            )
        with open(train_config_path) as f:
            logger_data = yaml.safe_load(f).get("wandb", None)

        if logger_data is None:
            return CSVLogger(save_dir=root_dir, name="")

        return WandbLogger(output_dir=root_dir, log_epoch=False, **logger_data)

    @classmethod
    def defaults(cls) -> dict:
        return {
            "torch": {"float32_matmul_precision": "high", "dtype": "float32"},
            "loader_kwargs": {"batch_size": 2, "num_workers": 0},
        }
