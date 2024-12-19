from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pytorch_lightning.loggers import CSVLogger, Logger

from graph_pes.data import GraphDataset
from graph_pes.scripts.test import DEFAULT_LOADER_KWARGS
from graph_pes.training.callbacks import WandbLogger


@dataclass
class TestingConfig:
    model_path: str
    """The path to the ``model.pt`` file."""

    data: Union[dict[str, GraphDataset], GraphDataset]  # noqa: UP007
    """
    A mapping from names to datasets.

    Results will be logged as ``"test/<name>/<metric>"``. This allows
    for testing on multiple datasets at once.
    """

    loader_kwargs: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_LOADER_KWARGS
    )
    """Keyword arguments for the data loader."""

    logger: Union[Literal["auto", "csv"], dict[str, Any]] = "auto"  # noqa: UP007
    """
    The logger to use for logging the test metrics.

    If ``"auto"``, we will attempt to find the training config
    from ``<model_path>/../train-config.yaml``, and use the logger
    from that config.

    If ``"csv"``, we will use a CSVLogger.

    If a dictionary, we will instantiate a new :class:`WandbLogger`
    with the provided arguments.
    """

    accelerator: str = "auto"
    """The accelerator to use for testing."""

    def get_logger(self) -> Logger:
        root_dir = Path(self.model_path).parent
        if self.logger == "csv":
            return CSVLogger(save_dir=root_dir, name="")
        elif isinstance(self.logger, dict):
            return WandbLogger(output_dir=root_dir, **self.logger)

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

        return WandbLogger(output_dir=root_dir, **logger_data)
