from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Union

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import CSVLogger, Logger

from graph_pes.config.utils import instantiate_config_from_dict
from graph_pes.data import GraphDataset
from graph_pes.data.loader import GraphDataLoader
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models import load_model
from graph_pes.scripts.utils import extract_config_dict_from_command_line
from graph_pes.training.task import PESLearningTask
from graph_pes.training.trainer import WandbLogger
from graph_pes.utils import distributed
from graph_pes.utils.logger import logger

DEFAULT_LOADER_KWARGS: Final[dict] = dict(batch_size=2, num_workers=0)


@dataclass
class TestConfig:
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


def test(config: TestConfig) -> None:
    logger.info(f"Testing model at {config.model_path}...")

    model = load_model(config.model_path)
    logger.info("Loaded model.")
    logger.debug(f"Model: {model}")

    loader_kwargs = {
        **DEFAULT_LOADER_KWARGS,
        **config.loader_kwargs,
        "shuffle": False,
    }
    datasets = (
        config.data if isinstance(config.data, dict) else {"test": config.data}
    )

    for dataset in datasets.values():
        if distributed.IS_RANK_0:
            dataset.prepare_data()
        dataset.setup()

    dataloaders = {
        name: GraphDataLoader(dataset, **loader_kwargs)
        for name, dataset in datasets.items()
    }
    trainer = pl.Trainer(
        logger=config.get_logger(),
        accelerator=config.accelerator,
        inference_mode=False,
    )
    test_model(model, dataloaders, trainer)


def test_model(
    model: GraphPESModel,
    data: dict[str, GraphDataLoader],
    trainer: pl.Trainer,
) -> None:
    assert trainer.test_loop.inference_mode is False
    testing_task = PESLearningTask.for_testing(
        model, test_names=list(data.keys())
    )
    trainer.test(testing_task, list(data.values()))


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    config_dict = extract_config_dict_from_command_line(
        "Test a GraphPES model using PyTorch Lightning."
    )
    try:
        _, config = instantiate_config_from_dict(config_dict, TestConfig)
    except Exception as e:
        raise ValueError(
            "Your configuration file could not be successfully parsed. "
            "Please check that it is formatted correctly. For examples, "
            "please see https://jla-gardner.github.io/graph-pes/cli/graph-pes-test.html"
        ) from e
    test(config)


if __name__ == "__main__":
    main()
