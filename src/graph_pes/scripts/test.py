from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl

from graph_pes.config.shared import instantiate_config_from_dict
from graph_pes.config.testing import TestingConfig
from graph_pes.models import load_model
from graph_pes.scripts.utils import (
    configure_general_options,
    extract_config_dict_from_command_line,
    update_summary,
)
from graph_pes.training.tasks import test_with_lightning
from graph_pes.utils import distributed
from graph_pes.utils.logger import logger


def test(config: TestingConfig) -> None:
    logger.info(f"Testing model at {config.model_path}...")

    configure_general_options(config.torch, seed=0)

    model = load_model(config.model_path)
    logger.info("Loaded model.")
    logger.debug(f"Model: {model}")

    datasets = config.get_datasets(model)

    for dataset in datasets.values():
        if distributed.IS_RANK_0:
            dataset.prepare_data()
        dataset.setup()

    trainer = pl.Trainer(
        logger=config.get_logger(),
        accelerator=config.accelerator,
        inference_mode=False,
    )

    test_with_lightning(
        trainer,
        model,
        datasets,
        config.loader_kwargs,
        config.prefix,
        user_eval_metrics=[],
    )

    summary_file = Path(config.model_path).parent / "summary.yaml"
    update_summary(trainer.logger, summary_file)


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    config_dict = extract_config_dict_from_command_line(
        "Test a GraphPES model using PyTorch Lightning."
    )
    _, config = instantiate_config_from_dict(config_dict, TestingConfig)
    test(config)


if __name__ == "__main__":
    main()
