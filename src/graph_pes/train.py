"""
Train a model from a configuration file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from graph_pes.util import nested_merge

from .config import Config
from .deploy import LAMMPSModel
from .logger import logger
from .training.ptl import train_with_lightning


def main(config: Config):
    logger.info(config)

    model = config.instantiate_model()
    logger.info(model)

    data = config.instantiate_data()
    logger.info(data)

    optimizer = config.fitting.instantiate_optimizer()
    logger.info(optimizer)

    scheduler = config.fitting.instantiate_scheduler()
    logger.info(scheduler)

    total_loss = config.instantiate_loss()
    logger.info(total_loss)

    train_with_lightning(
        model,
        data,
        loss=total_loss,
        fit_config=config.fitting,
        optimizer=optimizer,
        scheduler=scheduler,
        config_to_log=config.to_nested_dict(),
    )

    logger.info("Training complete.")

    lammps_model = LAMMPSModel(model)
    scripted_model = torch.jit.script(lammps_model)
    logger.info("Saving model to model.pt")
    torch.jit.save(scripted_model, "model.pt")


if __name__ == "__main__":
    # get config file from --config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # load default and user data
    with open(Path(__file__).parent / "configs/defaults.yaml") as f:
        defaults: dict[str, Any] = yaml.safe_load(f)
    with open(args.config) as f:
        user_config: dict[str, Any] = yaml.safe_load(f)

    # get the config object
    config_dict = nested_merge(defaults, user_config)
    config = Config.from_dict(config_dict)

    # TODO: command line overrides

    # go!
    main(config)
