"""
Train a model from a configuration file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from graph_pes.config import Config
from graph_pes.deploy import deploy_model
from graph_pes.logger import logger
from graph_pes.training.ptl import train_with_lightning
from graph_pes.util import nested_merge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # load default and user data
    with open(Path(__file__).parent.parent / "configs/defaults.yaml") as f:
        defaults: dict[str, Any] = yaml.safe_load(f)
    with open(args.config) as f:
        user_config: dict[str, Any] = yaml.safe_load(f)

    # get the config object
    config_dict = nested_merge(defaults, user_config)
    config = Config.from_dict(config_dict)

    # TODO: command line overrides

    logger.info(config)

    model = config.instantiate_model()

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

    logger.info("Training complete: deploying model for use with LAMMPS")
    deploy_model(model, cutoff=5.0, path="model.pt")


if __name__ == "__main__":
    main()
