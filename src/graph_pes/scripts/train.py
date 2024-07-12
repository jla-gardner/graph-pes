from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Any

import yaml
from graph_pes.config import Config
from graph_pes.deploy import deploy_model
from graph_pes.logger import logger
from graph_pes.training.ptl import train_with_lightning
from graph_pes.util import nested_merge


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a GraphPES model from a configuration file "
            "using PyTorch Lightning."
        ),
        epilog=(
            "Example usage: graph-pes-train --config config1.yaml --config "
            "config2.yaml fitting^loader_kwargs^batch_size=32 "
        ),
    )

    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help=(
            "Path to the configuration file. "
            "This argument can be used multiple times, with later files "
            "taking precedence over earlier ones in the case of conflicts."
        ),
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Config overrides in the form nested^key=value, "
            "separated by spaces, e.g. fitting^loader_kwargs^batch_size=32. "
        ),
    )

    return parser.parse_args()


def extract_config_from_command_line():
    args = parse_args()

    # load default config
    with open(Path(__file__).parent.parent / "configs/defaults.yaml") as f:
        defaults: dict[str, Any] = yaml.safe_load(f)

    # load user configs
    user_configs: list[dict[str, Any]] = []
    for config_path in args.config:
        with open(config_path) as f:
            user_configs.append(yaml.safe_load(f))

    # get the config object
    final_config_dict = defaults
    for user_config in user_configs:
        final_config_dict = nested_merge(final_config_dict, user_config)

    # apply overrides
    for override in args.overrides:
        if override.count("=") != 1:
            raise ValueError(
                f"Invalid override: {override}. "
                "Expected something of the form key=value"
            )
        key, value = override.split("=")
        keys = key.split("^")

        # parse the value
        with contextlib.suppress(yaml.YAMLError):
            value = yaml.safe_load(value)

        current = final_config_dict
        for k in keys[:-1]:
            current.setdefault(k, {})
            current = current[k]
        current[keys[-1]] = value

    return Config.from_dict(final_config_dict)


def train_from_config(config: Config):
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

    logger.info(
        "Training complete: deploying model for use with LAMMPS to "
        "./model.pt"
    )
    deploy_model(model, cutoff=5.0, path="model.pt")


def main():
    config = extract_config_from_command_line()
    train_from_config(config)


if __name__ == "__main__":
    main()
