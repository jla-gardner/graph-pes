from __future__ import annotations

import argparse
import contextlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning
import torch
import wandb
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.deploy import deploy_model
from graph_pes.logger import log_to_file, logger
from graph_pes.training.ptl import create_trainer, train_with_lightning
from graph_pes.util import nested_merge, random_id
from pytorch_lightning.loggers import CSVLogger, WandbLogger

warnings.filterwarnings(
    "ignore", message=".*There is a wandb run already in progress.*"
)


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


def extract_config_from_command_line() -> Config:
    args = parse_args()

    # load default config
    defaults = get_default_config_values()

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
    pytorch_lightning.seed_everything(config.general.seed)

    # time to the millisecond
    now = datetime.now().strftime("%F %T.%f")[:-3]
    logger.info(f"Started training at {now}")

    logger.info(config)

    model = config.instantiate_model()  # gets logged later

    data = config.instantiate_data()
    logger.info(data)

    optimizer = config.fitting.instantiate_optimizer()
    logger.info(optimizer)

    scheduler = config.fitting.instantiate_scheduler()
    logger.info(scheduler if scheduler is not None else "No LR scheduler.")

    total_loss = config.instantiate_loss()
    logger.info(total_loss)

    # set-up the trainer
    if config.wandb is not None:
        wandb.init(**config.wandb)
        assert wandb.run is not None
        run_id = wandb.run.id
    else:
        run_id = random_id()

    try:
        output_dir = Path(config.general.root_dir) / run_id
        logger.info(f"Output directory: {output_dir}")

        log_to_file(output_dir / "log.txt")

        if config.wandb is not None:
            lightning_logger = WandbLogger()
        else:
            lightning_logger = CSVLogger(
                version=run_id, save_dir=output_dir, name=""
            )

        trainer = create_trainer(
            early_stopping_patience=config.fitting.early_stopping_patience,
            logger=lightning_logger,
            valid_available=True,
            kwarg_overloads=config.fitting.trainer_kwargs,
            output_dir=output_dir,
        )
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(config.to_nested_dict())

        train_with_lightning(
            trainer,
            model,
            data,
            loss=total_loss,
            fit_config=config.fitting,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # log the final path to the trainer.logger.summary
        model_path = output_dir / "model.pt"
        lammps_model_path = output_dir / "lammps_model.pt"

        if trainer.logger is not None:
            trainer.logger.log_hyperparams(
                {
                    "model_path": model_path,
                    "lammps_model_path": lammps_model_path,
                }
            )

        torch.save(model, model_path)
        logger.info(
            "Training complete: deploying model for use with "
            f"LAMMPS to {model_path}"
        )
        deploy_model(model, cutoff=5.0, path=lammps_model_path)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if config.wandb is not None:
            wandb.finish()
        raise e


def main():
    config = extract_config_from_command_line()
    train_from_config(config)


if __name__ == "__main__":
    main()
