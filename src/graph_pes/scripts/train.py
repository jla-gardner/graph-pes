from __future__ import annotations

import argparse
import contextlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning
import torch
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.deploy import deploy_model
from graph_pes.logger import log_to_file, logger, set_level
from graph_pes.scripts.generation import config_auto_generation
from graph_pes.training.ptl import create_trainer, train_with_lightning
from graph_pes.util import nested_merge, random_dir, rank, uniform_repr
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger as PTLWandbLogger


class WandbLogger(PTLWandbLogger):
    """A subclass of WandbLogger that automatically sets the id and save_dir."""

    def __init__(self, output_dir: Path, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = output_dir.name
        if "save_dir" not in kwargs:
            kwargs["save_dir"] = str(output_dir.parent)
        super().__init__(**kwargs)
        self._kwargs = kwargs

    def __repr__(self):
        return uniform_repr(self.__class__.__name__, **self._kwargs)


OUTPUT_DIR = "graph-pes-output-dir"


class Communication:
    """A class to manage communication between ranks."""

    def __init__(self, dir: Path):
        assert dir.is_dir() and not dir.exists()
        dir.mkdir()
        self.dir = dir

    def send(self, key: str, value: Any):
        with open(self.dir / key, "w") as f:
            f.write(str(value))

    def receive(self, key: str) -> Any:
        with open(self.dir / key) as f:
            return f.read()

    def cleanup(self):
        shutil.rmtree(self.dir)


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
        help=(
            "Path to the configuration file. "
            "This argument can be used multiple times, with later files "
            "taking precedence over earlier ones in the case of conflicts. "
            "If no config files are provided, the script will auto-generate."
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

    if not args.config:
        # TODO: change this to just an alternative way to get user_config
        return config_auto_generation()

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
    # hash the config to get a unique identifier using sha256
    communication = Communication(dir=Path(config.hash()))

    # general things: seed and logging
    pytorch_lightning.seed_everything(config.general.seed)
    set_level(config.general.log_level)
    now_ms = datetime.now().strftime("%F %T.%f")[:-3]
    logger.info(f"Started training at {now_ms}")

    # rank-0 only things
    if rank() == 0:
        # set up directory structure
        output_dir = random_dir(root=Path(config.general.root_dir))
        assert not output_dir.exists()
        output_dir.mkdir(parents=True)
        # save the config
        with open(output_dir / "train-config.yaml", "w") as f:
            yaml.dump(config.to_nested_dict(), f)

        # communicate the output directory by saving it to a file
        communication.send(OUTPUT_DIR, output_dir)

    else:
        output_dir = Path(communication.receive(OUTPUT_DIR))

    # route logs to file
    log_to_file(file=output_dir / "logs" / f"rank-{rank()}.log")

    # instantiate and log things
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

    if config.wandb is not None:
        lightning_logger = WandbLogger(output_dir, **config.wandb)
    else:
        lightning_logger = CSVLogger(save_dir=output_dir, name="")
    logger.info(f"Logging using {lightning_logger}")

    trainer = create_trainer(
        early_stopping_patience=config.fitting.early_stopping_patience,
        logger=lightning_logger,
        valid_available=True,
        kwarg_overloads=config.fitting.trainer_kwargs,
        output_dir=output_dir,
    )
    assert trainer.logger is not None
    trainer.logger.log_hyperparams(config.to_nested_dict())

    try:
        train_with_lightning(
            trainer,
            model,
            data,
            loss=total_loss,
            fit_config=config.fitting,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    except Exception as e:
        communication.cleanup()
        raise e

    logger.info("Training complete.")

    try:
        # log the final path to the trainer.logger.summary
        model_path = output_dir / "model.pt"
        lammps_model_path = output_dir / "lammps_model.pt"

        trainer.logger.log_hyperparams(
            {
                "model_path": model_path,
                "lammps_model_path": lammps_model_path,
            }
        )
        logger.info(f"Model saved to {model_path}")
        torch.save(model, model_path)
        logger.info(
            f"Deploying model for use with LAMMPS to {lammps_model_path}"
        )
        deploy_model(model, cutoff=5.0, path=lammps_model_path)

    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    communication.cleanup()


def main():
    config = extract_config_from_command_line()
    train_from_config(config)


if __name__ == "__main__":
    main()
