from __future__ import annotations

import argparse
import contextlib
import shutil
from datetime import datetime
from pathlib import Path

import pytorch_lightning
import torch
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.scripts.generation import config_auto_generation
from graph_pes.training.callbacks import GraphPESCallback
from graph_pes.training.trainer import create_trainer, train_with_lightning
from graph_pes.utils.lammps import deploy_model
from graph_pes.utils.logger import log_to_file, logger, set_level
from graph_pes.utils.misc import (
    build_single_nested_dict,
    nested_merge_all,
    random_dir,
    uniform_repr,
)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GraphPES model using PyTorch Lightning.",
        epilog="Copyright 2023-24, John Gardner",
    )

    parser.add_argument(
        "args",
        nargs="*",
        help=(
            "Config files and command line specifications. "
            "Config files should be YAML (.yaml/.yml) files. "
            "Command line specifications should be in the form "
            "nested^key=value. "
            "Final config is built up from these items in a left "
            "to right manner, with later items taking precedence "
            "over earlier ones in the case of conflicts."
        ),
    )

    return parser.parse_args()


def extract_config_from_command_line() -> Config:
    args = parse_args()

    # load default config
    defaults = get_default_config_values()

    parsed_configs = []

    if not args.args:
        parsed_configs.append(config_auto_generation())

    for arg in args.args:
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            # it's a config file
            with open(arg) as f:
                parsed_configs.append(yaml.safe_load(f))

        elif "=" in arg:
            # it's an override
            key, value = arg.split("=")
            keys = key.split("^")

            # parse the value
            with contextlib.suppress(yaml.YAMLError):
                value = yaml.safe_load(value)

            nested_dict = build_single_nested_dict(keys, value)
            parsed_configs.append(nested_dict)

        else:
            logger.error(
                "We detected the following command line arguments: \n" "".join(
                    f"- {arg}\n" for arg in args.args
                )
                + "We expected all of these to be in the form key=value or "
                f"to end with .yaml or .yml - {arg} is invalid."
            )

            raise ValueError(
                f"Invalid argument: {arg}. "
                "Expected a YAML file or an override in the form key=value"
            )

    final_config_dict = nested_merge_all(defaults, *parsed_configs)
    return Config.from_dict(final_config_dict)


def train_from_config(config: Config):
    """
    Train a model from a configuration object.

    We let PyTorch Lightning automatically detect and spin up the
    distributed training run if available.

    Parameters
    ----------
    config
        The configuration object.
    """
    # If we are running in a single process setting, we are always rank 0.
    # This script will run from top to bottom as normal.
    #
    # We allow for automated handling of distributed training runs. Since
    # PyTorch Lightning can do this for us automatically, we delegate all
    # DDP setup to them.
    #
    # In the distributed setting, the order of events is as follows:
    # 1. the user runs `graph-pes-train`, launching a single process on the
    #    global rank 0 process.
    # 2. (on rank 0) this script runs through until the `trainer.fit` is called
    # 3. (on rank 0) the trainer spins up the DDP backend on this process and
    #    launches the remaining processes
    # 4. (on all non-0 ranks) this script runs again until `trainer.fit` is hit.
    #    IMPORTANT: before this point the trainer is instantiated, we cannot
    #    tell from Lightning whether we are rank 0 or not - we therefore use
    #    our own logic for this.
    # 5. (on all ranks) the trainer sets up the distributed backend and
    #    synchronizes the GPUs: training then proceeds as normal.
    #
    # There are certain behaviours that we want to only handle on rank 0:
    # we can use this^ order of events to determine this. Concretely, we
    # get an identifier that is:
    #  (a) unique to the training run (to avoid collisions with any others)
    #  (b) identical across all ranks
    # and check if a directory corresponding to this identifier exists:
    # if it does, we are not rank 0, otherwise we are. We can also communicate
    # information from rank 0 to other ranks by saving information into files
    # in this directory.

    communication_dir = Path(config.general.root_dir) / ".communication"

    # the config will be shared across all ranks, but will be different
    # between different training runs: hence a sha256 hash of the config
    # is a good unique identifier for the training run:
    training_run_id = config.hash()
    training_run_dir = communication_dir / training_run_id
    is_rank_0 = not training_run_dir.exists()
    training_run_dir.mkdir(exist_ok=True, parents=True)

    def cleanup():
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(training_run_dir)

        # also remove communication if empty
        with contextlib.suppress(FileNotFoundError):
            if not any(communication_dir.iterdir()):
                shutil.rmtree(communication_dir)

    log = (lambda *args, **kwargs: None) if not is_rank_0 else logger.info

    # torch things
    prec = config.general.torch.float32_matmul_precision
    torch.set_float32_matmul_precision(prec)
    logger.info(f"Using {prec} precision for float32 matrix multiplications.")

    ftype = config.general.torch.dtype
    logger.info(f"Using {ftype} as default dtype.")
    torch.set_default_dtype(
        {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[ftype]
    )
    # a nice setting for e3nn components that get scripted upon instantiation
    # - DYNAMIC refers to the fact that they will expect different input sizes
    #   at every iteration (graphs are not all the same size)
    # - 4 is the number of times we attempt to recompile before giving up
    torch.jit.set_fusion_strategy([("DYNAMIC", 4)])

    # general things:
    pytorch_lightning.seed_everything(config.general.seed)
    set_level(config.general.log_level)
    now_ms = datetime.now().strftime("%F %T.%f")[:-3]
    logger.info(f"Started training at {now_ms}")

    # generate / look up the output directory for this training run
    if is_rank_0:
        # set up directory structure
        if config.general.run_id is None:
            output_dir = random_dir(Path(config.general.root_dir))
        else:
            output_dir = Path(config.general.root_dir) / config.general.run_id
            version = 0
            while output_dir.exists():
                version += 1
                output_dir = (
                    Path(config.general.root_dir)
                    / f"{config.general.run_id}-{version}"
                )

            if version > 0:
                logger.warning(
                    f'Specified run ID "{config.general.run_id}" already '
                    f"exists. Using {output_dir.name} instead."
                )

        output_dir.mkdir(parents=True)
        # save the config, but with the run ID updated
        config.general.run_id = output_dir.name
        with open(output_dir / "train-config.yaml", "w") as f:
            yaml.dump(config.to_nested_dict(), f)

        # communicate the output directory by saving it to a file
        with open(training_run_dir / OUTPUT_DIR, "w") as f:
            f.write(str(output_dir))

    else:
        # get the output directory from rank 0
        with open(training_run_dir / OUTPUT_DIR) as f:
            output_dir = Path(f.read())
    log(f"Output directory: {output_dir}")

    # set up a logger on every rank - PTL handles this gracefully so that
    # e.g. we don't spin up >1 wandb experiment
    if config.wandb is not None:
        lightning_logger = WandbLogger(output_dir, **config.wandb)
    else:
        lightning_logger = CSVLogger(save_dir=output_dir, name="")
    log(f"Logging using {lightning_logger}")

    # create the trainer
    trainer_kwargs = {**config.fitting.trainer_kwargs}

    callbacks = trainer_kwargs.pop("callbacks", [])
    if config.fitting.swa is not None:
        callbacks.append(config.fitting.swa.instantiate_lightning_callback())
    callbacks.extend(config.fitting.instantiate_callbacks())

    for cb in callbacks:
        if isinstance(cb, GraphPESCallback):
            cb._register_root(output_dir)

    trainer = create_trainer(
        early_stopping_patience=config.fitting.early_stopping_patience,
        logger=lightning_logger,
        valid_available=True,
        kwarg_overloads=trainer_kwargs,
        output_dir=output_dir,
        progress=config.general.progress,
        callbacks=callbacks,
    )
    assert trainer.logger is not None
    trainer.logger.log_hyperparams(config.to_nested_dict())

    # route logs to a unique file for each rank: PTL now knows the global rank!
    log_to_file(file=output_dir / "logs" / f"rank-{trainer.global_rank}.log")

    # instantiate and log things
    log(config)

    model = config.instantiate_model()  # gets logged later

    data = config.instantiate_data()
    log(data)

    optimizer = config.fitting.instantiate_optimizer()
    log(optimizer)

    scheduler = config.fitting.instantiate_scheduler()
    log(scheduler if scheduler is not None else "No LR scheduler.")

    total_loss = config.instantiate_loss()
    log(total_loss)

    def save_model():
        if not is_rank_0:
            return

        try:
            # place model onto cpu
            nonlocal model
            model = model.to("cpu")

            # log the final path to the trainer.logger.summary
            model_path = output_dir / "model.pt"
            lammps_model_path = output_dir / "lammps_model.pt"

            assert trainer.logger is not None
            trainer.logger.log_hyperparams(
                {
                    "model_path": model_path,
                    "lammps_model_path": lammps_model_path,
                }
            )
            torch.save(model, model_path)
            log(f"Model saved to {model_path}")
            deploy_model(model, path=lammps_model_path)
            log(f"Deployed model for use with LAMMPS to {lammps_model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    logger.info(f"Starting training on rank {trainer.global_rank}.")
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
        cleanup()
        save_model()
        logger.error(f"Training failed: {e}")
        raise e

    log("Training complete.")

    save_model()
    cleanup()

    log(
        "Post-training cleanup complete. "
        "Awaiting final pytorch lightning shutdown..."
    )


def main():
    config = extract_config_from_command_line()
    train_from_config(config)


if __name__ == "__main__":
    main()
