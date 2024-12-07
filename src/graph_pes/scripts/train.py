from __future__ import annotations

import argparse
import contextlib
import os
import random
import shutil
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import yaml
from graph_pes.config import Config, get_default_config_values
from graph_pes.scripts.generation import config_auto_generation
from graph_pes.training.callbacks import GraphPESCallback
from graph_pes.training.trainer import create_trainer, train_with_lightning
from graph_pes.utils.logger import log_to_file, logger, set_level
from graph_pes.utils.misc import (
    build_single_nested_dict,
    nested_merge_all,
    random_dir,
    uniform_repr,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger as PTLWandbLogger


def set_global_seed(seed: int):
    # a non-verbose version of pl.seed_everything
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_SEED_WORKERS"] = "0"


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
            "my/nested/key=value. "
            "Final config is built up from these items in a left "
            "to right manner, with later items taking precedence "
            "over earlier ones in the case of conflicts. "
            "The data2objects package is used to resolve references "
            "and create objects directly from the config dictionary."
        ),
    )

    return parser.parse_args()


def extract_config_from_command_line() -> dict:
    args = parse_args()

    # load default config
    defaults = get_default_config_values()

    parsed_configs = []

    if not args.args:
        parsed_configs.append(config_auto_generation())

    for arg in args.args:
        arg: str

        if arg.endswith(".yaml") or arg.endswith(".yml"):
            # it's a config file
            try:
                with open(arg) as f:
                    parsed_configs.append(yaml.safe_load(f))
            except Exception as e:
                logger.error(
                    f"You specified a config file ({arg}) "
                    "that we couldn't load."
                )
                raise e

        elif "=" in arg:
            # it's an override
            key, value = arg.split("=", maxsplit=1)
            keys = key.split("/")

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

    return nested_merge_all(defaults, *parsed_configs)


def train_from_config(config_data: dict):
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

    # the config will be shared across all ranks, but will be different
    # between different training runs: hence a sha256 hash of the config
    # is a good unique identifier for the training run:
    training_run_id = sha256(str(config_data).encode()).hexdigest()
    communication_dir = (
        Path(config_data["general"]["root_dir"]) / ".communication"
    )
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

    set_level(config_data["general"]["log_level"])
    info = (lambda *args, **kwargs: None) if not is_rank_0 else logger.info
    debug = (lambda *args, **kwargs: None) if not is_rank_0 else logger.debug

    now_ms = datetime.now().strftime("%F %T.%f")[:-3]
    info(f"Started `graph-pes-train` at {now_ms}")

    debug("Parsing config...")
    config_data, config = Config.from_raw_config_dicts(config_data)
    info("Successfully parsed config.")

    # torch things
    prec = config.general.torch.float32_matmul_precision
    torch.set_float32_matmul_precision(prec)
    debug(f"Using {prec} precision for float32 matrix multiplications.")

    ftype = config.general.torch.dtype
    debug(f"Using {ftype} as default dtype.")
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
    set_global_seed(config.general.seed)

    # generate / look up the output directory for this training run
    # and handle the case where there is an ID collision by incrementing
    # the version number
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
        with open(output_dir / "train-config.yaml", "w") as f:
            yaml.dump(config_data, f)

        # communicate the output directory by saving it to a file
        with open(training_run_dir / OUTPUT_DIR, "w") as f:
            f.write(str(output_dir))

    else:
        # get the output directory from rank 0
        with open(training_run_dir / OUTPUT_DIR) as f:
            output_dir = Path(f.read())

    # update the run id
    config_data["general"]["run_id"] = output_dir.name
    config.general.run_id = output_dir.name
    info(f"ID for this training run: {config.general.run_id}")
    info(f"""\
Output for this training run can be found at:
   └─ {output_dir}
      ├─ logs/rank-0.log    # find a verbose log here
      ├─ model.pt           # the best model
      ├─ lammps_model.pt    # the best model deployed to LAMMPS
      └─ train-config.yaml  # the complete config used for this run\
""")

    # set up a logger on every rank - PTL handles this gracefully so that
    # e.g. we don't spin up >1 wandb experiment
    if config.wandb is not None:
        lightning_logger = WandbLogger(output_dir, **config.wandb)
    else:
        lightning_logger = CSVLogger(save_dir=output_dir, name="")
    debug(f"Logging using {lightning_logger}")

    # create the trainer
    trainer_kwargs = {**config.fitting.trainer_kwargs}

    # extract the callbacks from trainer_kwargs, since we
    # handle them specially below
    callbacks = trainer_kwargs.pop("callbacks", [])
    if config.fitting.swa is not None:
        callbacks.append(config.fitting.swa.instantiate_lightning_callback())
    callbacks.extend(config.fitting.callbacks)

    debug("Creating trainer...")
    trainer = create_trainer(
        early_stopping_patience=config.fitting.early_stopping_patience,
        logger=lightning_logger,
        valid_available=True,
        kwarg_overloads=trainer_kwargs,
        output_dir=output_dir,
        progress=config.general.progress,
        callbacks=callbacks,
    )

    # route logs to a unique file for each rank: PTL now knows the global rank!
    log_to_file(file=output_dir / "logs" / f"rank-{trainer.global_rank}.log")

    # special handling for GraphPESCallback: we need to register the
    # output directory with it so that it knows where to save the model etc.
    actual_callbacks: list[pl.Callback] = trainer.callbacks  # type: ignore
    for cb in actual_callbacks:
        if isinstance(cb, GraphPESCallback):
            cb._register_root(output_dir)
    debug(f"Callbacks: {actual_callbacks}")

    assert trainer.logger is not None
    trainer.logger.log_hyperparams(config_data)

    # instantiate and log things
    model = config.get_model()  # gets logged later

    data = config.get_data()
    debug(f"Data:\n{data}")

    optimizer = config.fitting.optimizer
    debug(f"Optimizer:\n{optimizer}")

    scheduler = config.fitting.scheduler
    _scheduler_str = scheduler if scheduler is not None else "No LR scheduler."
    debug(f"Scheduler:\n{_scheduler_str}")

    total_loss = config.get_loss()
    debug(f"Total loss:\n{total_loss}")

    debug(f"Starting training on rank {trainer.global_rank}.")
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
        logger.error(f"Training failed: {e}")
        raise e

    info("Training complete. Awaiting final Lightning and W&B shutdown...")
    cleanup()


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    config = extract_config_from_command_line()
    train_from_config(config)


if __name__ == "__main__":
    main()
