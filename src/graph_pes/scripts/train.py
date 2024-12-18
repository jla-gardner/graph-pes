from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from graph_pes.config import Config, get_default_config_values
from graph_pes.config.utils import instantiate_config_from_dict
from graph_pes.scripts.utils import extract_config_dict_from_command_line
from graph_pes.training.trainer import (
    WandbLogger,
    train_with_lightning,
    trainer_from_config,
)
from graph_pes.utils import distributed
from graph_pes.utils.logger import log_to_file, logger, set_log_level
from graph_pes.utils.misc import nested_merge_all, random_dir


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
    # 5. (on all ranks) the trainer sets up the distributed backend and
    #    synchronizes the GPUs: training then proceeds as normal.

    set_log_level(config_data["general"]["log_level"])

    now_ms = datetime.now().strftime("%F %T.%f")[:-3]
    logger.info(f"Started `graph-pes-train` at {now_ms}")

    logger.debug("Parsing config...")

    # handle default optimizer
    if config_data["fitting"]["optimizer"] is None:
        config_data["fitting"]["optimizer"] = yaml.safe_load(
            """
            +Optimizer:
                name: Adam
                lr: 0.001
            """
        )
    try:
        config_data, config = instantiate_config_from_dict(config_data, Config)
    except Exception as e:
        raise ValueError(
            "Your configuration file could not be successfully parsed. "
            "Please check that it is formatted correctly. For examples, "
            "please see https://jla-gardner.github.io/graph-pes/cli/graph-pes-train.html"
        ) from e
    logger.info("Successfully parsed config.")

    # generate / look up the output directory for this training run
    # and handle the case where there is an ID collision by incrementing
    # the version number
    if distributed.IS_RANK_0:
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

        # communicate the output directory to other ranks
        distributed.send_to_other_ranks("OUTPUT_DIR", str(output_dir))

    else:
        # get the output directory from rank 0
        output_dir = Path(distributed.receive_from_rank_0("OUTPUT_DIR"))

    # log
    log_to_file(output_dir)

    # torch things
    configure_general_options(config)

    # update the run id
    config_data["general"]["run_id"] = output_dir.name
    config.general.run_id = output_dir.name
    logger.info(f"ID for this training run: {config.general.run_id}")
    wandb_line = (
        """\
      ├─ .wandb.id          # file containing the wandb ID\n"""
        if config.wandb is not None
        else ""
    )
    logger.info(f"""\
Output for this training run can be found at:
   └─ {output_dir}
      ├─ logs/rank-0.log    # find a verbose log here{wandb_line}
      ├─ model.pt           # the best model
      ├─ lammps_model.pt    # the best model deployed to LAMMPS
      └─ train-config.yaml  # the complete config used for this run\
""")

    trainer = trainer_from_config(config, output_dir)

    assert trainer.logger is not None
    if config.wandb is not None:
        assert isinstance(trainer.logger, WandbLogger)
        (output_dir / ".wandb.id").write_text(str(trainer.logger._id))
    trainer.logger.log_hyperparams(config_data)

    # instantiate and log things
    model = config.get_model()  # gets logged later

    data = config.get_data()
    logger.debug(f"Data:\n{data}")

    optimizer = config.fitting.optimizer
    logger.debug(f"Optimizer:\n{optimizer}")

    scheduler = config.fitting.scheduler
    _scheduler_str = scheduler if scheduler is not None else "No LR scheduler."
    logger.debug(f"Scheduler:\n{_scheduler_str}")

    total_loss = config.get_loss()
    logger.debug(f"Total loss:\n{total_loss}")

    logger.debug(f"Starting training on rank {trainer.global_rank}.")
    train_with_lightning(
        trainer,
        model,
        data,
        loss=total_loss,
        fit_config=config.fitting,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    logger.info(
        "Training complete. Awaiting final Lightning and W&B shutdown..."
    )


def configure_general_options(config: Config):
    prec = config.general.torch.float32_matmul_precision
    torch.set_float32_matmul_precision(prec)
    logger.debug(f"Using {prec} precision for float32 matrix multiplications.")

    ftype = config.general.torch.dtype
    logger.debug(f"Using {ftype} as default dtype.")
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

    # a non-verbose version of pl.seed_everything
    seed = config.general.seed
    logger.debug(f"Using seed {seed} for reproducibility.")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_SEED_WORKERS"] = "0"


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    # build up the config dict from available sources:
    defaults = get_default_config_values()
    cli_config = extract_config_dict_from_command_line(
        "Train a GraphPES model using PyTorch Lightning."
    )
    config_dict = nested_merge_all(defaults, cli_config)

    # train
    train_from_config(config_dict)


if __name__ == "__main__":
    main()
