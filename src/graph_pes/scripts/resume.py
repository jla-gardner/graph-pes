import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

from graph_pes.config.shared import (
    instantiate_config_from_dict,
    parse_loss,
    parse_model,
)
from graph_pes.config.training import TrainingConfig
from graph_pes.data.loader import GraphDataLoader
from graph_pes.scripts.train import trainer_from_config
from graph_pes.training.tasks import TrainingTask
from graph_pes.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume a `graph-pes-train` training run.",
        epilog="Copyright 2023-25, John Gardner",
    )
    parser.add_argument(
        "train_directory",
        type=str,
        help=(
            "Path to the training directory. For instance, "
            "`graph-pes-results/abdcefg_hijklmn`"
        ),
    )

    return parser.parse_args()


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    args = parse_args()

    train_dir = Path(args.train_directory)
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")

    # find the latest checkpoint
    checkpoint_path = train_dir / "checkpoints/last.ckpt"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # and the training config
    config_path = train_dir / "train-config.yaml"
    assert config_path.exists(), f"Training config not found: {config_path}"

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # load the checkpoint
    config_data, config = instantiate_config_from_dict(
        config_data, TrainingConfig
    )
    task = TrainingTask.load_from_checkpoint(
        checkpoint_path,
        model=parse_model(config.model),
        loss=parse_loss(config.loss),
        optimizer=config.fitting.get_optimizer(),
        scheduler=config.fitting.get_scheduler(),
    )

    # create the trainer
    trainer = trainer_from_config(config, train_dir)
    if trainer.global_rank == 0:
        now_ms = datetime.now().strftime("%F %T.%f")[:-3]
        logger.info(f"Resuming training at {now_ms}")

    # resume training
    data = config.get_data(task.model)
    loader_kwargs = {**config.fitting.loader_kwargs}
    loader_kwargs["shuffle"] = True
    train_loader = GraphDataLoader(data.train, **loader_kwargs)
    loader_kwargs["shuffle"] = False
    valid_loader = GraphDataLoader(data.valid, **loader_kwargs)
    trainer.fit(task, train_loader, valid_loader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()
