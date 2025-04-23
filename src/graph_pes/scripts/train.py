from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from graph_pes.config.shared import (
    instantiate_config_from_dict,
    parse_loss,
    parse_model,
)
from graph_pes.config.training import EarlyStoppingConfig, TrainingConfig
from graph_pes.scripts.utils import (
    configure_general_options,
    extract_config_dict_from_command_line,
    update_summary,
)
from graph_pes.training.callbacks import (
    EarlyStoppingWithLogging,
    GraphPESCallback,
    LoggedProgressBar,
    ModelTimer,
    OffsetLogger,
    SaveBestModel,
    ScalesLogger,
    WandbLogger,
)
from graph_pes.training.tasks import test_with_lightning, train_with_lightning
from graph_pes.training.utils import VALIDATION_LOSS_KEY
from graph_pes.utils import distributed
from graph_pes.utils.logger import log_to_file, logger, set_log_level
from graph_pes.utils.misc import random_dir


def train_from_config(config_data: dict):
    """
    Train a model from a nested dictionary of config.

    We let PyTorch Lightning automatically detect and spin up the
    distributed training run if available.

    Parameters
    ----------
    config
        The configuration data.
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

    _log_level = config_data.get("general", {}).get("log_level", "INFO")
    set_log_level(_log_level)

    now_ms = datetime.now().strftime("%F %T.%f")[:-3]
    logger.info(f"Started `graph-pes-train` at {now_ms}")

    logger.debug("Parsing config...")

    config_data, config = instantiate_config_from_dict(
        config_data, TrainingConfig
    )
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

    log_to_file(output_dir)
    # use the updated output_dir to extract the actual run ID:
    config_data["general"]["run_id"] = output_dir.name
    config.general.run_id = output_dir.name
    logger.info(f"ID for this training run: {config.general.run_id}")
    logger.info(f"""\
Output for this training run can be found at:
   └─ {output_dir}
      ├─ rank-0.log         # find a verbose log here
      ├─ model.pt           # the best model (according to valid/loss/total)
      ├─ lammps_model.pt    # the best model deployed to LAMMPS
      ├─ train-config.yaml  # the complete config used for this run
      └─ summary.yaml       # the summary of the training run
""")

    configure_general_options(config.general.torch, config.general.seed)

    trainer = trainer_from_config(config, output_dir)

    assert trainer.logger is not None
    trainer.logger.log_hyperparams(config_data)

    # instantiate and log things
    model = parse_model(config.model)
    logger.debug(f"Model:\n{model}")

    data = config.get_data(model)
    logger.debug(f"Data:\n{data}")

    optimizer = config.fitting.get_optimizer()
    logger.debug(f"Optimizer:\n{optimizer}")

    scheduler = config.fitting.get_scheduler()
    _scheduler_str = scheduler if scheduler is not None else "No LR scheduler."
    logger.debug(f"Scheduler:\n{_scheduler_str}")

    total_loss = parse_loss(config.loss)
    logger.debug(f"Total loss:\n{total_loss}")

    if (
        abs(model.cutoff.item() - data.train[0].cutoff) > 1e-4
        and model.three_body_cutoff.item() > 0
    ):
        logger.warning(
            f"""You are using a model with 3 body contributions, but there is a cutoff mismatch:
   Model cutoff: {model.cutoff.item():.4f}
   Data cutoff: {data.train[0].cutoff:.4f}
   
This mismatch prevents efficient caching of three-body neighborlists during training,
which will significantly slow down your training process.

To fix: Ensure your model cutoff matches your dataset cutoff."""  # noqa: E501
        )

    model = train_with_lightning(
        trainer,
        model,
        data,
        loss=total_loss,
        fit_config=config.fitting,
        optimizer=optimizer,
        scheduler=scheduler,
        user_eval_metrics=[],
    )
    update_summary(trainer.logger, output_dir / "summary.yaml")
    logger.info("Training complete.")

    logger.info("Testing best model...")
    test_datasets = {
        "train": data.train,
        "valid": data.valid,
    }
    if data.test is not None:
        if isinstance(data.test, dict):
            test_datasets.update(data.test)
        else:
            test_datasets["test"] = data.test

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger._log_epoch = False

    tester = pl.Trainer(
        logger=trainer.logger,
        accelerator=trainer.accelerator,
        inference_mode=False,
    )
    test_with_lightning(
        tester,
        model,
        test_datasets,
        loader_kwargs=config.fitting.loader_kwargs,
        logging_prefix="test",
        user_eval_metrics=[],
    )
    update_summary(tester.logger, output_dir / "summary.yaml")

    logger.info("Testing complete.")
    logger.info("Awaiting final Lightning and W&B shutdown...")


def trainer_from_config(
    config: TrainingConfig,
    output_dir: Path,
) -> pl.Trainer:
    # set up a logger on every rank - PTL handles this gracefully so that
    # e.g. we don't spin up >1 wandb experiment
    if config.wandb is not None:
        lightning_logger = WandbLogger(
            output_dir, log_epoch=True, **config.wandb
        )
    else:
        lightning_logger = CSVLogger(save_dir=output_dir, name="")
    logger.debug(f"Logging using {lightning_logger}")

    # create the trainer
    trainer_kwargs = {**config.fitting.trainer_kwargs}
    trainer_kwargs["inference_mode"] = False

    # set up the callbacks
    callbacks = trainer_kwargs.pop("callbacks", [])
    callbacks.extend(config.fitting.callbacks)

    for klass in [OffsetLogger, ScalesLogger, ModelTimer, SaveBestModel]:
        if not any(isinstance(cb, klass) for cb in callbacks):
            callbacks.append(klass())
    if config.fitting.swa is not None:
        callbacks.append(config.fitting.swa.instantiate_lightning_callback())
    if not any(isinstance(c, ProgressBar) for c in callbacks):
        callbacks.append(
            {"rich": RichProgressBar(), "logged": LoggedProgressBar()}[
                config.general.progress
            ]
        )
    if not any(isinstance(c, LearningRateMonitor) for c in callbacks):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    early_stopping_config = None
    # TODO: remove this at a later date
    if config.fitting.early_stopping_patience is not None:
        logger.warning(
            "`early_stopping_patience` is deprecated. Use the `early_stopping` "
            "config option instead.",
        )
        early_stopping_config = EarlyStoppingConfig(
            patience=config.fitting.early_stopping_patience,
            min_delta=1e-6,
            monitor=VALIDATION_LOSS_KEY,
        )
    if config.fitting.early_stopping is not None:
        early_stopping_config = config.fitting.early_stopping

    if early_stopping_config is not None:
        callbacks.append(
            EarlyStoppingWithLogging(
                monitor=early_stopping_config.monitor,
                patience=early_stopping_config.patience,
                mode="min",
                min_delta=early_stopping_config.min_delta,
            )
        )

    if not any(isinstance(c, ModelCheckpoint) for c in callbacks):
        checkpoint_dir = None if not output_dir else output_dir / "checkpoints"
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor=VALIDATION_LOSS_KEY,
                    filename="best",
                    mode="min",
                    save_top_k=1,
                    save_weights_only=True,
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="last",
                    # ensure that we can load the complete trainer state,
                    # callbacks and all, for resuming later
                    save_weights_only=False,
                ),
            ]
        )

    # special handling for GraphPESCallback: we need to register the
    # output directory with it so that it knows where to save the model etc.
    for cb in callbacks:
        if isinstance(cb, GraphPESCallback):
            cb._register_root(output_dir)
    logger.debug(f"Callbacks: {callbacks}")

    return pl.Trainer(
        **trainer_kwargs,
        logger=lightning_logger,
        callbacks=callbacks,
    )


def main():
    # set the load-atoms verbosity to 1 by default to avoid
    # spamming logs with `rich` output
    os.environ["LOAD_ATOMS_VERBOSE"] = os.getenv("LOAD_ATOMS_VERBOSE", "1")

    # build up the config dict from available sources:
    config_dict = extract_config_dict_from_command_line(
        "Train a GraphPES model using PyTorch Lightning."
    )
    train_from_config(config_dict)


if __name__ == "__main__":
    main()
