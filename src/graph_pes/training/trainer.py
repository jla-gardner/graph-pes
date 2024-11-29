from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from graph_pes.config import FittingOptions
from graph_pes.data.datasets import FittingData
from graph_pes.data.loader import GraphDataLoader
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.loss import TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.task import PESLearningTask
from graph_pes.training.util import (
    LoggedProgressBar,
    ModelTimer,
    log_model_info,
    sanity_check,
)
from graph_pes.utils.logger import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger

VALIDATION_LOSS_KEY = "valid/loss/total"


def train_with_lightning(
    trainer: pl.Trainer,
    model: GraphPESModel,
    data: FittingData,
    loss: TotalLoss,
    fit_config: FittingOptions,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
):
    # - prepare the data
    if trainer.global_rank == 0:
        logger.info("Preparing data")
        data.train.prepare_data()
        data.valid.prepare_data()
    trainer.strategy.barrier("data prepare")

    logger.info("Setting up datasets")
    data.train.setup()
    data.valid.setup()

    loader_kwargs = {**fit_config.loader_kwargs}
    loader_kwargs["shuffle"] = True
    train_loader = GraphDataLoader(data.train, **loader_kwargs)
    loader_kwargs["shuffle"] = False
    valid_loader = GraphDataLoader(data.valid, **loader_kwargs)

    # - maybe do some pre-fitting
    if trainer.global_rank == 0 and fit_config.pre_fit_model:
        pre_fit_dataset = data.train
        if fit_config.max_n_pre_fit is not None:
            pre_fit_dataset = pre_fit_dataset.sample(fit_config.max_n_pre_fit)
        logger.info(
            f"Pre-fitting the model on {len(pre_fit_dataset):,} samples"
        )
        model.pre_fit_all_components(pre_fit_dataset)
    trainer.strategy.barrier("pre-fit")

    # - log the model info
    if trainer.global_rank == 0:
        log_model_info(model, trainer.logger)

    # - sanity checks
    sanity_check(model, next(iter(train_loader)))

    # - create the task (a pytorch lightning module)
    task = PESLearningTask(model, loss, optimizer, scheduler)

    # - train the model
    error = None
    try:
        trainer.fit(task, train_loader, valid_loader)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        error = e

    # - load the best weights
    try:
        task.load_best_weights(model, trainer)
    except Exception as e:
        logger.error(f"Failed to load best weights: {e}")
        if error is None:
            error = e
        raise error from None


def create_trainer(
    early_stopping_patience: int | None = None,
    kwarg_overloads: dict | None = None,
    valid_available: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    progress: Literal["rich", "logged"] = "rich",
) -> pl.Trainer:
    # create the default callbacks
    callbacks: dict[str, pl.Callback] = dict(
        lr=LearningRateMonitor(logging_interval="epoch"),
        checkpoint=ModelCheckpoint(
            dirpath=None if not output_dir else output_dir / "checkpoints",
            monitor=VALIDATION_LOSS_KEY if valid_available else None,
            filename="best",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
        progress_bar={
            "rich": RichProgressBar(),
            "logged": LoggedProgressBar(),
        }[progress],
        timer=ModelTimer(),
    )
    if early_stopping_patience is not None:
        if not valid_available:
            raise ValueError(
                "Early stopping requires validation data to be available"
            )
        callbacks["early_stopping"] = EarlyStopping(
            monitor=VALIDATION_LOSS_KEY,
            patience=early_stopping_patience,
            mode="min",
            min_delta=1e-6,
        )

    # find any user defined callbacks ...
    overloads = kwarg_overloads or {}
    overloaded_callbacks = overloads.pop("callbacks", [])

    # ... and overwrite the default callbacks where necessary
    for cb in overloaded_callbacks:
        # we don't want two progress bars: use the non-default one
        if isinstance(cb, ProgressBar):
            callbacks["progress_bar"] = cb
        # all other callbacks are just added on as extras
        else:
            callbacks[str(cb)] = cb

    return pl.Trainer(
        **overloads,
        logger=logger,
        callbacks=list(callbacks.values()),
    )
