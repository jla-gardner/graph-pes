from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, Logger
from pytorch_lightning.loggers import WandbLogger as PTLWandbLogger

from graph_pes.atomic_graph import to_batch
from graph_pes.config import FittingOptions
from graph_pes.config.config import Config
from graph_pes.data.datasets import FittingData
from graph_pes.data.loader import GraphDataLoader
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.callbacks import (
    EarlyStoppingWithLogging,
    GraphPESCallback,
    OffsetLogger,
    SaveBestModel,
    ScalesLogger,
)
from graph_pes.training.loss import TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.task import PESLearningTask
from graph_pes.training.util import (
    VALIDATION_LOSS_KEY,
    LoggedProgressBar,
    ModelTimer,
    log_model_info,
    sanity_check,
)
from graph_pes.utils.logger import log_to_file, logger
from graph_pes.utils.misc import uniform_repr
from graph_pes.utils.sampling import SequenceSampler


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

    # - do some pre-fitting
    pre_fit_graphs = SequenceSampler(data.train.graphs)
    if fit_config.max_n_pre_fit is not None:
        pre_fit_graphs = pre_fit_graphs.sample_at_most(fit_config.max_n_pre_fit)
    pre_fit_graphs = list(pre_fit_graphs)

    # optionally pre-fit the model
    if fit_config.pre_fit_model:
        logger.info(f"Pre-fitting the model on {len(pre_fit_graphs):,} samples")
        model.pre_fit_all_components(pre_fit_graphs)
    trainer.strategy.barrier("pre-fit")

    # always pre-fit the losses
    for subloss in loss.losses:
        subloss.pre_fit(to_batch(pre_fit_graphs))

    # - log the model info
    if trainer.global_rank == 0:
        log_model_info(model, trainer.logger)

    # - sanity checks
    logger.info("Sanity checking the model...")
    sanity_check(model, next(iter(train_loader)))

    # - create the task (a pytorch lightning module)
    task = PESLearningTask(model, loss, optimizer, scheduler)

    # - train the model
    logger.info("Starting fit...")
    try:
        trainer.fit(task, train_loader, valid_loader)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        try:
            task.load_best_weights(model, trainer)
        except Exception as e:
            logger.error(f"Failed to load best weights: {e}")


def create_trainer(
    early_stopping_patience: int | None = None,
    kwarg_overloads: dict | None = None,
    valid_available: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    progress: Literal["rich", "logged"] = "rich",
    callbacks: list[pl.Callback] | None = None,
) -> pl.Trainer:
    if callbacks is None:
        callbacks = []
    if kwarg_overloads is None:
        kwarg_overloads = {}
    callbacks.extend(kwarg_overloads.pop("callbacks", []))

    # add default callbacks
    callbacks.extend([ModelTimer(), SaveBestModel()])

    if not any(isinstance(c, ProgressBar) for c in callbacks):
        callbacks.append(
            {"rich": RichProgressBar(), "logged": LoggedProgressBar()}[progress]
        )

    if not any(isinstance(c, OffsetLogger) for c in callbacks):
        callbacks.append(OffsetLogger())

    if not any(isinstance(c, ModelCheckpoint) for c in callbacks):
        checkpoint_dir = None if not output_dir else output_dir / "checkpoints"
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor=VALIDATION_LOSS_KEY if valid_available else None,
                    filename="best",
                    mode="min",
                    save_top_k=1,
                    save_weights_only=True,
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="last",
                    # ensure that we can load the complete trainer state,
                    # callbacks and all
                    save_weights_only=False,
                ),
            ]
        )

    if not any(isinstance(c, LearningRateMonitor) for c in callbacks):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if early_stopping_patience is not None:
        if not valid_available:
            raise ValueError(
                "Early stopping requires validation data to be available"
            )
        callbacks.append(
            EarlyStoppingWithLogging(
                monitor=VALIDATION_LOSS_KEY,
                patience=early_stopping_patience,
                mode="min",
                min_delta=1e-6,
            )
        )

    return pl.Trainer(**kwarg_overloads, logger=logger, callbacks=callbacks)


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


def trainer_from_config(
    config: Config,
    output_dir: Path,
    logging_function: Callable = lambda x: None,
) -> pl.Trainer:
    # set up a logger on every rank - PTL handles this gracefully so that
    # e.g. we don't spin up >1 wandb experiment
    if config.wandb is not None:
        lightning_logger = WandbLogger(output_dir, **config.wandb)
    else:
        lightning_logger = CSVLogger(save_dir=output_dir, name="")
    logging_function(f"Logging using {lightning_logger}")

    # create the trainer
    trainer_kwargs = {**config.fitting.trainer_kwargs}

    # extract the callbacks from trainer_kwargs, since we
    # handle them specially below
    callbacks = trainer_kwargs.pop("callbacks", [])
    callbacks.extend(config.fitting.callbacks)

    default_callbacks = [OffsetLogger, ScalesLogger]
    for klass in default_callbacks:
        if not any(isinstance(cb, klass) for cb in callbacks):
            callbacks.append(klass())
    if config.fitting.swa is not None:
        callbacks.append(config.fitting.swa.instantiate_lightning_callback())

    logging_function("Creating trainer...")
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
    final_callbacks: list[pl.Callback] = trainer.callbacks  # type: ignore
    for cb in final_callbacks:
        if isinstance(cb, GraphPESCallback):
            cb._register_root(output_dir)
    logging_function(f"Callbacks: {final_callbacks}")

    return trainer
