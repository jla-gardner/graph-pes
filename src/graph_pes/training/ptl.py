from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from graph_pes.config import FittingOptions
from graph_pes.core import GraphPESModel, get_predictions
from graph_pes.data.dataset import FittingData
from graph_pes.data.loader import GraphDataLoader
from graph_pes.graphs import AtomicGraphBatch, LabelledBatch, keys
from graph_pes.graphs.operations import number_of_structures
from graph_pes.logger import logger
from graph_pes.training.loss import RMSE, Loss, PerAtomEnergyLoss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.util import log_model_info
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

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
    # - get the data ready in a way that is compatible with
    #   multi-GPU training
    if trainer.local_rank == 0:
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
    if fit_config.pre_fit_model:
        pre_fit_dataset = data.train
        if fit_config.max_n_pre_fit is not None:
            pre_fit_dataset = pre_fit_dataset.sample(fit_config.max_n_pre_fit)
        logger.info(f"Pre-fitting the model on {len(pre_fit_dataset)} samples")
        model.pre_fit(pre_fit_dataset)

    # - log the model info
    log_model_info(model, trainer.logger)

    # - create the task (a pytorch lightning module)
    task = LearnThePES(model, loss, optimizer, scheduler)

    # - train the model
    trainer.fit(task, train_loader, valid_loader)

    # - load the best weights
    task.load_best_weights(model, trainer)


class LearnThePES(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        loss: TotalLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.total_loss = loss
        self.properties: list[keys.LabelKey] = [
            component.property_key for component in self.total_loss.losses
        ]

        # TODO: we want to base this on actually available data
        # not just the total loss components
        validation_metrics = []

        existing_loss_names = [l.name for l in self.total_loss.losses]

        if keys.ENERGY in self.properties:
            pael = PerAtomEnergyLoss()
            if pael.name not in existing_loss_names:
                validation_metrics.append(PerAtomEnergyLoss())

        if keys.FORCES in self.properties:
            fr = Loss("forces", RMSE())
            if fr.name not in existing_loss_names:
                validation_metrics.append(fr)

        self.validation_metrics = validation_metrics

    def forward(self, graphs: AtomicGraphBatch) -> torch.Tensor:
        return self.model(graphs)

    def _step(self, graph: LabelledBatch, prefix: Literal["train", "valid"]):
        """
        Get (and log) the losses for a training/validation step.
        """

        def log(name: str, value: torch.Tensor | float):
            if isinstance(value, torch.Tensor):
                value = value.item()

            return self.log(
                f"{prefix}/{name}",
                value,
                prog_bar=prefix == "valid" and "loss" in name,
                on_step=prefix == "train",
                on_epoch=prefix == "valid",
                batch_size=number_of_structures(graph),
            )

        # generate prediction:
        predictions = get_predictions(
            self.model, graph, properties=self.properties, training=True
        )

        # compute the loss and its sub-components
        total_loss_result = self.total_loss(predictions, graph)

        # log
        log("loss/total", total_loss_result.loss_value)
        for name, loss_pair in total_loss_result.components.items():
            log(f"metrics/{name}", loss_pair.loss_value)
            log(f"loss/{name}_component", loss_pair.weighted_loss_value)

        # log additional values during validation
        if prefix == "valid":
            with torch.no_grad():
                for val_loss in self.validation_metrics:
                    value = val_loss(predictions, graph)
                    log(f"metrics/{val_loss.name}", value)

        return total_loss_result.loss_value

    def training_step(self, structure: LabelledBatch, _):
        return self._step(structure, "train")

    def validation_step(self, structure: LabelledBatch, _):
        return self._step(structure, "valid")

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        opt = self.optimizer_factory(self.model)
        if self.scheduler_factory is None:
            return opt

        scheduler = self.scheduler_factory(opt)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            trainer = self.trainer
            assert trainer is not None

            # if the user is validating once every n training epochs,
            # set the scheduler to update every n training epochs
            if trainer.check_val_every_n_epoch:
                config = {
                    "interval": "epoch",  # really this means training epoch
                    "frequency": trainer.check_val_every_n_epoch,
                    "monitor": VALIDATION_LOSS_KEY,
                    "strict": True,
                }

            # otherwise, they are validating every n training steps:
            # set the scheduler to update every n training steps
            else:
                if isinstance(trainer.val_check_interval, float):
                    raise ValueError(
                        "`graph-pes` does not support specifying "
                        "`val_check_interval` as a fraction of the training "
                        "data size. Please specify it as an integer number of "
                        "training steps."
                    )
                config = {
                    "interval": "step",
                    "frequency": trainer.val_check_interval,
                    "monitor": VALIDATION_LOSS_KEY,
                    "strict": True,
                }

        else:
            # vanilla LR scheduler: update every training step
            config = {
                "interval": "step",
                "frequency": 1,
            }

        logger.debug(f"Using LR scheduler config:\n{config}")
        config["scheduler"] = scheduler
        return {"optimizer": opt, "lr_scheduler": config}  # type: ignore

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        # we override the defaults to turn on gradient tracking for the
        # validation step since we (might) compute the forces using autograd
        torch.set_grad_enabled(True)

    @classmethod
    def load_best_weights(
        cls,
        model: GraphPESModel,
        trainer: pl.Trainer | None = None,
        checkpoint_path: Path | str | None = None,
    ):
        if checkpoint_path is None and trainer is None:
            raise ValueError(
                "Either trainer or checkpoint_path must be provided"
            )
        if checkpoint_path is None:
            path = trainer.checkpoint_callback.best_model_path  # type: ignore
        else:
            path = Path(checkpoint_path)
        logger.info(f'Loading best weights from "{path}"')
        checkpoint = torch.load(path)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)


def create_trainer(
    early_stopping_patience: int | None = None,
    kwarg_overloads: dict | None = None,
    valid_available: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
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
        progress_bar=RichProgressBar(),
        timer=ModelTimer(),
    )
    if early_stopping_patience is not None:
        callbacks["early_stopping"] = EarlyStopping(
            monitor=VALIDATION_LOSS_KEY,
            patience=early_stopping_patience,
            mode="min",
            min_delta=1e-6,
        )

    # find any user defined callbacks
    overloads = kwarg_overloads or {}
    overloaded_callbacks = overloads.pop("callbacks", [])

    # and overwrite the default callbacks where necessary
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


class ModelTimer(pl.Callback):
    def __init__(self):
        super().__init__()
        self.tick_ms: float | None = None

    def start(self):
        self.tick_ms = time.time_ns() // 1_000_000

    def stop(self, pl_module: LearnThePES, stage: Literal["train", "valid"]):
        assert self.tick_ms is not None
        duration_ms = time.time_ns() // 1_000_000 - self.tick_ms
        self.tick_ms = None

        for name, x in (
            ("step_duration_ms", duration_ms),
            ("its_per_s", 1_000 / duration_ms),
        ):
            pl_module.log(
                f"timer/{name}/{stage}",
                x,
                batch_size=1,
                on_epoch=stage == "valid",
                on_step=stage == "train",
            )

    def on_train_batch_start(self, *args, **kwargs):
        self.start()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: LearnThePES, *args, **kwargs
    ):
        self.stop(pl_module, "train")

    def on_validation_batch_start(self, *args, **kwargs):
        self.start()

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: LearnThePES, *args, **kwargs
    ):
        self.stop(pl_module, "valid")
