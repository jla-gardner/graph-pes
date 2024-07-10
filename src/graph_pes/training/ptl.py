from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import torch
from graph_pes.config import FittingOptions
from graph_pes.core import GraphPESModel, get_predictions
from graph_pes.data.module import GraphDataModule
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
    RichProgressBar,
)
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


def train_with_lightning(
    model: GraphPESModel,
    data: GraphDataModule,
    loss: TotalLoss,
    fit_config: FittingOptions,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    # fitting_options: FittingOptions,
    config_to_log: dict | None = None,
) -> pl.Trainer:
    """Hand off to Lightning."""

    # - check that we can actually do this
    if not data.has_stage("train") or not data.has_stage("val"):
        raise ValueError(
            "The data module must be able to provide training and validation "
            "data."
        )

    # - create the trainer
    trainer = create_training_trainer(fit_config, True)
    if config_to_log is not None and trainer.logger is not None:
        trainer.logger.log_hyperparams(config_to_log)

    # - get the data ready in a way that is compatible with
    #   multi-GPU training
    if trainer.local_rank == 0:
        data.prepare_data()
    trainer.strategy.barrier("data ready")

    # - maybe do some pre-fitting
    if fit_config.pre_fit_model:
        # TODO: log
        pre_fit_dataset = data.train_dataset()
        if fit_config.max_n_pre_fit is not None:
            pre_fit_dataset = pre_fit_dataset.sample(fit_config.max_n_pre_fit)
        logger.info("Pre-fitting the model")
        model.pre_fit(pre_fit_dataset)
    log_model_info(model)

    # - create the task (a pytorch lightning module)
    task = LearnThePES(model, loss, optimizer, scheduler)

    # - train the model
    trainer.fit(task, data)

    # - load the best weights
    task.load_best_weights(model, trainer)

    # - maybe hand off for some testing? (TODO)

    return trainer


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

    def _step(self, graph: LabelledBatch, prefix: Literal["train", "val"]):
        """
        Get (and log) the losses for a training/validation step.
        """

        def log(name: str, value: torch.Tensor | float):
            if isinstance(value, torch.Tensor):
                value = value.item()

            return self.log(
                f"{prefix}/{name}",
                value,
                prog_bar=prefix == "val" and "loss" in name,
                on_step=prefix == "train",
                on_epoch=prefix == "val",
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
        if prefix == "val":
            with torch.no_grad():
                for val_loss in self.validation_metrics:
                    value = val_loss(predictions, graph)
                    log(f"metrics/{val_loss.name}", value)

        return total_loss_result.loss_value

    def training_step(self, structure: LabelledBatch, _):
        return self._step(structure, "train")

    def validation_step(self, structure: LabelledBatch, _):
        return self._step(structure, "val")

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
                    "monitor": "val/loss/total",
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
                    "monitor": "val/loss/total",
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
        return {"optimizer": opt, "lr_scheduler": config}

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


def wandb_available():
    try:
        import wandb  # noqa: F401
        # TODO: login?

        return True
    except ImportError:
        return False


def create_training_trainer(
    fit_config: FittingOptions,
    val_available: bool = False,
) -> pl.Trainer:
    if val_available:
        default_checkpoint = ModelCheckpoint(
            monitor="val/loss/total",
            filename="best",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )
    else:
        default_checkpoint = ModelCheckpoint(
            filename="best",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        default_checkpoint,
        RichProgressBar(),
    ]
    if fit_config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss/total",
                patience=fit_config.early_stopping_patience,
                mode="min",
                min_delta=1e-6,
            )
        )

    defaults = {
        "accelerator": "auto",
        "max_epochs": 100,
        "enable_model_summary": False,
        "callbacks": callbacks,
    }

    # TODO intelligently override the callbacks
    trainer_kwargs = {**defaults, **fit_config.trainer_kwargs}

    if "logger" not in trainer_kwargs and wandb_available():
        trainer_kwargs["logger"] = pytorch_lightning.loggers.WandbLogger()

    return pl.Trainer(**trainer_kwargs)


# TODO
def create_testing_trainer(): ...
