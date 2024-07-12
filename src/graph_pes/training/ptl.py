from __future__ import annotations

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
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


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
    log_model_info(model)

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


def create_trainer(
    early_stopping_patience: int | None = None,
    kwarg_overloads: dict | None = None,
    val_available: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
) -> pl.Trainer:
    if val_available:
        default_checkpoint = ModelCheckpoint(
            dirpath=output_dir,
            monitor="val/loss/total",
            filename="best",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )
    else:
        default_checkpoint = ModelCheckpoint(
            dirpath=output_dir,
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
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss/total",
                patience=early_stopping_patience,
                mode="min",
                min_delta=1e-6,
            )
        )

    defaults = {
        "enable_model_summary": False,
        "callbacks": callbacks,
    }

    # TODO intelligently override the callbacks
    trainer_kwargs = {**defaults, **(kwarg_overloads or {})}
    return pl.Trainer(**trainer_kwargs, logger=logger)
