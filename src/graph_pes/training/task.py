from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    number_of_structures,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.loss import RMSE, Loss, PerAtomEnergyLoss, TotalLoss
from graph_pes.training.opt import LRScheduler, Optimizer
from graph_pes.training.util import VALIDATION_LOSS_KEY
from graph_pes.utils.logger import logger
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class PESLearningTask(pl.LightningModule):
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
        self.train_properties: list[PropertyKey] = [
            component.property for component in self.total_loss.losses
        ]

    def forward(self, graphs: AtomicGraph) -> torch.Tensor:
        """Get the energy"""
        return self.model.predict_energy(graphs)

    def on_train_start(self):
        super().on_train_start()
        self.model.train()

    def on_validation_start(self):
        super().on_validation_start()
        self.model.eval()

    def _step(self, graph: AtomicGraph, prefix: Literal["train", "valid"]):
        """Get (and log) the losses for a training/validation step."""

        def log(name: str, value: torch.Tensor | float):
            if isinstance(value, torch.Tensor):
                value = value.item()

            is_valid = prefix == "valid"

            return self.log(
                f"{prefix}/{name}",
                value,
                prog_bar=is_valid and "metric" in name,
                on_step=not is_valid,
                on_epoch=is_valid,
                sync_dist=is_valid,
                batch_size=number_of_structures(graph),
            )

        # generate prediction:
        if prefix == "valid":
            desired_properties = list(graph.properties.keys())
        else:
            desired_properties = self.train_properties

        predictions = self.model.predict(graph, properties=desired_properties)

        # compute the loss and its sub-components
        total_loss_result = self.total_loss(predictions, graph)

        # log
        log("loss/total", total_loss_result.loss_value)
        for name, loss_pair in total_loss_result.components.items():
            log(f"metrics/{name}", loss_pair.loss_value)
            log(f"loss/{name}_weighted", loss_pair.weighted_loss_value)

        # log additional values during validation
        if prefix == "valid":
            val_metrics: list[Loss] = []
            if "energy" in graph.properties:
                val_metrics.append(PerAtomEnergyLoss())
                val_metrics.append(Loss("energy", RMSE()))
            if "forces" in graph.properties:
                val_metrics.append(Loss("forces", RMSE()))
            if "stress" in graph.properties:
                val_metrics.append(Loss("stress", RMSE()))

            for metric in val_metrics:
                if metric.name in total_loss_result.components:
                    # don't double log
                    continue
                value = metric(predictions, graph)
                log(f"metrics/{metric.name}", value)

        return total_loss_result.loss_value

    def training_step(self, structure: AtomicGraph, _):
        return self._step(structure, "train")

    def validation_step(self, structure: AtomicGraph, _):
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
        checkpoint = torch.load(path, weights_only=True)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)
