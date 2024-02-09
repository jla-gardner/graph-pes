from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

from .core import GraphPESModel
from .data import AtomicGraph
from .data.batching import AtomicDataLoader, AtomicGraphBatch
from .loss import RMSE, Loss, WeightedLoss
from .transform import PerAtomScale, PerAtomStandardScaler, Scale
from .util import ALL_PROPERTIES, Property, PropertyKey

T = TypeVar("T", bound=GraphPESModel)


def train_model(
    model: T,
    train_data: list[AtomicGraph],
    val_data: list[AtomicGraph] | None = None,
    optimizer: Callable[[], torch.optim.Optimizer | OptimizerLRSchedulerConfig]
    | None = None,
    loss: WeightedLoss | Loss | None = None,
    *,
    batch_size: int = 32,
    pre_fit_model: bool = True,
    # pytorch lightning
    **trainer_kwargs,
) -> T:
    # sanity check, but also ensures things like per-atom parameters
    # are registered
    try:
        for graph in train_data:
            model(graph)
    except Exception as e:
        raise ValueError("The model does not appear to work") from e

    # TODO check using a strict flag that all the data have the same keys
    train_batch = AtomicGraphBatch.from_graphs(train_data)

    # process and validate the loss
    total_loss = process_loss(loss, train_data[0])
    training_on = [component.property_key for component in total_loss.losses]
    for prop in training_on:
        if prop not in get_existing_properties(train_data[0]):
            raise ValueError(
                f"Can't train on {prop} without the corresponding data"
            )
    if not training_on:
        raise ValueError("No properties to train on")

    expected_shapes = {
        Property.ENERGY: (train_batch.n_structures,),
        Property.FORCES: (train_batch.n_atoms, 3),
        Property.STRESS: (train_batch.n_structures, 3, 3),
    }
    for prop in training_on:
        if train_batch[prop].shape != expected_shapes[prop]:
            raise ValueError(
                f"Expected {prop} to have shape {expected_shapes[prop]}, "
                f"but found {train_batch[prop].shape}"
            )
    if Property.STRESS in training_on and not train_batch.has_cell:
        raise ValueError("Can't train on stress without cell information.")

    # create the data loaders
    train_loader = AtomicDataLoader(train_data, batch_size, shuffle=True)
    val_loader = (
        AtomicDataLoader(val_data, batch_size, shuffle=False)
        if val_data is not None
        else None
    )

    # deal with fitting transforms
    if pre_fit_model and Property.ENERGY in training_on:
        model.pre_fit(train_batch)
    total_loss.fit_transform(train_batch)

    # deal with the optimizer
    if optimizer is None:
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    else:
        opt = optimizer()

    # create the task (a pytorch lightning module)
    task = LearnThePES(model, opt, total_loss)

    # create the trainer
    kwargs = default_trainer_kwargs()
    kwargs.update(trainer_kwargs)
    trainer = pl.Trainer(**kwargs)

    # log info
    params = sum(p.numel() for p in model.parameters())
    device = trainer.accelerator.__class__.__name__.replace("Accelerator", "")
    print(f"Training on : {training_on}")
    print(f"# of params : {params}")
    print(f"Device      : {device}")
    print()

    # train
    trainer.fit(task, train_loader, val_loader)

    # load the best weights
    task.load_best_weights(model, trainer)

    return model


def get_existing_properties(graph: AtomicGraph) -> list[PropertyKey]:
    return [p for p in ALL_PROPERTIES if p in graph]


class LearnThePES(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        optimizer: torch.optim.Optimizer | OptimizerLRSchedulerConfig,
        total_loss: WeightedLoss,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.total_loss = total_loss
        self.properties: list[PropertyKey] = [
            component.property_key for component in total_loss.losses
        ]

    def forward(self, graphs: AtomicGraphBatch):
        return self.model(graphs)

    def _step(self, graph: AtomicGraphBatch, prefix: str):
        """
        Get (and log) the losses for a training/validation step.
        """

        def log(name, value, verbose=True):
            return self.log(
                f"{prefix}_{name}",
                value,
                prog_bar=verbose and prefix == "val",
                on_step=False,
                on_epoch=True,
                batch_size=graph.n_structures,
            )

        # generate prediction:
        predictions = self.model.predict(
            graph, properties=self.properties, training=True
        )

        # compute the losses
        total_loss = torch.scalar_tensor(0.0, device=self.device)

        for loss, weight in zip(
            self.total_loss.losses, self.total_loss.weights
        ):
            value = loss(predictions, graph)
            # log the unweighted components of the loss
            log(f"{loss.property_key}_{loss.name}", value)
            # but weight them when computing the total loss
            total_loss = total_loss + weight * value

            # log the raw values only during validation
            if prefix == "val":
                raw_value = loss.raw(predictions, graph)
                log(f"{loss.property_key}_raw_{loss.name}", raw_value)

        log("total_loss", total_loss)
        return total_loss

    def training_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "train")

    def validation_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "val")

    def configure_optimizers(self):
        return self.optimizer

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
        checkpoint = torch.load(path)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)


def process_loss(
    loss: WeightedLoss | Loss | None, graph: AtomicGraph
) -> WeightedLoss:
    if isinstance(loss, WeightedLoss):
        return loss
    elif isinstance(loss, Loss):
        return WeightedLoss([loss], [1.0])

    default_transforms = {
        Property.ENERGY: PerAtomStandardScaler(),  # TODO is this right?
        Property.FORCES: PerAtomScale(),
        Property.STRESS: Scale(),
    }
    default_weights = {
        Property.ENERGY: 1.0,
        Property.FORCES: 1.0,
        Property.STRESS: 1.0,
    }

    available_properties = get_existing_properties(graph)

    return WeightedLoss(
        [
            Loss(
                key,
                metric=RMSE(),
                transform=default_transforms[key],
            )
            for key in available_properties
        ],
        [default_weights[key] for key in available_properties],
    )


def default_trainer_kwargs() -> dict:
    return {
        "accelerator": "auto",
        "max_epochs": 100,
        "enable_model_summary": False,
        "callbacks": [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val_total_loss",
                filename="{epoch}-{val_total_loss:.4f}",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
            ),
        ],
    }


def device_info_filter(record):
    return (
        "PU available: " not in record.getMessage()
        and "LOCAL_RANK" not in record.getMessage()
    )


# disable verbose logging from pytorch lightning
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    device_info_filter
)
