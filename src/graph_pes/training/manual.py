from __future__ import annotations

from typing import Sequence

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    has_cell,
    number_of_atoms,
    number_of_structures,
    to_batch,
)
from graph_pes.config import FittingOptions
from graph_pes.data.datasets import (
    FittingData,
    GraphDataset,
    SequenceDataset,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.training.loss import RMSE, Loss, TotalLoss
from graph_pes.training.opt import Optimizer
from graph_pes.training.ptl import create_trainer, train_with_lightning
from graph_pes.utils.logger import logger


# TODO: add logger folder
def train_the_model(
    model: GraphPESModel,
    train_data: Sequence[AtomicGraph] | GraphDataset,
    val_data: Sequence[AtomicGraph] | GraphDataset,
    optimizer: Optimizer | None = None,  # TODO change to factory
    loss: TotalLoss | Loss | None = None,
    *,
    config_to_log: dict | None = None,
    batch_size: int = 32,
    pre_fit_model: bool = True,
    max_n_pre_fit: int | None = 5_000,
    # pytorch lightning
    trainer_options: dict | None = None,
):
    """
    Train a model on the given data.
    TODO: fill this out
    Simple.
    """

    # defaults:
    if trainer_options is None:
        trainer_options = {}
    if config_to_log is None:
        config_to_log = {}
    if optimizer is None:
        optimizer = Optimizer("Adam", lr=3e-3)

    # create data module
    if not isinstance(train_data, GraphDataset):
        train_data = SequenceDataset(train_data)
    if not isinstance(val_data, GraphDataset):
        val_data = SequenceDataset(val_data)

    # sanity check on cpu TODO: move to ptl
    dummy_batch = to_batch(list(train_data.up_to_n(10)))
    try:
        model.get_all_PES_predictions(dummy_batch)
    except Exception as e:
        raise ValueError("The model does not appear to work") from e

    # process and validate the loss
    logger.debug(f"Available properties: {train_data.properties}")
    total_loss = process_loss(loss, train_data.properties)

    training_on: list[PropertyKey] = [
        component.property for component in total_loss.losses
    ]
    for prop in training_on:
        if prop not in train_data.properties:
            raise ValueError(
                f"Can't train on {prop} without the corresponding labels. "
                f"Available labels: {train_data.properties}"
            )
    if not training_on:
        raise ValueError("Found no properties to train on")

    logger.info(f"{total_loss}")

    # sanity checking the data
    expected_shapes = {
        "energy": (number_of_structures(dummy_batch),),
        "forces": (number_of_atoms(dummy_batch), 3),
        "stress": (number_of_structures(dummy_batch), 3, 3),
    }
    for prop in training_on:
        if dummy_batch.properties[prop].shape != expected_shapes[prop]:
            raise ValueError(
                f"Expected {prop} to have shape {expected_shapes[prop]}, "
                f"but found {dummy_batch.properties[prop].shape}"
            )
    if "stress" in training_on and not has_cell(dummy_batch):
        raise ValueError("Can't train on stress without cell information.")

    # train the model
    options = FittingOptions(
        pre_fit_model=pre_fit_model,
        max_n_pre_fit=max_n_pre_fit,
        early_stopping_patience=100,
        trainer_kwargs=trainer_options,
        loader_kwargs=dict(batch_size=batch_size),
    )
    trainer = create_trainer(
        early_stopping_patience=100,
        valid_available=True,
        kwarg_overloads=trainer_options,
    )
    train_with_lightning(
        trainer,
        model,
        data=FittingData(train_data, val_data),
        loss=total_loss,
        fit_config=options,
        optimizer=optimizer,
        scheduler=None,
    )


def process_loss(
    loss: TotalLoss | Loss | None,
    available_properties: list[PropertyKey],
) -> TotalLoss:
    if isinstance(loss, TotalLoss):
        return loss

    elif isinstance(loss, Loss):
        return TotalLoss([loss], [1.0])

    if loss is not None:
        raise ValueError(
            "Invalid loss: must be a TotalLoss, a Loss, or None. "
            f"Got {type(loss)}"
        )

    return TotalLoss(
        losses=[Loss(key, metric=RMSE()) for key in available_properties],
        weights=None,
    )
