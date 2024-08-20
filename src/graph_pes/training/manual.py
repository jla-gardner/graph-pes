from __future__ import annotations

from typing import Sequence

from graph_pes.config import FittingOptions
from graph_pes.core import ConservativePESModel
from graph_pes.data.dataset import (
    FittingData,
    LabelledGraphDataset,
    SequenceDataset,
)
from graph_pes.graphs import (
    LabelledGraph,
    keys,
)
from graph_pes.graphs.operations import (
    has_cell,
    number_of_atoms,
    number_of_structures,
    to_batch,
)
from graph_pes.logger import logger
from graph_pes.training.loss import RMSE, Loss, TotalLoss
from graph_pes.training.opt import Adam, Optimizer
from graph_pes.training.ptl import create_trainer, train_with_lightning


# TODO: add logger folder
def train_the_model(
    model: ConservativePESModel,
    train_data: Sequence[LabelledGraph] | LabelledGraphDataset,
    val_data: Sequence[LabelledGraph] | LabelledGraphDataset,
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
        optimizer = Adam(lr=3e-3)

    # create data module
    if not isinstance(train_data, LabelledGraphDataset):
        train_data = SequenceDataset(train_data)
    if not isinstance(val_data, LabelledGraphDataset):
        val_data = SequenceDataset(val_data)

    # sanity check on cpu TODO: move to ptl
    dummy_batch = to_batch(list(train_data.up_to_n(10)))
    try:
        model(dummy_batch)
    except Exception as e:
        raise ValueError("The model does not appear to work") from e

    # process and validate the loss
    logger.debug(f"Available properties: {train_data.labels}")
    total_loss = process_loss(loss, train_data.labels)

    training_on = [component.property_key for component in total_loss.losses]
    for prop in training_on:
        if prop not in train_data.labels:
            raise ValueError(
                f"Can't train on {prop} without the corresponding labels. "
                f"Available labels: {train_data.labels}"
            )
    if not training_on:
        raise ValueError("Found no properties to train on")

    logger.info(f"{total_loss}")

    # sanity checking the data
    expected_shapes = {
        keys.ENERGY: (number_of_structures(dummy_batch),),
        keys.FORCES: (number_of_atoms(dummy_batch), 3),
        keys.STRESS: (number_of_structures(dummy_batch), 3, 3),
    }
    for prop in training_on:
        if dummy_batch[prop].shape != expected_shapes[prop]:
            raise ValueError(
                f"Expected {prop} to have shape {expected_shapes[prop]}, "
                f"but found {dummy_batch[prop].shape}"
            )
    if keys.STRESS in training_on and not has_cell(dummy_batch):
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
    available_properties: list[keys.LabelKey],
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
