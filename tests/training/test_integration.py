from __future__ import annotations

import pytorch_lightning as pl

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import to_batch
from graph_pes.config.training import FittingOptions
from graph_pes.data.datasets import DatasetCollection, GraphDataset
from graph_pes.training.callbacks import (
    EarlyStoppingWithLogging,
    LoggedProgressBar,
)
from graph_pes.training.loss import PerAtomEnergyLoss, TotalLoss
from graph_pes.training.opt import Optimizer
from graph_pes.training.tasks import train_with_lightning
from graph_pes.training.utils import VALIDATION_LOSS_KEY

from .. import helpers


@helpers.parameterise_all_models(expected_elements=["Cu"], cutoff=3)
def test_integration(model: GraphPESModel):
    if len(list(model.parameters())) == 0:
        # nothing to train
        return

    graphs = [
        AtomicGraph.from_ase(atoms, cutoff=3)
        for atoms in helpers.CU_TEST_STRUCTURES
    ]

    # Split data into train/val sets
    train_graphs = graphs[:8]
    val_graphs = graphs[8:]

    # pre-fit before measuring performance to ensure that
    # training improves the model
    model.pre_fit_all_components(train_graphs)

    train_batch = to_batch(train_graphs)
    assert "energy" in train_batch.properties

    loss = TotalLoss([PerAtomEnergyLoss()])

    def get_train_loss():
        return loss(
            model, train_batch, model.predict(train_batch, ["energy"])
        ).loss_value.item()

    before = get_train_loss()

    # Create trainer and train
    train_with_lightning(
        trainer=pl.Trainer(
            max_epochs=10,
            accelerator="cpu",
            callbacks=[
                LoggedProgressBar(),
                EarlyStoppingWithLogging(
                    monitor=VALIDATION_LOSS_KEY, patience=10
                ),
            ],
        ),
        model=model,
        data=DatasetCollection(
            train=GraphDataset(train_graphs),
            valid=GraphDataset(val_graphs),
        ),
        loss=loss,
        fit_config=FittingOptions(
            pre_fit_model=False,
            loader_kwargs={"batch_size": 8},
            max_n_pre_fit=100,
            early_stopping=None,
            early_stopping_patience=None,
            auto_fit_reference_energies=False,
        ),
        optimizer=Optimizer("Adam", lr=3e-4),
    )

    after = get_train_loss()

    assert after < before, "training did not improve the loss"
