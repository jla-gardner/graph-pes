from __future__ import annotations

import pytorch_lightning as pl
from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import to_batch
from graph_pes.config import FittingOptions
from graph_pes.data.datasets import FittingData, SequenceDataset
from graph_pes.training.loss import Loss, TotalLoss
from graph_pes.training.opt import Optimizer
from graph_pes.training.trainer import train_with_lightning

from .. import helpers


@helpers.parameterise_all_models(expected_elements=["Cu"])
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

    batch = to_batch(graphs)
    assert "energy" in batch.properties

    # Create loss and get initial performance
    loss = TotalLoss([Loss("energy")])
    before = loss(model.predict(batch, ["energy"]), batch)

    # Create trainer and train
    train_with_lightning(
        trainer=pl.Trainer(max_epochs=10, accelerator="cpu"),
        model=model,
        data=FittingData(
            train=SequenceDataset(train_graphs),
            valid=SequenceDataset(val_graphs),
        ),
        loss=loss,
        fit_config=FittingOptions(
            pre_fit_model=False,
            loader_kwargs={"batch_size": 4},
            max_n_pre_fit=100,
            early_stopping_patience=None,
        ),
        optimizer=Optimizer("Adam", lr=3e-4),
    )

    after = loss(model.predict(batch, ["energy"]), batch)

    assert after < before, "training did not improve the loss"
