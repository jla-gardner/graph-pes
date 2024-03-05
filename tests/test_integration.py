from __future__ import annotations

import pytest
from ase import Atoms
from ase.io import read
from graph_pes import get_predictions
from graph_pes.data import batch_graphs, convert_to_atomic_graphs
from graph_pes.models.zoo import LennardJones, Morse
from graph_pes.training import Loss, train_model

models = [
    LennardJones(),
    Morse(),
]


@pytest.mark.parametrize(
    "model",
    models,
    ids=[model.__class__.__name__ for model in models],
)
def test_integration(model):
    structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
    graphs = convert_to_atomic_graphs(structures, cutoff=3)
    batch = batch_graphs(graphs)

    loss = Loss("energy")
    before = loss(get_predictions(model, batch), batch)

    train_model(
        model,
        train_data=graphs[:8],
        val_data=graphs[8:],
        loss=loss,
        max_epochs=2,
        accelerator="cpu",
        callbacks=[],
    )

    after = loss(get_predictions(model, batch), batch)

    assert after < before, "training did not improve the loss"
