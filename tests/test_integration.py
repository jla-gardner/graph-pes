from __future__ import annotations

import pytest
from ase import Atoms
from ase.io import read
from graph_pes import get_predictions
from graph_pes.data.io import to_atomic_graphs
from graph_pes.graphs.operations import to_batch
from graph_pes.models import LennardJones, Morse, PaiNN, SchNet
from graph_pes.models.e3nn.nequip import NequIP
from graph_pes.models.tensornet import TensorNet
from graph_pes.training.manual import Loss, train_the_model

models = [
    LennardJones(),
    Morse(),
    SchNet(),
    PaiNN(),
    TensorNet(),
    NequIP(n_elements=1),
]


@pytest.mark.parametrize(
    "model",
    models,
    ids=[model.__class__.__name__ for model in models],
)
def test_integration(model):
    structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
    graphs = to_atomic_graphs(structures, cutoff=3)

    batch = to_batch(graphs)
    assert "energy" in batch

    model.pre_fit(graphs[:8])

    loss = Loss("energy")
    before = loss(get_predictions(model, batch), batch)

    train_the_model(
        model,
        train_data=graphs[:8],
        val_data=graphs[8:],
        loss=loss,
        trainer_options=dict(
            max_epochs=2,
            accelerator="cpu",
            callbacks=[],
            logger=None,
        ),
        pre_fit_model=False,
    )

    after = loss(get_predictions(model, batch), batch)

    assert after < before, "training did not improve the loss"
