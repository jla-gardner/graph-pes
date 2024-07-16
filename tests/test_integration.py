from __future__ import annotations

import helpers
import pytest
from graph_pes import get_predictions
from graph_pes.data.io import to_atomic_graphs
from graph_pes.graphs.operations import to_batch
from graph_pes.models import (
    LennardJones,
    Morse,
    PaiNN,
    SchNet,
    ZEmbeddingNequIP,
)
from graph_pes.models.tensornet import TensorNet
from graph_pes.training.manual import Loss, train_the_model

models = [
    LennardJones(),
    Morse(),
    SchNet(),
    PaiNN(),
    TensorNet(),
    ZEmbeddingNequIP(),
]


@pytest.mark.parametrize(
    "model",
    models,
    ids=[model.__class__.__name__ for model in models],
)
def test_integration(model):
    graphs = to_atomic_graphs(helpers.CU_TEST_STRUCTURES, cutoff=3)

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
        ),
        pre_fit_model=False,
    )

    after = loss(get_predictions(model, batch), batch)

    assert after < before, "training did not improve the loss"
