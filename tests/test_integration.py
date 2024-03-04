from __future__ import annotations

from ase import Atoms
from ase.io import read
from graph_pes.data import batch_graphs, convert_to_atomic_graphs
from graph_pes.models.pairwise import LennardJones
from graph_pes.training import Loss, train_model


def test_integration():
    structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
    graphs = convert_to_atomic_graphs(structures, cutoff=3)
    batch = batch_graphs(graphs)

    model = LennardJones()

    loss = Loss("energy")
    before = loss(model.predict(batch), batch)

    train_model(
        model,
        train_data=graphs[:8],
        val_data=graphs[8:],
        loss=loss,
        max_epochs=2,
        accelerator="cpu",
        callbacks=[],
    )

    after = loss(model.predict(batch), batch)

    assert after < before, "training did not improve the loss"
