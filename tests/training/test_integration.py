from __future__ import annotations

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import to_batch
from graph_pes.training.manual import Loss, train_the_model

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

    batch = to_batch(graphs)
    assert "energy" in batch.properties

    model.pre_fit_all_components(graphs[:8])

    loss = Loss("energy")
    before = loss(model.predict(batch, ["energy"]), batch)

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

    after = loss(model.predict(batch, ["energy"]), batch)

    assert after < before, "training did not improve the loss"
