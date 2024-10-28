from __future__ import annotations

import helpers
from graph_pes.data.io import to_atomic_graphs
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.graphs.operations import to_batch
from graph_pes.training.manual import Loss, train_the_model


@helpers.parameterise_all_models(expected_elements=["Cu"])
def test_integration(model: GraphPESModel):
    if len(list(model.parameters())) == 0:
        # nothing to train
        return

    graphs = to_atomic_graphs(helpers.CU_TEST_STRUCTURES, cutoff=3)

    batch = to_batch(graphs)
    assert "energy" in batch

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
