import torch
from ase.build import molecule
from graph_pes.core import NotNeeded
from graph_pes.data.io import to_atomic_graph
from graph_pes.models import NequIP


def test_direct_forces():
    model = NequIP(
        direct_force_predictions=True,
        elements=["C", "H"],
        n_channels=4,
        l_max=1,
    )
    graph = to_atomic_graph(molecule("CH4"), 3.7)

    # test that the model outputs forces directly...
    preds = model(graph)
    assert "forces" in preds
    assert preds["forces"].shape == (5, 3)

    # no InferForce output inferrer
    assert isinstance(model.output_inferrers["forces"], NotNeeded)

    # model outputs forces indirectly
    all_preds = model.get_all_PES_predictions(graph)
    assert "forces" in preds
    assert preds["forces"].shape == (5, 3)
    assert torch.allclose(preds["forces"], all_preds["forces"])

    # check equivariance: all force mags for the H atoms should be the same
    force_mags = torch.linalg.norm(preds["forces"], dim=1)
    idx = graph["atomic_numbers"] == 1
    assert torch.allclose(force_mags[idx], force_mags[idx][0])
    # and that these forces aren't all 0
    assert not torch.allclose(
        force_mags[idx], torch.zeros_like(force_mags[idx])
    )
