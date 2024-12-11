import torch
from ase.build import molecule

from graph_pes import AtomicGraph
from graph_pes.models import NequIP


def test_direct_forces():
    model = NequIP(
        direct_force_predictions=True,
        elements=["C", "H"],
        features=dict(channels=4, l_max=1, use_odd_parity=True),
    )
    graph = AtomicGraph.from_ase(molecule("CH4"))

    # test that the model outputs forces directly...
    preds = model(graph)
    assert "forces" in preds
    assert preds["forces"].shape == (5, 3)

    # model outputs forces indirectly
    all_preds = model.get_all_PES_predictions(graph)
    assert "forces" in all_preds
    assert all_preds["forces"].shape == (5, 3)
    assert torch.allclose(preds["forces"], all_preds["forces"])

    # check equivariance: all force mags for the H atoms should be the same
    force_mags = torch.linalg.norm(preds["forces"], dim=1)
    idx = graph.Z == 1
    assert torch.allclose(force_mags[idx], force_mags[idx][0])
    # and that these forces aren't all 0
    assert not torch.allclose(
        force_mags[idx], torch.zeros_like(force_mags[idx])
    )
