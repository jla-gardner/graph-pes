import numpy as np
import pytest
import torch
from ase.build import molecule

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.models import Orb

from .. import helpers

CUTOFF = 1.2  # bond length of methane is ~1.09 Å


@helpers.parameterise_all_models(expected_elements=["H", "C"], cutoff=CUTOFF)
def test_equivariance(model: GraphPESModel):
    if isinstance(model, Orb):
        # orb is not equivariant to rotation
        return

    methane = molecule("CH4")
    og_graph = AtomicGraph.from_ase(methane, cutoff=CUTOFF)
    og_predictions = model.get_all_PES_predictions(og_graph)

    # get a (repeatably) random rotation matrix
    R, _ = np.linalg.qr(np.random.RandomState(42).randn(3, 3))

    # rotate the molecule
    new_methane = methane.copy()
    shift = new_methane.positions[0]
    new_methane.positions = (new_methane.positions - shift).dot(R) + shift
    new_graph = AtomicGraph.from_ase(new_methane, cutoff=CUTOFF)
    new_predictions = model.get_all_PES_predictions(new_graph)

    # pre-checks:
    np.testing.assert_allclose(
        np.linalg.norm(new_methane.positions - shift, axis=1),
        np.linalg.norm(methane.positions - shift, axis=1),
    )

    # now check:
    # 1. invariance of energy prediction
    torch.testing.assert_close(
        og_predictions["energy"],
        new_predictions["energy"],
        atol=2e-5,
        rtol=1e-3,
    )

    # 2. invariance of force magnitude
    torch.testing.assert_close(
        og_predictions["forces"].norm(dim=-1),
        new_predictions["forces"].norm(dim=-1),
        atol=2e-5,
        rtol=1e-3,
    )

    # 3. equivariance of forces
    _dtype = og_predictions["forces"].dtype
    torch.testing.assert_close(
        og_predictions["forces"] @ torch.tensor(R.T, dtype=_dtype),
        new_predictions["forces"],
        # auto-grad is not perfect, and we lose
        # precision, particularly with larger models
        # and default float32 dtype
        atol=3e-3,
        # some of the og predictions are 0: a large
        # relative error is not a problem here
        rtol=10,
    )

    # 4. molecule is symmetric: forces should ~0 on the central C,
    #    and of equal magnitude on the H atoms, and H atom
    #    local energies should be the same
    force_norms = new_predictions["forces"].norm(dim=-1)
    c_force = force_norms[new_graph.Z == 6]
    assert c_force.item() == pytest.approx(0.0, abs=3e-4)

    h_forces = force_norms[new_graph.Z == 1]
    assert h_forces.min().item() == pytest.approx(
        h_forces.max().item(), abs=1e-4
    )

    h_local_energies = new_predictions["local_energies"][new_graph.Z == 1]
    _min, _max = h_local_energies.min().item(), h_local_energies.max().item()
    assert _min == pytest.approx(_max, abs=1e-6)
