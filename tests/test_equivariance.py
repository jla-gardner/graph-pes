import helpers
import numpy as np
import pytest
import torch
from ase.build import molecule
from graph_pes.core import ConservativePESModel, get_predictions
from graph_pes.data.io import to_atomic_graph

CUTOFF = 1.0


@helpers.parameterise_all_models(expected_elements=["H", "C"])
def test_equivariance(model: ConservativePESModel):
    methane = molecule("CH4")
    methane.center(vacuum=10)
    og_graph = to_atomic_graph(methane, cutoff=CUTOFF)
    og_predictions = get_predictions(model, og_graph)

    # get a (repeatably) random rotation matrix
    R, _ = np.linalg.qr(np.random.RandomState(42).randn(3, 3))

    # rotate the molecule
    new_methane = methane.copy()
    shift = new_methane.positions.mean(axis=0)
    new_methane.positions = (new_methane.positions - shift).dot(R) + shift
    new_graph = to_atomic_graph(new_methane, cutoff=CUTOFF)
    new_predictions = get_predictions(model, new_graph)

    # now checks:
    # 1. invariance of energy prediction
    torch.testing.assert_close(
        og_predictions["energy"],
        new_predictions["energy"],
        atol=1e-6,
        rtol=1e-6,
    )

    # 2. invariance of force magnitude
    torch.testing.assert_close(
        og_predictions["forces"].norm(dim=-1),
        new_predictions["forces"].norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )

    # 3. equivariance of forces
    _dtype = og_predictions["forces"].dtype
    torch.testing.assert_close(
        og_predictions["forces"] @ torch.tensor(R.T, dtype=_dtype),
        new_predictions["forces"],
        atol=1e-6,
        rtol=1e-6,
    )

    # 4. molecule is symetric: forces should ~0 on the central C,
    #    and of equal magnitude on the H atoms
    force_norms = new_predictions["forces"].norm(dim=-1)
    c_force = force_norms[new_graph["atomic_numbers"] == 6]
    assert c_force.item() == pytest.approx(0.0)

    h_forces = force_norms[new_graph["atomic_numbers"] == 1]
    assert h_forces.min().item() == pytest.approx(h_forces.max().item())
