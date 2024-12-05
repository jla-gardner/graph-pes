from __future__ import annotations

import ase.build
import numpy as np
import pytest
import torch
from ase import Atoms
from graph_pes import AtomicGraph
from graph_pes.atomic_graph import get_cell_volume, number_of_edges, to_batch
from graph_pes.models.pairwise import LennardJones

no_pbc = AtomicGraph.from_ase(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    cutoff=1.5,
)
pbc = AtomicGraph.from_ase(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=True, cell=(2, 2, 2)),
    cutoff=1.5,
)


def test_predictions():
    expected_shapes = {
        "energy": (),
        "forces": (2, 3),
        "stress": (3, 3),
        "virial": (3, 3),
        "local_energies": (2,),
    }

    model = LennardJones()

    # no stress should be predicted for non-periodic systems
    predictions = model.get_all_PES_predictions(no_pbc)
    assert set(predictions.keys()) == {"energy", "forces", "local_energies"}

    for key in "energy", "forces", "local_energies":
        assert predictions[key].shape == expected_shapes[key]

    # if we ask for stress, we get an error:
    with pytest.raises(ValueError):
        model.predict(no_pbc, properties=["stress"])

    # with pbc structures, we should get all predictions
    predictions = model.get_all_PES_predictions(pbc)
    assert set(predictions.keys()) == {
        "energy",
        "forces",
        "stress",
        "virial",
        "local_energies",
    }

    for key in "energy", "forces", "stress", "virial", "local_energies":
        assert predictions[key].shape == expected_shapes[key]


def test_batched_prediction():
    batch = to_batch([pbc, pbc])

    expected_shapes = {
        "energy": (2,),  # two structures
        "forces": (4, 3),  # four atoms
        "stress": (2, 3, 3),  # two structures
        "virial": (2, 3, 3),  # two structures
    }

    predictions = LennardJones().get_all_PES_predictions(batch)

    for key in "energy", "forces", "stress", "virial":
        assert predictions[key].shape == expected_shapes[key]


def test_isolated_atom():
    atom = Atoms("H", positions=[(0, 0, 0)], pbc=False)
    graph = AtomicGraph.from_ase(atom, cutoff=1.5)
    assert number_of_edges(graph) == 0

    predictions = LennardJones().get_all_PES_predictions(graph)
    assert torch.allclose(predictions["forces"], torch.zeros(1, 3))


def test_stress_and_virial():
    model = LennardJones(cutoff=5.0)

    # get stress and virial predictions
    structure = ase.build.bulk("Si", "diamond", a=5.43)
    graph = AtomicGraph.from_ase(structure, cutoff=5.0)

    s = model.predict_stress(graph)
    v = model.predict_virial(graph)
    volume = get_cell_volume(graph)
    torch.testing.assert_close(s, -v / volume)

    # ensure correct scaling
    np.product = np.prod  # fix ase for new versions of numpy
    structure2 = structure.copy().repeat((2, 2, 2))
    graph2 = AtomicGraph.from_ase(structure2, cutoff=5.0)
    s2 = model.predict_stress(graph2)
    v2 = model.predict_virial(graph2)

    # use diagonal elements of stress and virial tensor to avoid
    # issues with v low values on off diagonals due to numerical error
    def get_diagonal(tensor):
        return torch.diagonal(tensor, dim1=-2, dim2=-1)

    # stress is an intensive property, so should remain the same
    # under a repeated unit cell
    torch.testing.assert_close(get_diagonal(s), get_diagonal(s2))

    # virial is an extensive property, so should scale with volume
    # which in this case is a factor of 8 larger
    torch.testing.assert_close(8 * get_diagonal(v), get_diagonal(v2))
