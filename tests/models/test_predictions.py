from __future__ import annotations

import pytest
import torch
from ase import Atoms
from graph_pes import AtomicGraph
from graph_pes.atomic_graph import number_of_edges, to_batch
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
        "local_energies",
    }

    for key in "energy", "forces", "stress", "local_energies":
        assert predictions[key].shape == expected_shapes[key]


def test_batched_prediction():
    batch = to_batch([pbc, pbc])

    expected_shapes = {
        "energy": (2,),  # two structures
        "forces": (4, 3),  # four atoms
        "stress": (2, 3, 3),  # two structures
    }

    predictions = LennardJones().get_all_PES_predictions(batch)

    for key in "energy", "forces", "stress":
        assert predictions[key].shape == expected_shapes[key]


def test_isolated_atom():
    atom = Atoms("H", positions=[(0, 0, 0)], pbc=False)
    graph = AtomicGraph.from_ase(atom, cutoff=1.5)
    assert number_of_edges(graph) == 0

    predictions = LennardJones().get_all_PES_predictions(graph)
    assert torch.allclose(predictions["forces"], torch.zeros(1, 3))
