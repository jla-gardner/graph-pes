from __future__ import annotations

import pytest
import torch
from ase import Atoms
from graph_pes import get_predictions
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs import keys
from graph_pes.graphs.operations import number_of_edges, to_batch
from graph_pes.models.pairwise import LennardJones

no_pbc = to_atomic_graph(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    cutoff=1.5,
)
pbc = to_atomic_graph(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=True, cell=(2, 2, 2)),
    cutoff=1.5,
)


def test_predictions():
    expected_shapes = {
        keys.ENERGY: (),
        keys.FORCES: (2, 3),
        keys.STRESS: (3, 3),
    }

    model = LennardJones()

    # by default, get_predictions returns energy and forces on
    # structures with no cell:
    predictions = get_predictions(model, no_pbc)
    assert set(predictions.keys()) == {"energy", "forces"}

    for key in keys.ENERGY, keys.FORCES:
        assert predictions[key].shape == expected_shapes[key]

    # if we ask for stress, we get an error:
    with pytest.raises(ValueError):
        get_predictions(model, no_pbc, property="stress")

    # with pbc structures, we should get all three predictions
    predictions = get_predictions(model, pbc)
    assert set(predictions.keys()) == {"energy", "forces", "stress"}

    for key in keys.ENERGY, keys.FORCES, keys.STRESS:
        assert predictions[key].shape == expected_shapes[key]


def test_batched_prediction():
    batch = to_batch([pbc, pbc])

    expected_shapes = {
        keys.ENERGY: (2,),  # two structures
        keys.FORCES: (4, 3),  # four atoms
        keys.STRESS: (2, 3, 3),  # two structures
    }

    predictions = get_predictions(LennardJones(), batch)

    for key in keys.ENERGY, keys.FORCES, keys.STRESS:
        assert predictions[key].shape == expected_shapes[key]


def test_isolated_atom():
    atom = Atoms("H", positions=[(0, 0, 0)], pbc=False)
    graph = to_atomic_graph(atom, cutoff=1.5)
    assert number_of_edges(graph) == 0

    predictions = get_predictions(LennardJones(), graph)
    assert torch.allclose(predictions["forces"], torch.zeros(1, 3))


def test_general_api():
    with pytest.raises(ValueError):
        get_predictions(
            LennardJones(),
            no_pbc,
            property="energy",
            properties=["forces"],
        )  # type: ignore
