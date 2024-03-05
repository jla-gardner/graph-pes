from __future__ import annotations

from ase import Atoms
from ase.io import read
from graph_pes.core import get_predictions
from graph_pes.data import (
    convert_to_atomic_graph,
    convert_to_atomic_graphs,
    is_periodic,
    number_of_atoms,
    number_of_edges,
)
from graph_pes.models.pairwise import LennardJones

structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
graphs = convert_to_atomic_graphs(structures, cutoff=3)


def test_model():
    model = LennardJones()
    predictions = get_predictions(model, graphs)
    assert "energy" in predictions
    assert "forces" in predictions
    assert "stress" in predictions and is_periodic(graphs[0])
    assert predictions["energy"].shape == (len(graphs),)
    assert predictions["stress"].shape == (len(graphs), 3, 3)


def test_isolated_atom():
    atom = Atoms("He", positions=[[0, 0, 0]])
    graph = convert_to_atomic_graph(atom, cutoff=3)
    assert number_of_atoms(graph) == 1 and number_of_edges(graph) == 0

    model = LennardJones()
    assert model(graph) == 0
