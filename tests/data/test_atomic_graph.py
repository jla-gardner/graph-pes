from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from graph_pes.data import (
    AtomicGraph,
    convert_to_atomic_graph,
    convert_to_atomic_graphs,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
)

ISOLATED_ATOM = Atoms("H", positions=[(0, 0, 0)], pbc=False)
PERIODIC_ATOM = Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(1, 1, 1))
RANDOM_STRUCTURE = Atoms(
    "H8",
    positions=np.random.RandomState(42).rand(8, 3),
    pbc=True,
    cell=np.eye(3),
)
STRUCTURES = [ISOLATED_ATOM, PERIODIC_ATOM, RANDOM_STRUCTURE]
GRAPHS = convert_to_atomic_graphs(STRUCTURES, cutoff=1.0)


@pytest.mark.parametrize("structure, graph", zip(STRUCTURES, GRAPHS))
def test_general(structure: Atoms, graph: AtomicGraph):
    assert number_of_atoms(graph) == len(structure)

    n_edges = number_of_edges(graph)
    assert n_edges == graph["neighbour_index"].shape[1]
    assert n_edges == neighbour_vectors(graph).shape[0]
    assert n_edges == neighbour_distances(graph).shape[0]


def test_iso_atom():
    graph = convert_to_atomic_graph(ISOLATED_ATOM, cutoff=1.0)
    assert number_of_atoms(graph) == 1
    assert number_of_edges(graph) == 0


def test_periodic_atom():
    graph = convert_to_atomic_graph(PERIODIC_ATOM, cutoff=1.1)
    assert number_of_atoms(graph) == 1

    # 6 neighbours: up, down, left, right, front, back
    assert number_of_edges(graph) == 6


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 1.5])
def test_random_structure(cutoff: int):
    graph = convert_to_atomic_graph(RANDOM_STRUCTURE, cutoff=cutoff)
    assert number_of_atoms(graph) == 8

    assert neighbour_distances(graph).max() <= cutoff


# def test_warning_on_position():
#     # check that a warning is raised if the user tries to access the positions
#     # directly for a structure with a unit cell
#     with pytest.warns(UserWarning):
#         _ = GRAPHS[1].positions


def test_get_labels():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    graph = convert_to_atomic_graph(atoms, cutoff=1.0)

    forces = graph["forces"]
    assert forces.shape == (2, 3)

    energy = graph["energy"]
    assert energy.item() == -1.0

    with pytest.raises(KeyError):
        graph["missing"]
