from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from ase import Atoms
from graph_pes.data.io import to_atomic_graph, to_atomic_graphs
from graph_pes.graphs import AtomicGraph
from graph_pes.graphs.operations import (
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
    sum_over_neighbours,
)

ISOLATED_ATOM = Atoms("H", positions=[(0, 0, 0)], pbc=False)
PERIODIC_ATOM = Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(1, 1, 1))
DIMER = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
RANDOM_STRUCTURE = Atoms(
    "H8",
    positions=np.random.RandomState(42).rand(8, 3),
    pbc=True,
    cell=np.eye(3),
)
STRUCTURES = [ISOLATED_ATOM, PERIODIC_ATOM, DIMER, RANDOM_STRUCTURE]
GRAPHS = to_atomic_graphs(STRUCTURES, cutoff=1.0)


@pytest.mark.parametrize("structure, graph", zip(STRUCTURES, GRAPHS))
def test_general(structure: Atoms, graph: AtomicGraph):
    assert number_of_atoms(graph) == len(structure)

    n_edges = number_of_edges(graph)
    assert n_edges == graph["neighbour_index"].shape[1]
    assert n_edges == neighbour_vectors(graph).shape[0]
    assert n_edges == neighbour_distances(graph).shape[0]


def test_iso_atom():
    graph = to_atomic_graph(ISOLATED_ATOM, cutoff=1.0)
    assert number_of_atoms(graph) == 1
    assert number_of_edges(graph) == 0


def test_periodic_atom():
    graph = to_atomic_graph(PERIODIC_ATOM, cutoff=1.1)
    assert number_of_atoms(graph) == 1

    # 6 neighbours: up, down, left, right, front, back
    assert number_of_edges(graph) == 6


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 1.5])
def test_random_structure(cutoff: int):
    graph = to_atomic_graph(RANDOM_STRUCTURE, cutoff=cutoff)
    assert number_of_atoms(graph) == 8

    assert neighbour_distances(graph).max() <= cutoff


def test_get_labels():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    graph = to_atomic_graph(atoms, cutoff=1.0)

    forces = graph["forces"]
    assert forces.shape == (2, 3)

    energy = graph["energy"]
    assert energy.item() == -1.0

    with pytest.raises(KeyError):
        graph["missing"]  # type: ignore


# Within each of these structures, each atom has the same number of neighbours,
# making it easy to test that the sum over neighbours is correct
@pytest.mark.parametrize("structure", [ISOLATED_ATOM, DIMER, PERIODIC_ATOM])
def test_sum_over_neighbours(structure):
    graph = to_atomic_graph(structure, cutoff=1.1)
    N = number_of_atoms(graph)
    E = number_of_edges(graph)
    n_neighbours = (graph["neighbour_index"][0] == 0).sum()

    # check that summing is
    # (a) shape preserving
    # (b) compatible with torch.jit.script

    torchscript_sum: Callable[[torch.Tensor, AtomicGraph], torch.Tensor] = (
        torch.jit.script(sum_over_neighbours)
    )
    for shape in [(E,), (E, 2), (E, 2, 3), (E, 2, 2, 2)]:
        for summing_fn in (
            sum_over_neighbours,
            torchscript_sum,
        ):
            edge_property = torch.ones(shape)
            result = summing_fn(edge_property, graph)
            assert result.shape == (N, *shape[1:])
            assert (result == n_neighbours).all()
