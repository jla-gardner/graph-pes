from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from ase import Atoms

from graph_pes.atomic_graph import (
    AtomicGraph,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
    number_of_neighbours,
    sum_over_central_atom_index,
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
CHARGED_STRUCTURE = Atoms(
    "H",
    positions=[(0, 0, 0)],
    pbc=False,
    info={
        "total_charge": 1,
    },
)
STRUCTURES = [
    ISOLATED_ATOM,
    PERIODIC_ATOM,
    DIMER,
    RANDOM_STRUCTURE,
    CHARGED_STRUCTURE,
]
GRAPHS = [AtomicGraph.from_ase(s, cutoff=1.0) for s in STRUCTURES]


@pytest.mark.parametrize("structure, graph", zip(STRUCTURES, GRAPHS))
def test_general(structure: Atoms, graph: AtomicGraph):
    assert number_of_atoms(graph) == len(structure)

    n_edges = number_of_edges(graph)
    assert n_edges == graph.neighbour_list.shape[1]
    assert n_edges == neighbour_vectors(graph).shape[0]
    assert n_edges == neighbour_distances(graph).shape[0]
    assert graph.other["total_spin"].item() == 1.0


def test_iso_atom():
    graph = AtomicGraph.from_ase(ISOLATED_ATOM, cutoff=1.0)
    assert number_of_atoms(graph) == 1
    assert number_of_edges(graph) == 0


def test_periodic_atom():
    graph = AtomicGraph.from_ase(PERIODIC_ATOM, cutoff=1.1)
    assert number_of_atoms(graph) == 1

    # 6 neighbours: up, down, left, right, front, back
    assert number_of_edges(graph) == 6


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 1.5])
def test_random_structure(cutoff: int):
    graph = AtomicGraph.from_ase(RANDOM_STRUCTURE, cutoff=cutoff)
    assert number_of_atoms(graph) == 8

    assert neighbour_distances(graph).max() <= cutoff


def test_charged_structure():
    graph = AtomicGraph.from_ase(CHARGED_STRUCTURE, cutoff=1.0)
    assert number_of_atoms(graph) == 1
    assert number_of_edges(graph) == 0
    assert graph.other["total_charge"].item() == 1.0


def test_get_labels():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    graph = AtomicGraph.from_ase(atoms, cutoff=1.0)

    assert graph.properties["forces"].shape == (2, 3)

    assert graph.properties["energy"].item() == -1.0

    with pytest.raises(KeyError):
        graph.properties["missing"]  # type: ignore


# Within each of these structures, each atom has the same number of neighbours,
# making it easy to test that the sum over neighbours is correct
@pytest.mark.parametrize("structure", [ISOLATED_ATOM, DIMER, PERIODIC_ATOM])
def test_sum_over_neighbours(structure):
    graph = AtomicGraph.from_ase(structure, cutoff=1.1)
    N = number_of_atoms(graph)
    E = number_of_edges(graph)
    n_neighbours = (graph.neighbour_list[0] == 0).sum()

    # check that summing is
    # (a) shape preserving
    # (b) compatible with torch.jit.script

    torchscript_sum: Callable[[torch.Tensor, AtomicGraph], torch.Tensor] = (
        torch.jit.script(sum_over_neighbours)
    )  # type: ignore
    for shape in [(E,), (E, 2), (E, 2, 3), (E, 2, 2, 2)]:
        for summing_fn in (
            sum_over_neighbours,
            torchscript_sum,
        ):
            edge_property = torch.ones(shape)
            result = summing_fn(edge_property, graph)
            assert result.shape == (N, *shape[1:])
            assert (result == n_neighbours).all()


def test_number_of_neighbours():
    graph = AtomicGraph.from_ase(ISOLATED_ATOM, cutoff=1.0)
    n = number_of_neighbours(graph, include_central_atom=False)
    assert n.shape == (1,)
    assert n.item() == 0

    n = number_of_neighbours(graph, include_central_atom=True)
    assert n.shape == (1,)
    assert n.item() == 1

    graph = AtomicGraph.from_ase(DIMER, cutoff=2.0)
    n = number_of_neighbours(graph, include_central_atom=False)
    assert n.shape == (2,)
    assert (n == 1).all()


@pytest.mark.parametrize(
    "shape",
    [tuple(), (4,), (4, 5)],
)
def test_sum_over_central_atom(shape: tuple[int, ...]):
    graph = AtomicGraph.create_with_defaults(
        R=torch.rand(2, 3),
        Z=torch.tensor([1, 1]),
    )

    p = torch.rand(7, *shape)
    index = torch.tensor([0] * 7)

    result = sum_over_central_atom_index(p, index, graph)
    assert result.shape == (2, *shape)

    # check that the sum is correct
    torch.testing.assert_close(result[0], p.sum(dim=0))
    # and that the other elements are zero
    torch.testing.assert_close(result[1], torch.zeros_like(p[0]))


def test_other():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False)
    atoms.info["foo"] = 1
    atoms.arrays["bar"] = np.array([1, 2, 3])
    graph = AtomicGraph.from_ase(
        atoms, cutoff=1.0, others_to_include=["foo", "bar"]
    )
    assert graph.other["foo"].item() == 1
    assert (graph.other["bar"] == torch.tensor([1, 2, 3])).all()
