from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from graph_pes.atomic_graph import (
    AtomicGraph,
    neighbour_distances,
    neighbour_triplets,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
    number_of_neighbours,
    sum_over_central_atom_index,
    sum_over_neighbours,
    triplet_bond_descriptors,
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
GRAPHS = [AtomicGraph.from_ase(s, cutoff=1.0) for s in STRUCTURES]


@pytest.mark.parametrize("structure, graph", zip(STRUCTURES, GRAPHS))
def test_general(structure: Atoms, graph: AtomicGraph):
    assert number_of_atoms(graph) == len(structure)

    n_edges = number_of_edges(graph)
    assert n_edges == graph.neighbour_list.shape[1]
    assert n_edges == neighbour_vectors(graph).shape[0]
    assert n_edges == neighbour_distances(graph).shape[0]


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


def check_angle_measures(a: float, b: float, theta: float):
    #  (2)            create a molecule with atoms at
    #   |                  (1) at [0, 0, 0]
    #   |                  (2) at [a, 0, 0]
    #   | a                (3) at [b * cos(theta), b * sin(theta), 0]
    #   |
    #   |
    #  (1) theta
    #    \
    #     \ b
    #      \
    #       \
    #        (3)

    a = float(a)
    b = float(b)

    rad_theta = torch.deg2rad(torch.tensor(theta))
    R = torch.tensor(
        [
            [0, 0, 0],
            [a, 0, 0],
            [torch.cos(rad_theta) * b, torch.sin(rad_theta) * b, 0],
        ]
    )

    graph = AtomicGraph.from_ase(molecule("H2O"))._replace(R=R)

    triplet_idxs, angle, r_ij, r_ik = triplet_bond_descriptors(graph)

    assert len(angle) == len(triplet_idxs) == 6

    ############################################################
    # first triplet is 0-1-2
    # therefore angle is theta
    # and r_ij and r_ik are a and b
    assert list(triplet_idxs[0]) == [0, 1, 2]
    torch.testing.assert_close(angle[0], rad_theta)
    torch.testing.assert_close(r_ij[0], torch.tensor(a))
    torch.testing.assert_close(r_ik[0], torch.tensor(b))

    ############################################################
    # second triplet is 0-2-1
    assert list(triplet_idxs[1]) == [0, 2, 1]
    torch.testing.assert_close(angle[1], rad_theta)
    torch.testing.assert_close(r_ij[1], torch.tensor(b))
    torch.testing.assert_close(r_ik[1], torch.tensor(a))

    ############################################################
    # third triplet is 1-0-2
    assert list(triplet_idxs[2]) == [1, 0, 2]
    torch.testing.assert_close(r_ij[2], torch.tensor(a))
    # use cosine rule to get r_ik
    c = torch.sqrt(a**2 + b**2 - 2 * a * b * torch.cos(rad_theta))
    torch.testing.assert_close(r_ik[2], c)
    # use sine rule to get angle
    sin_phi = b * torch.sin(rad_theta) / c
    phi = torch.asin(sin_phi)
    torch.testing.assert_close(angle[2], phi)

    ############################################################
    # fourth triplet is 1-2-0
    assert list(triplet_idxs[3]) == [1, 2, 0]
    torch.testing.assert_close(r_ij[3], c)
    torch.testing.assert_close(r_ik[3], torch.tensor(a))
    torch.testing.assert_close(angle[3], phi)

    ############################################################
    # fifth triplet is 2-0-1
    assert list(triplet_idxs[4]) == [2, 0, 1]
    sin_zeta = a * torch.sin(rad_theta) / c
    zeta = torch.asin(sin_zeta)
    torch.testing.assert_close(angle[4], zeta)

    ############################################################
    # sixth triplet is 2-1-0
    assert list(triplet_idxs[5]) == [2, 1, 0]
    torch.testing.assert_close(r_ij[5], c)
    torch.testing.assert_close(r_ik[5], torch.tensor(b))
    torch.testing.assert_close(angle[5], zeta)


def test_angle_measures():
    # test a range of angles
    for angle in torch.linspace(10, 180, 10):
        check_angle_measures(1, 1, angle.item())

    # and a range of bond lengths
    for length in torch.linspace(0.5, 2, 10):
        check_angle_measures(1.0, length.item(), 123)


def test_triplets_on_isolated_atoms():
    # deliberately no neighbour list
    graph = AtomicGraph.create_with_defaults(
        R=torch.rand(3, 3), Z=torch.rand(3)
    )
    assert graph.neighbour_list.shape == (2, 0)

    triplets, _ = neighbour_triplets(graph)
    assert triplets.shape == (0, 3)

    atoms = Atoms("H", positions=[(0.5, 0.5, 0.5)], cell=np.eye(3), pbc=True)
    graph = AtomicGraph.from_ase(atoms, cutoff=1.1)
    assert number_of_atoms(graph) == 1

    # 6 neighbours (up, down, left, right, front, back)
    assert number_of_edges(graph) == 6

    # 6 * 5 = 30 triplets
    # (up, [down, left, right, front, back]),
    # (down, [up, left, right, front, back]),
    # etc.
    triplets, _ = neighbour_triplets(graph)
    assert triplets.shape == (30, 3)


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
