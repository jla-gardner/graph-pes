from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule

from graph_pes import AtomicGraph
from graph_pes.atomic_graph import number_of_atoms, number_of_edges
from graph_pes.utils.threebody import (
    triplet_bond_descriptors,
    triplet_edge_pairs,
)


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

    def assert_close(a, b):
        return torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    ############################################################
    # first triplet is 0-1-2
    # therefore angle is theta
    # and r_ij and r_ik are a and b
    assert list(triplet_idxs[0]) == [0, 1, 2]
    assert_close(angle[0], rad_theta)
    assert_close(r_ij[0], torch.tensor(a))
    assert_close(r_ik[0], torch.tensor(b))

    ############################################################
    # second triplet is 0-2-1
    assert list(triplet_idxs[1]) == [0, 2, 1]
    assert_close(angle[1], rad_theta)
    assert_close(r_ij[1], torch.tensor(b))
    assert_close(r_ik[1], torch.tensor(a))

    ############################################################
    # third triplet is 1-0-2
    assert list(triplet_idxs[2]) == [1, 0, 2]
    assert_close(r_ij[2], torch.tensor(a))
    # use cosine rule to get r_ik
    c = torch.sqrt(a**2 + b**2 - 2 * a * b * torch.cos(rad_theta))
    assert_close(r_ik[2], c)
    # use sine rule to get angle
    sin_phi = b * torch.sin(rad_theta) / c
    phi = torch.asin(sin_phi)
    assert_close(angle[2], phi)

    ############################################################
    # fourth triplet is 1-2-0
    assert list(triplet_idxs[3]) == [1, 2, 0]
    assert_close(r_ij[3], c)
    assert_close(r_ik[3], torch.tensor(a))
    assert_close(angle[3], phi)

    ############################################################
    # fifth triplet is 2-0-1
    assert list(triplet_idxs[4]) == [2, 0, 1]
    sin_zeta = a * torch.sin(rad_theta) / c
    zeta = torch.asin(sin_zeta)
    assert_close(angle[4], zeta)

    ############################################################
    # sixth triplet is 2-1-0
    assert list(triplet_idxs[5]) == [2, 1, 0]
    assert_close(r_ij[5], c)
    assert_close(r_ik[5], torch.tensor(b))
    assert_close(angle[5], zeta)


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

    triplets = triplet_edge_pairs(graph, graph.cutoff)
    assert triplets.shape == (0, 2)

    atoms = Atoms("H", positions=[(0.5, 0.5, 0.5)], cell=np.eye(3), pbc=True)
    graph = AtomicGraph.from_ase(atoms, cutoff=1.1)
    assert number_of_atoms(graph) == 1

    # 6 neighbours (up, down, left, right, front, back)
    assert number_of_edges(graph) == 6

    # 6 * 5 = 30 triplets
    # (up, [down, left, right, front, back]),
    # (down, [up, left, right, front, back]),
    # etc.
    triplets = triplet_edge_pairs(graph, graph.cutoff)
    assert triplets.shape == (30, 2)

    triplets = triplet_edge_pairs(graph, 0.5)
    assert triplets.shape == (0, 2)
