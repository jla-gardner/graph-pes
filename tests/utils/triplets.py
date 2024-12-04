import torch
from ase.build import molecule
from graph_pes import AtomicGraph
from graph_pes.utils.triplets import (
    neighbour_triplets,
    triplet_bond_descriptors,
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
    Z = torch.tensor([1, 2, 3])

    graph = AtomicGraph.from_ase(molecule("H2O"))._replace(R=R, Z=Z)

    triplets, _ = neighbour_triplets(graph)
    angle, r_ij, r_ik = triplet_bond_descriptors(graph)

    assert len(angle) == len(triplets) == 6

    ############################################################
    # first triplet is 1-2-3
    # therefore angle is theta
    # and r_ij and r_ik are a and b
    t1 = triplets[0]
    assert list(graph.Z[t1]) == [1, 2, 3]
    torch.testing.assert_close(angle[0], rad_theta)
    torch.testing.assert_close(r_ij[0], torch.tensor(a))
    torch.testing.assert_close(r_ik[0], torch.tensor(b))

    ############################################################
    # second triplet is 1-3-2
    t2 = triplets[1]
    assert list(graph.Z[t2]) == [1, 3, 2]
    torch.testing.assert_close(angle[1], rad_theta)
    torch.testing.assert_close(r_ij[1], torch.tensor(b))
    torch.testing.assert_close(r_ik[1], torch.tensor(a))

    ############################################################
    # third triplet is 2-1-3
    t3 = triplets[2]
    assert list(graph.Z[t3]) == [2, 1, 3]
    torch.testing.assert_close(r_ij[2], torch.tensor(a))
    # use cosine rule to get r_ik
    c = torch.sqrt(a**2 + b**2 - 2 * a * b * torch.cos(rad_theta))
    torch.testing.assert_close(r_ik[2], c)
    # use sine rule to get angle
    sin_phi = b * torch.sin(rad_theta) / c
    phi = torch.asin(sin_phi)
    torch.testing.assert_close(angle[2], phi)

    ############################################################
    # fourth triplet is 2-3-1
    t4 = triplets[3]
    assert list(graph.Z[t4]) == [2, 3, 1]
    torch.testing.assert_close(r_ij[3], c)
    torch.testing.assert_close(r_ik[3], torch.tensor(a))
    torch.testing.assert_close(angle[3], phi)

    ############################################################
    # fifth triplet is 3-1-2
    t5 = triplets[4]
    assert list(graph.Z[t5]) == [3, 1, 2]
    sin_zeta = a * torch.sin(rad_theta) / c
    zeta = torch.asin(sin_zeta)
    torch.testing.assert_close(angle[4], zeta)

    ############################################################
    # sixth triplet is 3-2-1
    t6 = triplets[5]
    assert list(graph.Z[t6]) == [3, 2, 1]
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
