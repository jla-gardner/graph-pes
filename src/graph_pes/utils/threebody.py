from __future__ import annotations

import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    neighbour_vectors,
    number_of_edges,
)


def angle_spanned_by(v1: torch.Tensor, v2: torch.Tensor):
    """
    Calculate angles between corresponding vectors in two batches.

    Parameters
    ----------
    v1
        First batch of vectors, shape (N, 3)
    v2
        Second batch of vectors, shape (N, 3)

    Returns
    -------
    torch.Tensor
        Angles in radians, shape (N,)
    """
    # Compute dot product
    dot_product = torch.sum(v1 * v2, dim=1)

    # Compute magnitudes
    v1_mag = torch.linalg.vector_norm(v1, dim=1)
    v2_mag = torch.linalg.vector_norm(v2, dim=1)

    # Compute cosine of angle, add small epsilon to prevent division by zero
    cos_angle = dot_product / (v1_mag * v2_mag + 1e-8)

    # Clamp cosine values to handle numerical instabilities
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-8, 1.0 - 1e-8)

    # Compute angle using arccos
    return torch.arccos(cos_angle)


@torch.no_grad()
def neighbour_triplets(graph: AtomicGraph) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Find all the triplets :math:`(i, j, k)` such that
    :math:`k` and :math:`j` are neighbours of :math:`i`,
    and:

    * :math:`j \neq k` for non periodic graphs
    * :math:`j, k` do not refer to the same image of an atom (respecting pbcs)
      for periodic graphs

    Returns
    -------
    triplet_idxs
        A ``(Y, 3)`` shaped tensor indicating the triplets, such that

        .. code-block:: python

            # get the y'th triplet
            i, j, k = triplet_idxs[y]

    triplet_vectors
        A ``(Y, 2, 3)`` shaped tensor indicating the vectors, such that

        .. code-block:: python

            # get the y'th vector triplet
            v_ij, v_ik = vector_triplets[y]

    Examples
    --------
    Note that the triplets are permutation sensitive, and hence
    both ``(i, j, k)`` and ``(i, k, j)`` are included:

    >>> graph = AtomicGraph.from_ase(molecule("H2O"))
    >>> triplet_idxs, triplet_vectors = neighbour_triplets(graph)
    >>> triplet_idxs
    tensor([[0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]])
    """

    vectors = neighbour_vectors(graph)

    # graph.neighbour_list is an (2, E) shaped tensor
    # indicating the indices of the neighbours of each atom
    # the first row is the central atom index
    # the second row is the neighbour atom index
    # the indices are 0-indexed
    central_atoms = graph.neighbour_list[0]
    neighbour_atoms = graph.neighbour_list[1]

    # we want to find all the triplets (i, j, k) such that
    # k and j are neighbours of i, and j != k (respecting pbc ghost atoms)
    triplet_idxs = []
    triplets_vectors = []

    for i in torch.unique(central_atoms):
        mask = central_atoms == i
        relevant_neighbours = neighbour_atoms[mask]
        relevant_vectors = vectors[mask]

        N = relevant_neighbours.shape[0]
        _idx = torch.cartesian_prod(
            torch.arange(N),
            torch.arange(N),
        )  # (N**2, 2)
        _idx = _idx[_idx[:, 0] != _idx[:, 1]]  # (N**2 - N, 2)

        kj = relevant_neighbours[_idx]  # (N**2 - N, 2)
        ikj = torch.hstack([i.expand(kj.shape[0]).view(-1, 1), kj])
        triplet_idxs.append(ikj)

        vkj = relevant_vectors[_idx]  # (N**2 - N, 2, 3)
        triplets_vectors.append(vkj)

    if len(triplet_idxs) == 0:
        return (
            torch.zeros(0, 3, device=graph.R.device).long(),
            torch.zeros(0, 2, 3, device=graph.R.device),
        )

    triplet_idxs = torch.vstack(triplet_idxs)
    triplets_vectors = torch.vstack(triplets_vectors)

    return triplet_idxs, triplets_vectors


@torch.no_grad()
def triplet_bond_descriptors(
    graph: AtomicGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    For each triplet :math:`(i, j, k)`, get the bond angle :math:`\theta_{jik}`
    (in radians) and the two bond lengths :math:`r_{ij}` and :math:`r_{ik}`.

    Returns
    -------
    triplet_idxs
        The triplet indices, :math:`(i, j, k)`, of shape ``(Y, 3)``.
    angle
        The bond angle :math:`\theta_{jik}`, shape ``(Y,)``.
    r_ij
        The bond length :math:`r_{ij}`, shape ``(Y,)``.
    r_ik
        The bond length :math:`r_{ik}`, shape ``(Y,)``.

    Examples
    --------
    >>> graph = AtomicGraph.from_ase(molecule("H2O"))
    >>> angle, r_ij, r_ik = triplet_bond_descriptors(graph)
    >>> torch.rad2deg(angle)
    tensor([103.9999, 103.9999,  38.0001,  38.0001,  38.0001,  38.0001])
    """

    triplet_idxs, vector_triplets = neighbour_triplets(graph)

    if vector_triplets.shape[0] == 0:
        return (
            triplet_idxs,
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
        )

    v1 = vector_triplets[:, 0]
    v2 = vector_triplets[:, 1]

    return (
        triplet_idxs,
        angle_spanned_by(v1, v2),
        torch.linalg.vector_norm(v1, dim=-1),
        torch.linalg.vector_norm(v2, dim=-1),
    )


@torch.no_grad()
def neighbour_triples_edge_view(
    graph: AtomicGraph,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find all the triplets :math:`(i, j, k)` such that
    :math:`k` and :math:`j` are both neighbours of :math:`i`,
    and:

    * :math:`j \neq k` for non periodic graphs
    * :math:`j, k` do not refer to the same image of an atom (respecting pbcs)
      for periodic graphs

    Returns
    -------
    edge_pairs: torch.Tensor
        A ``(Y, 2)`` shaped tensor indicating the edges, such that

        .. code-block:: python

            a, b = edge_pairs[y]
            i, j = graph.neighbour_list[a]
            i, k = graph.neighbour_list[b]

    triplets_per_edge: torch.Tensor
        A ``(E)`` shaped tensor indicating in how many ordered pairs of edges
        each edge is first in, such that

        .. code-block:: python

            triplets_per_edge[a] == sum(edge_pairs[:, 0] == a)
    """
    central_atoms = graph.neighbour_list[0]

    edge_indexes = torch.arange(number_of_edges(graph), device=graph.R.device)

    edge_pairs = []
    triplets_per_edge = torch.zeros_like(edge_indexes)

    for i in torch.unique(central_atoms):
        mask = central_atoms == i
        relevant_edge_indexes = edge_indexes[mask]

        N = relevant_edge_indexes.shape[0]
        _idx = torch.cartesian_prod(
            torch.arange(N),
            torch.arange(N),
        )  # (N**2, 2)
        _idx = _idx[_idx[:, 0] != _idx[:, 1]]  # (N**2 - N, 2)

        pairs_for_i = relevant_edge_indexes[_idx]
        edge_pairs.append(pairs_for_i)

    edge_pairs = torch.cat(edge_pairs)
    for j in range(triplets_per_edge.shape[0]):
        triplets_per_edge[j] = (edge_pairs[:, 0] == j).sum()
    return edge_pairs, triplets_per_edge
