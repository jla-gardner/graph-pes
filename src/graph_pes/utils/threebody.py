from __future__ import annotations

import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    get_vectors,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
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
    cos_angle = dot_product / (v1_mag * v2_mag)

    # Clamp cosine values to handle numerical instabilities
    cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-7, max=1.0 - 1e-7)

    # Compute angle using arccos
    return torch.arccos(cos_angle)


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

    edge_pairs = triplet_edge_pairs(graph, graph.cutoff)  # (Y, 2)

    ij = graph.neighbour_list[:, edge_pairs[:, 0]]  # (2, Y)
    k = graph.neighbour_list[1, edge_pairs[:, 1]].unsqueeze(0)  # (1, Y)
    triplet_idxs = torch.cat([ij, k], dim=0).transpose(0, 1)  # (Y, 3)

    if triplet_idxs.shape[0] == 0:
        return (
            triplet_idxs,
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
        )

    v = neighbour_vectors(graph)
    v1 = v[edge_pairs[:, 0]]
    v2 = v[edge_pairs[:, 1]]

    return (
        triplet_idxs,
        angle_spanned_by(v1, v2),
        torch.linalg.vector_norm(v1, dim=-1),
        torch.linalg.vector_norm(v2, dim=-1),
    )


def triplet_edge_pairs(
    graph: AtomicGraph,
    three_body_cutoff: float,
) -> torch.Tensor:
    r"""
    Find all the pairs of edges, :math:`a = (i, j), b = (i, k)`, such that:

    * :math:`i, j, k \in \{0, 1, \dots, N-1\}` are indices of distinct
      (images of) atoms within the graph
    * :math:`j \neq k`
    * :math:`r_{ij} \leq` ``three_body_cutoff``
    * :math:`r_{ik} \leq` ``three_body_cutoff``

    Returns
    -------
    edge_pairs: torch.Tensor
        A ``(Y, 2)`` shaped tensor indicating the edges, such that

        .. code-block:: python

            a, b = edge_pairs[y]
            i, j = graph.neighbour_list[:,a]
            i, k = graph.neighbour_list[:,b]
    """

    if three_body_cutoff > graph.cutoff + 1e-6:
        raise ValueError(
            "Three-body cutoff is greater than the graph cutoff. "
            "This is not currently supported."
        )

    # check if already cached, using old .format to be torchscript compatible
    # NB this gets added in the to_batch function, which is called on the worker
    #    threads. Since this function is slow, this speeds up training, but
    #    should not be used for MD/inference. Hence we don't cache any results
    #    to the graph within this function.
    key = "__threebody-{:.3f}-{:.3f}".format(three_body_cutoff, graph.cutoff)  # noqa: UP032
    if key in graph.other:
        v = graph.other.get(key)
        if v is not None:
            return v

    with torch.no_grad():
        edge_indexes = torch.arange(
            number_of_edges(graph), device=graph.R.device
        )

        three_body_mask = neighbour_distances(graph) < three_body_cutoff
        relevant_edge_indexes = edge_indexes[three_body_mask]
        relevant_central_atoms = graph.neighbour_list[0][relevant_edge_indexes]

        edge_pairs = []

        for i in range(number_of_atoms(graph)):
            mask = relevant_central_atoms == i
            masked_edge_indexes = relevant_edge_indexes[mask]

            # number of edges of distance <= three_body_cutoff
            # that have i as a central atom
            N = masked_edge_indexes.shape[0]
            _idx = torch.cartesian_prod(
                torch.arange(N),
                torch.arange(N),
            )  # (N**2, 2)
            _idx = _idx[_idx[:, 0] != _idx[:, 1]]  # (N**2 - N, 2)

            pairs_for_i = masked_edge_indexes[_idx]
            edge_pairs.append(pairs_for_i)

        return torch.cat(edge_pairs)


def triplet_edges(
    graph: AtomicGraph,
    three_body_cutoff: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Finds all ``Y`` triplets ``(i, j, k)`` such that:

    * ``i, j, k`` are indices of distinct (images of) atoms within the graph
    * ``r_{ij} <=`` ``three_body_cutoff``
    * ``r_{ik} <=`` ``three_body_cutoff``

    Returns
    -------
    i: torch.Tensor
        The central atom indices, shape ``(Y,)``.
    j: torch.Tensor
        The first paired atom indices, shape ``(Y,)``.
    k: torch.Tensor
        The second paired atom indices, shape ``(Y,)``.
    r_ij: torch.Tensor
        The bond length :math:`r_{ij}`, shape ``(Y,)``.
    r_ik: torch.Tensor
        The bond length :math:`r_{ik}`, shape ``(Y,)``.
    r_jk: torch.Tensor
        The bond length :math:`r_{jk}`, shape ``(Y,)``.
    """

    ij_ik = triplet_edge_pairs(graph, three_body_cutoff)
    ij = graph.neighbour_list[:, ij_ik[:, 0]]
    ik = graph.neighbour_list[:, ij_ik[:, 1]]
    shifts_ij = graph.neighbour_cell_offsets[ij_ik[:, 0]]
    shifts_ik = graph.neighbour_cell_offsets[ij_ik[:, 1]]

    v_ij = get_vectors(graph, i=ij[0, :], j=ij[1, :], shifts=shifts_ij)
    v_ik = get_vectors(graph, i=ik[0, :], j=ik[1, :], shifts=shifts_ik)
    v_jk = v_ik - v_ij

    r_ij = torch.norm(v_ij, dim=-1)
    r_ik = torch.norm(v_ik, dim=-1)
    r_jk = torch.norm(v_jk, dim=-1)

    return (
        ij[0, :],
        ij[1, :],
        ik[1, :],
        r_ij,
        r_ik,
        r_jk,
    )
