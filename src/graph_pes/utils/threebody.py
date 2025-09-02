from __future__ import annotations

import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    get_vectors,
    neighbour_distances,
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
    dot_product = torch.sum(v1 * v2, dim=1)

    v1_mag = torch.linalg.vector_norm(v1, dim=1)
    v2_mag = torch.linalg.vector_norm(v2, dim=1)

    cos_angle = dot_product / (v1_mag * v2_mag)
    cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-7, max=1.0 - 1e-7)

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

    ij, S_ij, ik, S_ik = triplet_edge_pairs(graph, graph.cutoff)  # (Y, 2)

    triplet_idxs = torch.cat([ij, ik[1, :].unsqueeze(0)], dim=0).transpose(
        0, 1
    )  # (Y, 3)

    if triplet_idxs.shape[0] == 0:
        return (
            triplet_idxs,
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
        )

    v1 = get_vectors(graph, i=ij[0, :], j=ij[1, :], shifts=S_ij)
    v2 = get_vectors(graph, i=ik[0, :], j=ik[1, :], shifts=S_ik)

    return (
        triplet_idxs,
        angle_spanned_by(v1, v2),
        torch.linalg.vector_norm(v1, dim=-1),
        torch.linalg.vector_norm(v2, dim=-1),
    )


def triplet_edge_pairs(
    graph: AtomicGraph,
    three_body_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Find all the pairs of edges, :math:`(i, j), (i, k)`, such that:

    * :math:`i, j, k \in \{0, 1, \dots, N-1\}` are indices of distinct
      (images of) atoms within the graph
    * :math:`j \neq k`
    * :math:`r_{ij} \leq` ``three_body_cutoff``
    * :math:`r_{ik} \leq` ``three_body_cutoff``

    Returns
    -------
    ij: torch.Tensor
    S_ij: torch.Tensor
    ik: torch.Tensor
    S_ik: torch.Tensor
    """

    if three_body_cutoff > graph.cutoff + 1e-6:
        raise ValueError(
            "Three-body cutoff is greater than the graph cutoff. "
            "This is not currently supported."
        )

    # check if already cached, creating the key in a torchscript compatible way
    # NB this gets added in the to_batch function, which is called on the worker
    #    threads. Since this function is slow, this speeds up training, but
    #    should not be used for MD/inference. Hence we don't cache any results
    #    to the graph within this function.

    # stupidly verbose to make torchscript happy
    cutoff_str = str(
        torch.round(torch.tensor(float(three_body_cutoff)), decimals=3).item()
    )
    key = "__threebody-" + cutoff_str
    ij = graph.other.get(key + "-ij")
    S_ij = graph.other.get(key + "-Sij")
    ik = graph.other.get(key + "-ik")
    S_ik = graph.other.get(key + "-Sik")
    if ij is not None:  # noqa: SIM102
        if S_ij is not None:  # noqa: SIM102
            if ik is not None:  # noqa: SIM102
                if S_ik is not None:  # noqa: SIM102
                    return (ij, S_ij, ik, S_ik)

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

        edge_pairs = torch.cat(edge_pairs)
        if edge_pairs.shape[0] == 0:
            return (
                torch.zeros(2, 0, device=graph.R.device),
                torch.zeros(0, 3, device=graph.R.device),
                torch.zeros(2, 0, device=graph.R.device),
                torch.zeros(0, 3, device=graph.R.device),
            )

        a, b = edge_pairs[:, 0], edge_pairs[:, 1]

        return (
            graph.neighbour_list[:, a],
            graph.neighbour_cell_offsets[a],
            graph.neighbour_list[:, b],
            graph.neighbour_cell_offsets[b],
        )


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

    ij, S_ij, ik, S_ik = triplet_edge_pairs(graph, three_body_cutoff)

    v_ij = get_vectors(graph, i=ij[0, :], j=ij[1, :], shifts=S_ij)
    v_ik = get_vectors(graph, i=ik[0, :], j=ik[1, :], shifts=S_ik)
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
