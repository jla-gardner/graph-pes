import torch

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import (
    PropertyKey,
    neighbour_distances,
    neighbour_vectors,
    number_of_edges,
    sum_per_structure,
)
from graph_pes.utils.threebody import (
    angle_spanned_by,
    triplet_edge_pairs,
)


class MatterSim_M3Gnet_Wrapper(GraphPESModel):
    def __init__(self, model: torch.nn.Module):
        super().__init__(
            cutoff=model.model_args["cutoff"],  # type: ignore
            implemented_properties=["local_energies"],
            three_body_cutoff=model.model_args["threebody_cutoff"],  # type: ignore
        )
        self.model = model

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        # pre-compute
        edge_lengths = neighbour_distances(graph)  # (E)
        edge_pairs = triplet_edge_pairs(graph, self.three_body_cutoff.item())
        triplets_per_leading_edge = count_number_of_triplets_per_leading_edge(
            edge_pairs, graph
        )
        r_ik = edge_lengths[edge_pairs[:, 1]]
        v = neighbour_vectors(graph)
        v_ij = v[edge_pairs[:, 0]]
        v_ik = v[edge_pairs[:, 1]]
        angle = angle_spanned_by(v_ij, v_ik)

        num_atoms = sum_per_structure(
            torch.ones_like(graph.Z), graph
        ).unsqueeze(-1)

        # num_bonds is of shape (n_structures,) such that
        # num_bonds[i] = sum(graph.neighbour_list[0] == i)
        bonds_per_atom = torch.zeros_like(graph.Z)
        bonds_per_atom = bonds_per_atom.scatter_add(
            dim=0,
            index=graph.neighbour_list[0],
            src=torch.ones_like(graph.neighbour_list[0]),
        )
        num_bonds = sum_per_structure(bonds_per_atom, graph).unsqueeze(-1)

        three_body_indices = edge_pairs
        num_triple_ij = triplets_per_leading_edge.unsqueeze(-1)

        # use the forward pass of M3Gnet
        atom_attr = self.model.atom_embedding(self.model.one_hot_atoms(graph.Z))
        edge_attr = self.model.rbf(edge_lengths)
        edge_attr_zero = edge_attr
        edge_attr = self.model.edge_encoder(edge_attr)
        three_basis = self.model.sbf(r_ik, angle)

        for conv in self.model.graph_conv:
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                graph.neighbour_list,
                three_basis,
                three_body_indices,
                edge_lengths.unsqueeze(-1),
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        local_energies = self.model.final(atom_attr).view(-1)
        local_energies = self.model.normalizer(local_energies, graph.Z)

        return {"local_energies": local_energies}


def mattersim(load_path: str = "mattersim-v1.0.0-1m") -> GraphPESModel:
    """
    Load a ``mattersim`` model from a checkpoint file, and convert it to a
    :class:`~graph_pes.GraphPESModel` on the CPU.

    Parameters
    ----------
    load_path: str
        The path to the ``mattersim`` checkpoint file. Expected to be one of
        ``mattersim-v1.0.0-1m`` or ``mattersim-v1.0.0-5m`` currently.
    """
    from mattersim.forcefield.potential import Potential

    model = Potential.from_checkpoint(  # type: ignore
        load_path,
        load_training_state=False,  # only load the model
        device="cpu",  # manage the device ourself later
    ).model
    return MatterSim_M3Gnet_Wrapper(model)


@torch.no_grad()
def count_number_of_triplets_per_leading_edge(
    edge_pairs: torch.Tensor,
    graph: AtomicGraph,
):
    """
    Return ``T`` of shape ``(E,)`` where ``T[e]`` is the number of edge pairs
    that have edge number ``e`` as the first edge in the pair.

    Parameters
    ----------
    edge_pairs: torch.Tensor
        A ``(E, 2)`` shaped tensor indicating pairs of edges that form a
        triplet ``(i, j, k)`` (see :func:`triplet_edge_pairs`).
    graph: AtomicGraph
        The graph from which the edge pairs were derived.

    Returns
    -------
    triplets_per_edge: torch.Tensor
        A ``(E,)`` shaped tensor where ``triplets_per_edge[e]`` is the
        number of edge pairs that have edge ``e`` as the first edge in the
        pair.

    """
    triplets_per_edge = torch.zeros(
        number_of_edges(graph), device=graph.R.device, dtype=torch.long
    )
    return triplets_per_edge.scatter_add(
        dim=0,
        index=edge_pairs[:, 0],
        src=torch.ones_like(edge_pairs[:, 0]),
    )
