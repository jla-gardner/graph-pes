# note that this file is called MatterSim, so as to avoid
# clash with the mattersim package we import from

import torch

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import (
    PropertyKey,
    neighbour_distances,
    sum_per_structure,
)
from graph_pes.utils.threebody import (
    neighbour_triples_edge_view,
    triplet_bond_descriptors,
)


class MatterSim_M3Gnet_Wrapper(GraphPESModel):
    def __init__(self, model: torch.nn.Module):
        super().__init__(
            cutoff=model.model_args["cutoff"],  # type: ignore
            implemented_properties=["local_energies"],
        )
        self.model = model

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        # pre-compute
        edge_lengths = neighbour_distances(graph)  # (E)
        _, angle, _, r_ik = triplet_bond_descriptors(graph)
        three_body_indices, num_triple_ij = neighbour_triples_edge_view(graph)
        num_atoms = sum_per_structure(torch.ones_like(graph.Z), graph)

        # num_bonds is of shape (n_structures,) such that
        # num_bonds[i] = sum(graph.neighbour_list[0] == i)
        bonds_per_atom = torch.zeros_like(graph.Z)
        bonds_per_atom = bonds_per_atom.scatter_add(
            dim=0,
            index=graph.neighbour_list[0],
            src=torch.ones_like(graph.neighbour_list[0]),
        )
        num_bonds = sum_per_structure(bonds_per_atom, graph)

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
                num_bonds.unsqueeze(-1),
                num_triple_ij.unsqueeze(-1),
                num_atoms.unsqueeze(-1),
            )

        local_energies = self.model.final(atom_attr).view(-1)
        local_energies = self.model.normalizer(local_energies, graph.Z)

        return {"local_energies": local_energies}


def mattersim(load_path: str = "mattersim-v1.0.0-1m") -> GraphPESModel:
    from mattersim.forcefield.potential import Potential

    model = Potential.from_checkpoint(  # type: ignore
        load_path,
        load_training_state=False,  # only load the model
        device="cpu",  # manage the device ourself later
    ).model
    return MatterSim_M3Gnet_Wrapper(model)
