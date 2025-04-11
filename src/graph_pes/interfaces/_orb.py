from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ase.geometry.cell import cell_to_cellpar

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    edges_per_graph,
    is_batch,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    structure_sizes,
    to_batch,
    trim_edges,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import voigt_6_to_full_3x3

if TYPE_CHECKING:
    from orb_models.forcefield.conservative_regressor import (
        ConservativeForcefieldRegressor,
    )
    from orb_models.forcefield.direct_regressor import (
        DirectForcefieldRegressor,
    )


def from_graph_pes_to_orb_batch(
    graph: AtomicGraph,
    cutoff: float,
    max_neighbours: int,
):
    from orb_models.forcefield.base import AtomGraphs as OrbGraph

    if not is_batch(graph):
        graph = to_batch([graph])

    graph = trim_edges(graph, cutoff)
    distances = neighbour_distances(graph)

    new_nl, new_offsets = [], []
    for i in range(number_of_atoms(graph)):
        mask = graph.neighbour_list[0] == i
        d = distances[mask]
        if d.numel() == 0:
            continue
        elif d.numel() < max_neighbours:
            new_nl.append(graph.neighbour_list[:, mask])
            new_offsets.append(graph.neighbour_cell_offsets[mask])
        else:
            topk = torch.topk(d, k=max_neighbours, largest=False)
            new_nl.append(graph.neighbour_list[:, mask][:, topk.indices])
            new_offsets.append(graph.neighbour_cell_offsets[mask][topk.indices])

    graph = graph._replace(
        neighbour_list=torch.hstack(new_nl),
        neighbour_cell_offsets=torch.vstack(new_offsets),
    )

    node_features = {
        "atomic_numbers": graph.Z.long(),
        "positions": graph.R,
        "atomic_numbers_embedding": torch.nn.functional.one_hot(
            graph.Z, num_classes=118
        ),
        "atom_identity": torch.arange(number_of_atoms(graph))
        .long()
        .to(graph.R.device),
    }

    edge_features = {
        "vectors": neighbour_vectors(graph),
        "unit_shifts": graph.neighbour_cell_offsets,
    }

    lattices = []
    for cell in graph.cell:
        lattices.append(
            torch.from_numpy(cell_to_cellpar(cell.cpu().numpy())).float()
        )
    lattice = torch.vstack(lattices).to(graph.R.device)

    graph_features = {
        "cell": graph.cell,
        "pbc": torch.Tensor([False, False, False])
        if not graph.pbc
        else graph.pbc,
        "lattice": lattice,
    }

    return OrbGraph(
        senders=graph.neighbour_list[0],
        receivers=graph.neighbour_list[1],
        n_node=structure_sizes(graph),
        n_edge=edges_per_graph(graph),
        node_features=node_features,
        edge_features=edge_features,
        system_features=graph_features,
        node_targets={},
        edge_targets={},
        system_targets={},
        fix_atoms=None,
        tags=torch.zeros(number_of_atoms(graph)),
        radius=cutoff,
        max_num_neighbors=torch.tensor([max_neighbours]),
        system_id=None,
    )


class OrbWrapper(GraphPESModel):
    def __init__(self, orb: torch.nn.Module):
        from orb_models.forcefield.conservative_regressor import (
            ConservativeForcefieldRegressor,
        )
        from orb_models.forcefield.direct_regressor import (
            DirectForcefieldRegressor,
        )

        assert isinstance(
            orb, (DirectForcefieldRegressor, ConservativeForcefieldRegressor)
        )
        super().__init__(
            cutoff=orb.system_config.radius,
            implemented_properties=[
                "local_energies",
                "energy",
                "forces",
                "stress",
            ],
        )
        self._orb = orb

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        orb_graph = from_graph_pes_to_orb_batch(
            graph,
            self._orb.system_config.radius,
            self._orb.system_config.max_num_neighbors,
        )
        preds: dict[PropertyKey, torch.Tensor] = self._orb.predict(orb_graph)  # type: ignore
        preds["stress"] = voigt_6_to_full_3x3(preds["stress"])

        if not is_batch(graph):
            preds["energy"] = preds["energy"][0]

        preds["local_energies"] = torch.zeros(number_of_atoms(graph)).to(
            graph.Z.device
        )

        print(self.training, preds["energy"].shape)

        return preds

    @property
    def orb_model(
        self,
    ) -> "DirectForcefieldRegressor | ConservativeForcefieldRegressor":
        return self._orb


def orb_model(name: str = "orb-v3-direct-20-omat") -> OrbWrapper:
    from orb_models.forcefield import pretrained

    orb = pretrained.ORB_PRETRAINED_MODELS[name](device="cpu")
    for param in orb.parameters():
        param.requires_grad = True
    return OrbWrapper(orb)
