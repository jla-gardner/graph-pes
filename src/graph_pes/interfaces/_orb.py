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
from graph_pes.interfaces.base import InterfaceModel
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

    if new_nl:
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
        "atom_identity": torch.arange(number_of_atoms(graph)).long(),
    }

    edge_features = {
        "vectors": neighbour_vectors(graph),
        "unit_shifts": graph.neighbour_cell_offsets,
    }

    lattices = []
    for cell in graph.cell.clone().detach():
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
    ).to(device=graph.R.device, dtype=graph.R.dtype)


class OrbWrapper(InterfaceModel):
    """
    A wrapper around an ``orb-models`` model that converts it into a
    :class:`~graph_pes.GraphPESModel`.

    Parameters
    ----------
    orb
        The ``orb-models`` model to wrap.
    """

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

    def convert_to_underlying_input(self, graph: AtomicGraph):
        return from_graph_pes_to_orb_batch(
            graph,
            self._orb.system_config.radius,
            self._orb.system_config.max_num_neighbors,
        )

    def raw_forward_pass(
        self,
        input,
        is_batched: bool,
        properties: list[PropertyKey],
    ) -> dict[PropertyKey, torch.Tensor]:
        preds: dict[PropertyKey, torch.Tensor] = self._orb.predict(input)  # type: ignore

        # no easy access to these, but required by the interface, so set to 0
        preds["local_energies"] = torch.zeros_like(
            input.node_features["atomic_numbers"]
        ).float()

        # handle some of the outputs
        if "grad_forces" in preds:
            preds["forces"] = preds.pop("grad_forces")  # type: ignore
        if "grad_stress" in preds:
            preds["stress"] = preds.pop("grad_stress")  # type: ignore
        if "stress" in preds:
            preds["stress"] = voigt_6_to_full_3x3(preds["stress"])

        # underlying orb model returns things in batched format.
        # we want to de-batch things if only a single graph is provided
        if not is_batched:
            preds["energy"] = preds["energy"][0]
            preds["stress"] = preds["stress"].squeeze()

        return preds

    @property
    def orb_model(
        self,
    ) -> "DirectForcefieldRegressor | ConservativeForcefieldRegressor":
        r"""
        Access the underlying ``orb-models`` model.

        One use case of this is to to use ``graph-pes``\ 's fine-tuning
        functionality to adapt an existing ``orb-models`` model to a new
        dataset. You can then re-extract the underlying ``orb-models`` model
        using this property and use it in other ``orb-models`` workflows.
        """
        return self._orb


def orb_model(name: str = "orb-v3-direct-20-omat") -> OrbWrapper:
    """
    Load a pre-trained Orb model, and convert it into a
    :class:`~graph_pes.GraphPESModel`.

    See the `orb-models <https://github.com/orbital-materials/orb-models>`_
    repository for more information on the available models.
    As of 2025-04-11, the following are available:

    * ``"orb-v3-conservative-20-omat"``
    * ``"orb-v3-conservative-inf-omat"``
    * ``"orb-v3-direct-20-omat"``
    * ``"orb-v3-direct-inf-omat"``
    * ``"orb-v3-conservative-20-mpa"``
    * ``"orb-v3-conservative-inf-mpa"``
    * ``"orb-v3-direct-20-mpa"``
    * ``"orb-v3-direct-inf-mpa"``
    * ``"orb-v2"``
    * ``"orb-d3-v2"``
    * ``"orb-d3-sm-v2"``
    * ``"orb-d3-xs-v2"``
    * ``"orb-mptraj-only-v2"``

    Parameters
    ----------
    name: str
        The name of the model to load.
    """
    import torch._functorch.config
    from orb_models.forcefield import pretrained

    torch._functorch.config.donated_buffer = False

    orb = pretrained.ORB_PRETRAINED_MODELS[name](device="cpu")
    for param in orb.parameters():
        param.requires_grad = True
    return OrbWrapper(orb)
