from __future__ import annotations

import numpy as np
import torch

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    index_over_neighbours,
    neighbour_distances,
    sum_over_neighbours,
    triplet_bond_descriptors,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.distances import (
    DistanceExpansion,
    GaussianSmearing,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.utils.nn import (
    MLP,
    PerElementEmbedding,
    ShiftedSoftplus,
    UniformModuleList,
)

__author__ = "Stefan Bringuier"
__email__ = "stefan.bringuier@gmail.com"
__license__ = "See LICENSE"


class M3GNetInteraction(torch.nn.Module):
    r"""
    M3GNet interaction block.

    This implements the core interaction block of the M3GNet architecture as
    described in https://doi.org/10.1038/s43588-022-00349-3

    The block performs the following operations:
    1. Pre-linear transformation of node features
    2. Three-body message passing using:
       - Two-body messages from radial basis functions
       - Three-body angular messages from bond angles
    3. Message aggregation
    4. Post-linear transformation
    """

    def __init__(
        self,
        channels: int,
        expansion_features: int,
        cutoff: float,
        basis_type: type[DistanceExpansion],
    ):
        super().__init__()

        self.cutoff = cutoff

        self.pre_linear = torch.nn.Linear(channels, channels, bias=False)

        self.radial_basis = basis_type(expansion_features, cutoff)
        self.two_body_net = MLP(
            [expansion_features, expansion_features, channels],
            activation=ShiftedSoftplus(),
            bias=False,
        )

        self.three_body_net = MLP(
            [3, channels // 2, channels],
            activation=ShiftedSoftplus(),
            bias=False,
        )

        self.post_linear = torch.nn.Linear(channels, channels, bias=False)

    def compute_three_body_messages(
        self,
        node_features: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        triplet_idxs, angles, r_ij, r_ik = triplet_bond_descriptors(graph)

        if triplet_idxs.shape[0] == 0:
            return torch.zeros_like(node_features[graph.neighbour_list[0]])

        three_body_features = torch.stack(
            [r_ij, r_ik, torch.cos(angles)], dim=-1
        )
        three_body_weights = self.three_body_net(three_body_features)

        # Weight the neighbor j features
        neighbor_features = node_features[triplet_idxs[:, 1]]
        messages = neighbor_features * three_body_weights

        edge_messages = torch.zeros_like(node_features[graph.neighbour_list[0]])
        edge_messages.index_add_(0, triplet_idxs[:, 1], messages)

        return edge_messages

    def forward(
        self,
        features: torch.Tensor,
        neighbour_distances: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        h = self.pre_linear(features)

        radial_basis = self.radial_basis(neighbour_distances.unsqueeze(-1))

        # Apply smooth cutoff
        cutoff_mask = (neighbour_distances < self.cutoff).float()
        cutoff_values = 0.5 * (
            1 + torch.cos(np.pi * neighbour_distances / self.cutoff)
        )
        radial_basis = (
            radial_basis
            * cutoff_values.unsqueeze(-1)
            * cutoff_mask.unsqueeze(-1)
        )

        # Two-body * 3-body messages
        neighbor_h = index_over_neighbours(h, graph)
        two_body_messages = neighbor_h * self.two_body_net(radial_basis)
        three_body_messages = self.compute_three_body_messages(h, graph)

        # Message aggregation
        messages = two_body_messages + three_body_messages
        h = sum_over_neighbours(messages, graph)

        return self.post_linear(h)


class M3GNet(GraphPESModel):
    r"""
    Implementation of `M3GNet <https://doi.org/10.1038/s43588-022-00349-3>`_
    in the `graph-pes` library. Incorporates key features of the
    original M3GNet, including:

    - **Three-body Interactions**: through the `M3GNetInteraction` blocks.
    - **Radial Basis Functions**: Implemented via the `DistanceExpansion`
    class, so to be similar to the original M3GNet.
    - **Layered Architecture**: Composed of multiple interaction layers
      (`M3GNetInteraction`).
    - **Chemical Embedding**: Uses `PerElementEmbedding` to encode atomic
      features.

    Citation:

    .. code:: bibtex

        @article{Chen_Ong_2022,
            title        = {
                A universal graph deep learning interatomic potential for the
                periodic table
            },
            author       = {Chen, Chi and Ong, Shyue Ping},
            year         = 2022,
            month        = nov,
            journal      = {Nature Computational Science},
            volume       = 2,
            number       = 11,
            pages        = {718–728},
            doi          = {10.1038/s43588-022-00349-3},
            issn         = {2662-8457},
            url          = {https://www.nature.com/articles/s43588-022-00349-3},
            language     = {en}
        }
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        channels: int = 64,
        expansion_features: int = 50,
        layers: int = 3,
        expansion: type[DistanceExpansion] | None = None,
    ):
        super().__init__(
            cutoff=cutoff,
            implemented_properties=["local_energies"],
        )

        if expansion is None:
            expansion = GaussianSmearing

        self.chemical_embedding = PerElementEmbedding(channels)
        self.scaler = LocalEnergiesScaler()

        # Stack of M3GNet interaction blocks
        self.interactions = UniformModuleList(
            M3GNetInteraction(channels, expansion_features, cutoff, expansion)
            for _ in range(layers)
        )

        self.read_out = MLP(
            [channels, channels // 2, 1],
            activation=ShiftedSoftplus(),
        )

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        h = self.chemical_embedding(graph.Z)

        d = neighbour_distances(graph)

        for interaction in self.interactions:
            h = h + interaction(h, d, graph)

        local_energies = self.read_out(h).squeeze()
        local_energies = self.scaler(local_energies, graph)

        return {"local_energies": local_energies}
