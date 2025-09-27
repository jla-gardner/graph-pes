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
)
from graph_pes.utils.threebody import triplet_bond_descriptors
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.distances import (
    DistanceExpansion,
    GaussianSmearing,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.utils.nn import (
    PerElementEmbedding,
    UniformModuleList,
)

__author__ = "Stefan Bringuier"
__email__ = "stefan.bringuier@gmail.com"
__license__ = "See LICENSE"


class GatedMLP(torch.nn.Module):
    """
    Gated MLP implementation following the original M3GNet architecture.
    """

    def __init__(
        self,
        neurons: list[int],
        activation: str = "swish",
        use_bias: bool = True,
    ):
        super().__init__()

        # Main MLP path
        layers = []
        for i in range(len(neurons) - 1):
            layers.append(torch.nn.Linear(neurons[i], neurons[i + 1], bias=use_bias))
            if i < len(neurons) - 2:  # last layer is linear
                if activation == "swish":
                    layers.append(torch.nn.SiLU())
                else:
                    layers.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*layers)

        # Gated path
        gate_layers = []
        for i in range(len(neurons) - 1):
            gate_layers.append(
                torch.nn.Linear(neurons[i], neurons[i + 1], bias=use_bias)
            )
            if i < len(neurons) - 2:
                if activation == "swish":
                    gate_layers.append(torch.nn.SiLU())
                else:
                    gate_layers.append(torch.nn.ReLU())
        gate_layers.append(torch.nn.Sigmoid())  # Sigmoid at end
        self.gate = torch.nn.Sequential(*gate_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.gate(x)


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

    Key Tensor Shapes:
    - Input node features: [n_atoms, channels]
    - Radial basis features: [n_edges, expansion_features]
    - Two-body messages: [n_edges, channels]
    - Three-body messages: [n_triplets, channels]
    - Output node features: [n_atoms, channels]
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

        # [n_edges, expansion_features] -> [n_edges, channels]
        self.two_body_net = GatedMLP(
            neurons=[expansion_features, expansion_features, channels],
            activation="swish",
            use_bias=False,
        )

        # [n_triplets, 3] -> [n_triplets, channels]
        self.three_body_net = GatedMLP(
            neurons=[3, 64, channels],  # Original architecture: [3->64->channels]?
            activation="swish",
            use_bias=False,
        )

        self.post_linear = torch.nn.Linear(channels, channels, bias=False)

    def _compute_three_body_messages(
        self,
        node_features: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        """
        Compute three-body messages for each edge in the graph.

        Args:
            node_features (torch.Tensor): [n_atoms, channels]
            graph (AtomicGraph): Graph object containing the atomic graph structure

        Returns:
            torch.Tensor: [n_edges, channels]
        """
        triplet_idxs, angles, r_ij, r_ik = triplet_bond_descriptors(graph)

        if triplet_idxs.shape[0] == 0:
            return torch.zeros_like(node_features[graph.neighbour_list[0]])

        # [n_triplets, 3] -> [n_triplets, channels]
        three_body_features = torch.stack([r_ij, r_ik, torch.cos(angles)], dim=-1)
        three_body_weights = self.three_body_net(three_body_features)

        # [n_triplets, channels]
        neighbor_features = node_features[triplet_idxs[:, 1]]

        # [n_triplets, channels]
        messages = neighbor_features * three_body_weights

        # Aggregate messages for each edge
        # [n_edges, channels]
        edge_messages = torch.zeros_like(node_features[graph.neighbour_list[0]])
        edge_messages.index_add_(0, triplet_idxs[:, 1], messages)

        return edge_messages

    def forward(
        self,
        features: torch.Tensor,
        neighbour_distances: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        """
        Forward pass of the M3GNet interaction block.

        Args:
            features (torch.Tensor): [n_atoms, channels]
            neighbor_distances (torch.Tensor): [n_edges]
            graph (AtomicGraph): Graph object containing the atomic graph structure

        Returns:
            torch.Tensor: [n_atoms, channels]
        """
        # [n_atoms, channels] -> [n_atoms, channels]
        h = self.pre_linear(features)

        # [n_edges,] -> [n_edges, expansion_features]
        radial_basis = self.radial_basis(neighbour_distances.unsqueeze(-1))

        cutoff_mask = (neighbour_distances < self.cutoff).float()
        # Smooth cutoff values [n_edges,]
        cutoff_values = 0.5 * (1 + torch.cos(np.pi * neighbour_distances / self.cutoff))
        # [n_edges, expansion_features]
        radial_basis = (
            radial_basis * cutoff_values.unsqueeze(-1) * cutoff_mask.unsqueeze(-1)
        )

        # Messages
        # [n_edges, channels] -> [n_edges, channels]
        neighbor_h = index_over_neighbours(h, graph)
        two_body_messages = neighbor_h * self.two_body_net(radial_basis)
        three_body_messages = self._compute_three_body_messages(h, graph)

        # Message Aggregation
        # [n_edges, channels] -> [n_atoms, channels]
        messages = two_body_messages + three_body_messages
        h = sum_over_neighbours(messages, graph)

        # [n_atoms, channels]
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
            pages        = {718-728},
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
        # All are [n_atoms, channels]
        self.interactions = UniformModuleList(
            M3GNetInteraction(channels, expansion_features, cutoff, expansion)
            for _ in range(layers)
        )

        # Original M3GNet readout: [64->32->1] ?
        self.read_out = GatedMLP(
            neurons=[channels, 32, 1],
            activation="swish",
            use_bias=True,
        )

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        # [n_atoms,] -> [n_atoms, channels]
        h = self.chemical_embedding(graph.Z)

        # [n_edges,]
        d = neighbour_distances(graph)

        # Update node features, [n_atoms, channels]
        for interaction in self.interactions:
            h = h + interaction(h, d, graph)

        # Local energies from readout
        # [n_atoms, channels] -> [n_atoms, 1] -> [n_atoms,]
        local_energies = self.read_out(h).squeeze()
        local_energies = self.scaler(local_energies, graph)

        return {"local_energies": local_energies}
