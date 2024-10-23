from __future__ import annotations

from typing import Callable

import e3nn.util.jit
import torch
from e3nn import o3
from graph_pes.core import GraphPESModel
from graph_pes.graphs import keys
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import neighbour_distances, neighbour_vectors
from graph_pes.models.components.aggregation import NeighbourAggregationMode
from graph_pes.models.components.distances import (
    DistanceExpansion,
    PolynomialEnvelope,
    get_distance_expansion,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.models.e3nn.utils import (
    LinearReadOut,
    NonLinearReadOut,
    ReadOut,
    SphericalHarmonics,
)
from graph_pes.nn import (
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)

# class MACEInteraction(torch.nn.Module):
#     def forward(
#         self,
#         node_features: torch.Tensor,
#         node_attributes: torch.Tensor,
#         edge_features: torch.Tensor,
#         sph_harmonics: torch.Tensor,
#         graph: AtomicGraph,
#     ) -> torch.Tensor:
#         new_node_features, sc = self.


class MACELayer(torch.nn.Module):
    def __init__(
        self,
        l_max: int,
        correlation: int,
        n_node_attributes: int,
        node_feature_irreps: str,
        edge_feature_irreps: str,
        hidden_irreps: str,
        use_sc: bool,
        aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        self.interaction = MACEInteraction(
            l_max=l_max,
            correlation=correlation,
            n_node_attributes=n_node_attributes,
            aggregation=aggregation,
        )

        self.contraction = SymmetricContraction(
            n_node_attributes=n_node_attributes,
            node_feature_irreps=hidden_irreps,
            edge_feature_irreps=edge_feature_irreps,
            sph_irreps=sph_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
        )

        if use_sc:
            self.redisual_update = o3.FullyConnectedTensorProduct(
                node_feature_irreps,
                node_feature_irreps,
                hidden_irreps,
            )
        else:
            self.redisual_update = None

    def forward(
        self,
        node_features: torch.Tensor,
        node_attributes: torch.Tensor,
        edge_features: torch.Tensor,
        sph_harmonics: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        # interact
        node_features = self.interaction(
            node_features,
            node_attributes,
            edge_features,
            sph_harmonics,
            graph,
        )

        # contract
        node_features = self.contraction(node_features, node_attributes)

        # residual update
        if self.redisual_update is not None:
            update = self.redisual_update(node_features, node_attributes)
            node_features = node_features + update

        return node_features


@e3nn.util.jit.compile_mode("script")
class _BaseMACE(GraphPESModel):
    """
    The MACE architecture.
    """

    def __init__(
        self,
        # radial things
        cutoff: float,
        n_radial: int,
        radial_expansion_type: type[DistanceExpansion] | str,
        # node things
        n_node_attributes: int,
        node_attribute_generator: Callable[[torch.Tensor], torch.Tensor],
        hidden_irreps: str,
        # message passing
        layers: int,
        l_max: int,
        correlation: int,
        neighbour_aggregation: NeighbourAggregationMode,
        use_self_connection: bool,
    ):
        super().__init__(
            cutoff=cutoff,
            implemented_properties=["local_energies"],
        )

        # radial things
        self.spherical_harmonics = SphericalHarmonics(
            l_max, normalize=True, normalization="component"
        )
        if isinstance(radial_expansion_type, str):
            radial_expansion_type = get_distance_expansion(
                radial_expansion_type
            )
        self.radial_expansion = HaddamardProduct(
            radial_expansion_type(
                n_features=n_radial, cutoff=cutoff, trainable=True
            ),
            PolynomialEnvelope(cutoff=cutoff),
        )

        # node things
        self.node_attribute_generator = node_attribute_generator
        scalar_channels = o3.Irreps(hidden_irreps).count("0e")
        self.initial_node_embedding = PerElementEmbedding(scalar_channels)

        # message passing
        current_node_input_irreps = f"{scalar_channels}x0e"
        self.layers: UniformModuleList[MACELayer] = UniformModuleList([])
        edge_irreps = f"{n_radial}x0e"

        for _ in range(layers):
            layer = MACELayer(
                l_max=l_max,
                correlation=correlation,
                n_node_attributes=n_node_attributes,
                node_feature_irreps=current_node_input_irreps,
                edge_feature_irreps=edge_irreps,
                hidden_irreps=hidden_irreps,
                use_sc=use_self_connection,
                aggregation=neighbour_aggregation,
            )
            self.layers.append(layer)
            current_node_input_irreps = layer.node_output_irreps

        self.readouts: UniformModuleList[ReadOut] = UniformModuleList(
            [LinearReadOut(hidden_irreps) for _ in range(layers - 1)]
            + [NonLinearReadOut(hidden_irreps)]
        )

        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        # pre-compute some things
        Z = graph["atomic_numbers"]
        vectors = neighbour_vectors(graph)
        sph_harmonics = self.spherical_harmonics(vectors)
        edge_features = self.radial_expansion(
            neighbour_distances(graph).view(-1, 1)
        )
        node_attributes = self.node_attribute_generator(Z)

        # generate initial node features
        node_features = self.initial_node_embedding(Z)

        # update node features through message passing layers
        per_atom_energies = []
        for layer, readout in zip(self.layers, self.readouts):
            node_features = layer(
                node_features,
                node_attributes,
                edge_features,
                sph_harmonics,
                graph,
            )
            per_atom_energies.append(readout(node_features))

        # sum up the per-atom energies
        local_energies = torch.sum(
            torch.stack(per_atom_energies), dim=0
        ).squeeze()

        # return scaled local energy predictions
        return {"local_energies": self.scaler(local_energies, graph)}
