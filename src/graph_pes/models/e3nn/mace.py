from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, cast

import torch
from e3nn import o3
from graph_pes.core import GraphPESModel
from graph_pes.graphs import DEFAULT_CUTOFF, keys
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import (
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
)
from graph_pes.models.components.aggregation import (
    NeighbourAggregation,
    NeighbourAggregationMode,
)
from graph_pes.models.components.distances import (
    DistanceExpansion,
    PolynomialEnvelope,
    get_distance_expansion,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.models.e3nn.utils import (
    Contraction,
    LinearReadOut,
    NonLinearReadOut,
    ReadOut,
    SphericalHarmonics,
    as_irreps,
    build_limited_tensor_product,
    to_full_irreps,
)
from graph_pes.nn import (
    MLP,
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)


# @e3nn.util.jit.compile_mode("script")
class MACEInteraction(torch.nn.Module):
    """
    MACE interaction block
    """

    def __init__(
        self,
        # input nodes
        node_features_in: o3.Irreps,
        # input edges
        sph_harmonics: o3.Irreps,
        radial_basis_features: int,
        mlp_layers: list[int],
        # output nodes
        target_features_out: o3.Irreps,
        # other
        aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        self.pre_linear = o3.Linear(
            node_features_in,
            node_features_in,
            internal_weights=True,
            shared_weights=True,
        )

        self.tp = build_limited_tensor_product(
            node_features_in,
            sph_harmonics,
            [ir for _, ir in target_features_out],
        )
        mid_features = self.tp.irreps_out.simplify()

        self.weight_generator = MLP(
            [radial_basis_features] + mlp_layers + [self.tp.weight_numel],
            "SiLU",
            bias=False,
        )

        self.post_linear = o3.Linear(
            mid_features,
            target_features_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.aggregator = NeighbourAggregation.parse(aggregation)

        self.reshape = reshape_irreps(target_features_out)

        # book-keeping
        self.irreps_in = node_features_in
        self.irreps_out = target_features_out

    def forward(
        self,
        node_features: torch.Tensor,
        sph_harmonics: torch.Tensor,
        radial_basis: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        # pre-linear
        node_features = self.pre_linear(node_features)  # (N, a)

        # tensor product: mix node and edge features
        neighbour_features = index_over_neighbours(
            node_features, graph
        )  # (E, a)
        weights = self.weight_generator(radial_basis)  # (E, b)
        messages = self.tp(
            neighbour_features,
            sph_harmonics,
            weights,
        )  # (E, c)

        # aggregate
        total_message = self.aggregator(messages, graph)  # (N, c)

        # post-linear
        node_features = self.post_linear(total_message)  # (N, d)

        return self.reshape(node_features)


# @compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)


@dataclass
class NodeDescription:
    n_features: int
    n_attributes: int
    hidden_features: list[o3.Irrep]

    def hidden_irreps(self) -> o3.Irreps:
        return to_full_irreps(self.n_features, self.hidden_features)


class MACELayer(torch.nn.Module):
    def __init__(
        self,
        input_features: o3.Irreps,
        nodes: NodeDescription,
        correlation: int,
        sph_harmonics: o3.Irreps,
        radial_basis_features: int,
        mlp_layers: list[int],
        use_sc: bool,
        aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        target_mid_features = as_irreps(
            [(nodes.n_features, ir) for (_, ir) in sph_harmonics]
        )
        self.interaction = MACEInteraction(
            node_features_in=input_features,
            sph_harmonics=sph_harmonics,
            radial_basis_features=radial_basis_features,
            mlp_layers=mlp_layers,
            target_features_out=target_mid_features,
            aggregation=aggregation,
        )
        actual_mid_features = [ir for _, ir in self.interaction.irreps_out]

        # Initialize contractions directly in MACELayer
        self.contractions = UniformModuleList(
            [
                Contraction(
                    num_features=nodes.n_features,
                    n_node_attributes=nodes.n_attributes,
                    irrep_s_in=actual_mid_features,
                    irrep_out=target_irrep,
                    correlation=correlation,
                )
                for target_irrep in nodes.hidden_features
            ]
        )

        if use_sc:
            self.redisual_update = o3.FullyConnectedTensorProduct(
                irreps_in1=input_features,
                irreps_in2=o3.Irreps(f"{nodes.n_attributes}x0e"),
                irreps_out=nodes.hidden_irreps(),
            )
        else:
            self.redisual_update = None

        self.post_linear = o3.Linear(
            nodes.hidden_irreps(),
            nodes.hidden_irreps(),
            internal_weights=True,
            shared_weights=True,
        )

        # book-keeping
        self.irreps_in = input_features
        self.irreps_out = nodes.hidden_irreps()

    def forward(
        self,
        node_features: torch.Tensor,  # (N, irreps_in)
        node_attributes: torch.Tensor,
        sph_harmonics: torch.Tensor,
        radial_basis: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        # interact
        internal_node_features = self.interaction(
            node_features,
            sph_harmonics,
            radial_basis,
            graph,
        )  # (N, irreps_mid)

        # contract using the contractions directly
        contracted_features = torch.cat(
            [
                contraction(internal_node_features, node_attributes)
                for contraction in self.contractions
            ],
            dim=-1,
        )  # (N, irreps_out)

        # residual update
        if self.redisual_update is not None:
            update = self.redisual_update(
                node_features,
                node_attributes,
            )  # (N, irreps_out)
            contracted_features = contracted_features + update

        # linear update
        node_features = self.post_linear(contracted_features)  # (N, irreps_out)

        return node_features


# @e3nn.util.jit.compile_mode("script")
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
        mlp_layers: list[int],
        # node things
        nodes: NodeDescription,
        node_attribute_generator: Callable[[torch.Tensor], torch.Tensor],
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

        if o3.Irrep("0e") not in nodes.hidden_features:
            raise ValueError("MACE requires a 0e hidden feature")

        # radial things
        sph_harmonics = cast(o3.Irreps, o3.Irreps.spherical_harmonics(l_max))
        self.spherical_harmonics = SphericalHarmonics(
            sph_harmonics,
            normalize=True,
            normalization="component",
        )
        if isinstance(radial_expansion_type, str):
            radial_expansion_type = get_distance_expansion(
                radial_expansion_type
            )
        self.radial_expansion = HaddamardProduct(
            radial_expansion_type(
                n_features=n_radial, cutoff=cutoff, trainable=True
            ),
            PolynomialEnvelope(cutoff=cutoff, p=5),
        )

        # node things
        self.node_attribute_generator = node_attribute_generator
        self.initial_node_embedding = PerElementEmbedding(nodes.n_features)

        # message passing
        current_node_irreps = as_irreps(nodes.n_features * o3.Irrep("0e"))
        self.layers: UniformModuleList[MACELayer] = UniformModuleList([])

        for _ in range(layers):
            layer = MACELayer(
                input_features=current_node_irreps,
                nodes=nodes,
                correlation=correlation,
                sph_harmonics=sph_harmonics,
                radial_basis_features=n_radial,
                mlp_layers=mlp_layers,
                use_sc=use_self_connection,
                aggregation=neighbour_aggregation,
            )
            self.layers.append(layer)
            current_node_irreps = layer.irreps_out

        self.readouts: UniformModuleList[ReadOut] = UniformModuleList(
            [LinearReadOut(nodes.hidden_irreps()) for _ in range(layers - 1)]
            + [NonLinearReadOut(nodes.hidden_irreps())]
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
                sph_harmonics,
                edge_features,
                graph,
            )
            per_atom_energies.append(readout(node_features))

        # sum up the per-atom energies
        local_energies = torch.sum(
            torch.stack(per_atom_energies), dim=0
        ).squeeze()

        # return scaled local energy predictions
        return {"local_energies": self.scaler(local_energies, graph)}


def parse_irreps(irreps: str | list[str]) -> list[o3.Irrep]:
    if isinstance(irreps, str):
        try:
            return [o3.Irrep(ir) for ir in irreps.split(" + ")]
        except ValueError:
            raise ValueError(
                f"Unable to parse {irreps} as irreps. "
                "Expected a string of the form '0e + 1o'"
            ) from None
    try:
        return [o3.Irrep(ir) for ir in irreps]
    except ValueError:
        raise ValueError(
            f"Unable to parse {irreps} as irreps. "
            "Expected a list of strings of the form ['0e', '1o']"
        ) from None


DEFAULT_MLP_LAYERS: Final[list[int]] = [16, 16]


class MACE(_BaseMACE):
    def __init__(
        self,
        elements: list[str],
        cutoff: float = DEFAULT_CUTOFF,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        mlp_layers: list[int] = DEFAULT_MLP_LAYERS,
        n_features: int = 16,
        hidden_irreps: str | list[str] = "0e + 1o",
        n_radial: int = 8,
        l_max: int = 2,
        layers: int = 1,
        correlation: int = 3,
        aggregation: NeighbourAggregationMode = "constant_fixed",
        self_connection: bool = True,
    ):
        Z_embedding = AtomicOneHot(elements)
        Z_dim = len(elements)
        hidden_irrep_s = parse_irreps(hidden_irreps)
        nodes = NodeDescription(
            n_features=n_features,
            n_attributes=Z_dim,
            hidden_features=hidden_irrep_s,
        )

        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            mlp_layers=mlp_layers,
            nodes=nodes,
            node_attribute_generator=Z_embedding,
            l_max=l_max,
            layers=layers,
            correlation=correlation,
            neighbour_aggregation=aggregation,
            use_self_connection=self_connection,
        )


class ZEmbeddingMACE(_BaseMACE):
    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        z_embed_dim: int = 4,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        mlp_layers: list[int] = DEFAULT_MLP_LAYERS,
        n_radial: int = 8,
        n_features: int = 16,
        hidden_irreps: str | list[str] = "0e + 1o",
        l_max: int = 2,
        layers: int = 1,
        correlation: int = 3,
        aggregation: NeighbourAggregationMode = "constant_fixed",
        self_connection: bool = True,
    ):
        Z_embedding = PerElementEmbedding(z_embed_dim)
        hidden_irrep_s = parse_irreps(hidden_irreps)
        nodes = NodeDescription(
            n_features=n_features,
            n_attributes=z_embed_dim,
            hidden_features=hidden_irrep_s,
        )

        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            mlp_layers=mlp_layers,
            nodes=nodes,
            node_attribute_generator=Z_embedding,
            l_max=l_max,
            layers=layers,
            correlation=correlation,
            neighbour_aggregation=aggregation,
            use_self_connection=self_connection,
        )
