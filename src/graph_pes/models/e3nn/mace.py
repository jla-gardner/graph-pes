from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, cast

import torch
import torch.fx
from e3nn import o3
from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
)
from graph_pes.graph_pes_model import GraphPESModel
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
from graph_pes.models.e3nn.mace_utils import (
    Contraction,
    ContractionConfig,
    UnflattenIrreps,
    parse_irreps,
)
from graph_pes.models.e3nn.utils import (
    LinearReadOut,
    NonLinearReadOut,
    ReadOut,
    SphericalHarmonics,
    as_irreps,
    build_limited_tensor_product,
    to_full_irreps,
)
from graph_pes.utils.nn import (
    MLP,
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)


class MACEInteraction(torch.nn.Module):
    """
    The MACE interaction block.

    Generates new node embeddings from the old node embeddings and the
    spherical harmonic expansion and mangitudes of the neighbour vectors.
    """

    def __init__(
        self,
        # input nodes
        irreps_in: list[o3.Irrep],
        channels: int,
        # input edges
        sph_harmonics: o3.Irreps,
        radial_basis_features: int,
        mlp_layers: list[int],
        # other
        aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        irreps_out = [ir for _, ir in sph_harmonics]

        features_in = as_irreps([(channels, ir) for ir in irreps_in])
        self.pre_linear = o3.Linear(
            features_in,
            features_in,
            internal_weights=True,
            shared_weights=True,
        )

        self.tp = build_limited_tensor_product(
            features_in,
            sph_harmonics,
            irreps_out,
        )
        mid_features = self.tp.irreps_out.simplify()
        assert all(ir in mid_features for ir in irreps_out)

        self.weight_generator = MLP(
            [radial_basis_features] + mlp_layers + [self.tp.weight_numel],
            "SiLU",
            bias=False,
        )

        features_out = as_irreps([(channels, ir) for (_, ir) in sph_harmonics])
        self.post_linear = o3.Linear(
            mid_features,
            features_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.aggregator = NeighbourAggregation.parse(aggregation)

        self.reshape = UnflattenIrreps(irreps_out, channels)

        # book-keeping
        self.irreps_in = features_in
        self.irreps_out = features_out

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

        return self.reshape(node_features)  # (N, channels, d')


@dataclass
class NodeDescription:
    channels: int
    attributes: int
    hidden_features: list[o3.Irrep]

    def hidden_irreps(self) -> o3.Irreps:
        return to_full_irreps(self.channels, self.hidden_features)


class MACELayer(torch.nn.Module):
    def __init__(
        self,
        irreps_in: list[o3.Irrep],
        nodes: NodeDescription,
        correlation: int,
        sph_harmonics: o3.Irreps,
        radial_basis_features: int,
        mlp_layers: list[int],
        use_sc: bool,
        aggregation: NeighbourAggregationMode,
    ):
        super().__init__()

        self.interaction = MACEInteraction(
            irreps_in=irreps_in,
            channels=nodes.channels,
            sph_harmonics=sph_harmonics,
            radial_basis_features=radial_basis_features,
            mlp_layers=mlp_layers,
            aggregation=aggregation,
        )
        actual_mid_features = [ir for _, ir in self.interaction.irreps_out]

        self.contractions = UniformModuleList(
            [
                Contraction(
                    config=ContractionConfig(
                        num_features=nodes.channels,
                        n_node_attributes=nodes.attributes,
                        irrep_s_in=actual_mid_features,
                        irrep_out=target_irrep,
                    ),
                    correlation=correlation,
                )
                for target_irrep in nodes.hidden_features
            ]
        )

        if use_sc:
            self.residual_update = o3.FullyConnectedTensorProduct(
                irreps_in1=[(nodes.channels, ir) for ir in irreps_in],
                irreps_in2=o3.Irreps(f"{nodes.attributes}x0e"),
                irreps_out=nodes.hidden_irreps(),
            )
        else:
            self.residual_update = None

        self.post_linear = o3.Linear(
            nodes.hidden_irreps(),
            nodes.hidden_irreps(),
            internal_weights=True,
            shared_weights=True,
        )

        # book-keeping
        self.irreps_in = irreps_in
        self.irreps_out = nodes.hidden_irreps()

    def forward(
        self,
        node_features: torch.Tensor,
        node_attributes: torch.Tensor,
        sph_harmonics: torch.Tensor,
        radial_basis: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        # A MACE layer operates on:
        # - node features with multiplicity M, e.g. M=16: 16x0e + 16x1o
        # - node attributes with multiplicity A e.g. A=5: 5x0e
        # - spherical harmonics up to l_max, e.g. l_max=2: 1x0e + 1x1o + 1x2e

        # interact
        internal_node_features = self.interaction(
            node_features,
            sph_harmonics,
            radial_basis,
            graph,
        )  # (N, M, irreps)

        # contract using the contractions directly
        contracted_features = torch.cat(
            [
                contraction(internal_node_features, node_attributes)
                for contraction in self.contractions
            ],
            dim=-1,
        )  # (N, irreps_out)

        # residual update
        if self.residual_update is not None:
            update = self.residual_update(
                node_features,
                node_attributes,
            )  # (N, irreps_out)
            contracted_features = contracted_features + update

        # linear update
        node_features = self.post_linear(contracted_features)  # (N, irreps_out)

        return node_features


class _BaseMACE(GraphPESModel):
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
            raise ValueError("MACE requires a `0e` hidden feature")

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
        self.initial_node_embedding = PerElementEmbedding(nodes.channels)

        # message passing
        current_node_irreps = [o3.Irrep("0e")]
        self.layers: UniformModuleList[MACELayer] = UniformModuleList([])

        for _ in range(layers):
            layer = MACELayer(
                irreps_in=current_node_irreps,
                nodes=nodes,
                correlation=correlation,
                sph_harmonics=sph_harmonics,
                radial_basis_features=n_radial,
                mlp_layers=mlp_layers,
                use_sc=use_self_connection,
                aggregation=neighbour_aggregation,
            )
            self.layers.append(layer)
            current_node_irreps = [ir for _, ir in layer.irreps_out]

        self.readouts: UniformModuleList[ReadOut] = UniformModuleList(
            [LinearReadOut(nodes.hidden_irreps()) for _ in range(layers - 1)]
            + [NonLinearReadOut(nodes.hidden_irreps())]
        )

        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        # pre-compute some things
        vectors = neighbour_vectors(graph)
        sph_harmonics = self.spherical_harmonics(vectors)
        edge_features = self.radial_expansion(
            neighbour_distances(graph).view(-1, 1)
        )
        node_attributes = self.node_attribute_generator(graph.Z)

        # generate initial node features
        node_features = self.initial_node_embedding(graph.Z)

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


DEFAULT_MLP_LAYERS: Final[list[int]] = [16, 16]


class MACE(_BaseMACE):
    r"""
    The `MACE <https://arxiv.org/abs/2206.07697>`__ architecture.

    One-hot encodings of the atomic numbers are used to condition the
    ``TensorProduct`` update in the residual connection of the message passing
    layers, as well as the contractions in the message passing layers.

    Please cite the following if you use this model in your research:

    .. code-block:: bibtex

        @misc{Batatia2022MACE,
            title = {
                MACE: Higher Order Equivariant Message Passing
                Neural Networks for Fast and Accurate Force Fields
            },
            author = {
                Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and
                Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor
            },
            year = {2022},
            doi = {10.48550/arXiv.2206.07697},
        }

    Parameters
    ----------
    elements
        list of elements that this MACE model will be able to handle.
    cutoff
        radial cutoff (in Å) for the radial expansion (and message passing)
    n_radial
        number of bases to expand the radial distances into
    radial_expansion_type
        type of radial expansion to use. See :class:`~graph_pes.models.components.distances.DistanceExpansion`
        for available options
    mlp_layers
        the widths of layers in the MLPs that map the radial basis functions
        to the weights of the interactions' tensor products
    channels
        the multiplicity of the node features corresponding to each irrep
        specified in ``hidden_irreps``
    hidden_irreps
        string representations of the :class:`e3nn.o3.Irrep`\ s to use
        for representing the node features between each message passing layer
    l_max
        the highest order to consider in:
        * the spherical harmonics expansion of the neighbour vectors
        * the irreps of node features used within each message passing layer
    correlation
        maximum correlation (body-order) of the messages
    layers
        number of message passing layers
    aggregation
        the type of aggregation to use when creating total messages from
        neigbour messages :math:`m_{j \rightarrow i}`
    self_connection
        whether to use self-connections in the message passing layers

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> from graph_pes.models import MACE
        >>> model = MACE(
        ...     elements=["H", "C", "N", "O"],
        ...     cutoff=5.0,
        ...     channels=16,
        ...     radial_expansion_type="Bessel",
        ... )

    Specification in a YAML file:

    .. code-block:: yaml

        model:
            graph_pes.models.MACE:
                elements: [H, C, N, O]
                cutoff: 5.0
                radial_expansion_type: GaussianSmearing
                mlp_layers: [16, 16]

    """  # noqa: E501

    def __init__(
        self,
        elements: list[str],
        # radial things
        cutoff: float = DEFAULT_CUTOFF,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        mlp_layers: list[int] = DEFAULT_MLP_LAYERS,
        # node things
        channels: int = 128,
        hidden_irreps: str | list[str] = "0e + 1o",
        # message passing things
        l_max: int = 3,
        layers: int = 2,
        correlation: int = 3,
        aggregation: NeighbourAggregationMode = "constant_fixed",
        self_connection: bool = True,
    ):
        Z_embedding = AtomicOneHot(elements)
        Z_dim = len(elements)
        hidden_irrep_s = parse_irreps(hidden_irreps)
        nodes = NodeDescription(
            channels=channels,
            attributes=Z_dim,
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
    """
    A variant of MACE that uses a fixed-size (``z_embed_dim``) per-element
    embedding of the atomic numbers to condition the ``TensorProduct`` update
    in the residual connection of the message passing layers, as well as the
    contractions in the message passing layers.

    Please cite the following if you use this model in your research:

    .. code-block:: bibtex

        @misc{Batatia2022MACE,
            title = {
                MACE: Higher Order Equivariant Message Passing
                Neural Networks for Fast and Accurate Force Fields
            },
            author = {
                Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and
                Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor
            },
            year = {2022},
            doi = {10.48550/arXiv.2206.07697},
        }

    All paramters are identical to :class:`~graph_pes.models.MACE`, except for the following:

    - ``elements`` is not required or used here
    - ``z_embed_dim`` controls size of the per-element embedding
    """  # noqa: E501

    def __init__(
        self,
        z_embed_dim: int = 4,
        # radial things
        cutoff: float = DEFAULT_CUTOFF,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] | str = "Bessel",
        mlp_layers: list[int] = DEFAULT_MLP_LAYERS,
        # node things
        channels: int = 128,
        hidden_irreps: str | list[str] = "0e + 1o",
        # message passing things
        l_max: int = 3,
        layers: int = 2,
        correlation: int = 3,
        aggregation: NeighbourAggregationMode = "constant_fixed",
        self_connection: bool = True,
    ):
        Z_embedding = PerElementEmbedding(z_embed_dim)
        hidden_irrep_s = parse_irreps(hidden_irreps)
        nodes = NodeDescription(
            channels=channels,
            attributes=z_embed_dim,
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
