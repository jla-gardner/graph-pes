from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    edge_wise_softmax,
    keep_at_most_k_neighbours,
    neighbour_distances,
    neighbour_vectors,
    remove_mean_and_net_torque,
    sum_over_central_atom_index,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.distances import Bessel, PolynomialEnvelope
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.utils.nn import (
    MLP,
    PerElementEmbedding,
    ShiftedSoftplus,
    UniformModuleList,
)

from .e3nn.utils import SphericalHarmonics

# TODO: penalise rotational grad


NormType = Literal["layer", "rms"]
AttentionGate = Literal["sigmoid", "softmax"]


def get_norm(norm_type: NormType):
    if norm_type == "layer":
        return torch.nn.LayerNorm
    elif norm_type == "rms":
        return torch.nn.RMSNorm
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class OrbEncoder(torch.nn.Module):
    """Generates node and edge features embeddings for an atomic graph."""

    def __init__(
        self,
        cutoff: float,
        channels: int,
        radial_features: int,
        l_max: int,
        edge_outer_product: bool,
        mlp_layers: int,
        mlp_hidden_dim: int,
        activation: str,
        norm_type: NormType,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.channels = channels
        self.edge_outer_product = edge_outer_product

        # nodes
        self.Z_embedding = PerElementEmbedding(channels)
        self.Z_layer_norm = torch.nn.LayerNorm(channels)

        # edges
        self.rbf = Bessel(radial_features, cutoff, trainable=False)
        self.envelope = PolynomialEnvelope(p=4, cutoff=cutoff)
        self.sh = SphericalHarmonics(
            [l for l in range(l_max + 1)],
            normalize=True,
            normalization="component",
        )
        sh_dim: int = self.sh.irreps_out.dim  # type: ignore
        self.edge_dim = (
            radial_features * sh_dim
            if edge_outer_product
            else radial_features + sh_dim
        )
        self.edge_mlp = MLP(
            [self.edge_dim] + [mlp_hidden_dim] * mlp_layers + [channels],
            activation,
        )
        self.edge_layer_norm = get_norm(norm_type)(channels)

    def forward(self, graph: AtomicGraph) -> tuple[Tensor, Tensor]:
        node_emb = self.Z_layer_norm(self.Z_embedding(graph.Z))

        # featurise angles
        v = neighbour_vectors(graph)
        sh_emb = self.sh(v)

        # featurise distances
        d = torch.linalg.norm(v, dim=-1)
        rbf_emb = self.rbf(d)

        # combine
        if self.edge_outer_product:
            edge_emb = rbf_emb[:, :, None] * sh_emb[:, None, :]
        else:
            edge_emb = torch.cat([rbf_emb, sh_emb], dim=1)
        edge_emb = edge_emb.view(-1, self.edge_dim)

        # smooth cutoff
        c = self.envelope(d)
        edge_feats = edge_emb * c.unsqueeze(-1)

        # mlp
        edge_emb = self.edge_layer_norm(self.edge_mlp(edge_feats))

        return node_emb, edge_emb


class OrbMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        channels: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
        activation: str,
        norm_type: NormType,
        attention_gate: AttentionGate,
        distance_smoothing: bool,
    ):
        super().__init__()

        self.node_mlp = torch.nn.Sequential(
            MLP(
                [channels * 3] + [mlp_hidden_dim] * mlp_layers + [channels],
                activation,
            ),
            get_norm(norm_type)(channels),
        )

        self.edge_mlp = torch.nn.Sequential(
            MLP(
                [channels * 3] + [mlp_hidden_dim] * mlp_layers + [channels],
                activation,
            ),
            get_norm(norm_type)(channels),
        )

        self.receive_attn = torch.nn.Linear(channels, 1)
        self.send_attn = torch.nn.Linear(channels, 1)

        self.attention_gate = attention_gate

        if distance_smoothing:
            self.envelope = PolynomialEnvelope(p=4, cutoff=cutoff)
        else:
            self.envelope = None

    def forward(
        self,
        node_emb: Tensor,  # (N, C)
        edge_emb: Tensor,  # (E, C)
        graph: AtomicGraph,
    ) -> tuple[Tensor, Tensor]:
        # calculate per-edge attention weights based on both
        # senders and receivers
        if self.attention_gate == "softmax":
            receive_attn_weights = edge_wise_softmax(
                self.receive_attn(edge_emb), graph, aggregation="receivers"
            )
            send_attn_weights = edge_wise_softmax(
                self.send_attn(edge_emb), graph, aggregation="senders"
            )
        elif self.attention_gate == "sigmoid":
            receive_attn_weights = torch.sigmoid(self.receive_attn(edge_emb))
            send_attn_weights = torch.sigmoid(self.send_attn(edge_emb))
        else:
            raise ValueError(f"Unknown attention gate: {self.attention_gate}")

        # optionally decay these weights near the cutoff
        if self.envelope is not None:
            envelope = self.envelope(neighbour_distances(graph)).unsqueeze(-1)
            receive_attn_weights = receive_attn_weights * envelope
            send_attn_weights = send_attn_weights * envelope

        # generate new edge features
        new_edge_features = torch.cat(
            [
                edge_emb,
                node_emb[graph.neighbour_list[0]],
                node_emb[graph.neighbour_list[1]],
            ],
            dim=1,
        )
        new_edge_features = self.edge_mlp(new_edge_features)

        #  generate new node features from attention weights
        senders, receivers = graph.neighbour_list[0], graph.neighbour_list[1]
        sent_total_message = sum_over_central_atom_index(  # (N, C)
            new_edge_features * send_attn_weights, senders, graph
        )
        received_total_message = sum_over_central_atom_index(  # (N, C)
            new_edge_features * receive_attn_weights, receivers, graph
        )
        new_node_features = torch.cat(
            [node_emb, sent_total_message, received_total_message],
            dim=1,
        )
        new_node_features = self.node_mlp(new_node_features)

        # residual connection
        node_emb = node_emb + new_node_features
        edge_emb = edge_emb + new_edge_features

        return node_emb, edge_emb


class Orb(GraphPESModel):
    r"""
    The `Orb-v3 <https://arxiv.org/abs/2504.06231>`__ architecture.

    Citation:

    .. code-block:: bibtex

        @misc{Rhodes-25-04,
            title = {Orb-v3: Atomistic Simulation at Scale},
            author = {
                Rhodes, Benjamin and Vandenhaute, Sander 
                and {\v S}imkus, Vaidotas and Gin, James and Godwin, Jonathan 
                and Duignan, Tim and Neumann, Mark
            },
            year = {2025},
            publisher = {arXiv},
            doi = {10.48550/arXiv.2504.06231},
        }

    Parameters
    ----------
    cutoff
        The cutoff radius for interatomic interactions.
    conservative
        If ``True``, the model will generate force predictions as the negative
        gradient of the energy with respect to atomic positions. If ``False``,
        the model will have a separate force prediction head.
    channels
        The number of channels in the model.
    layers
        The number of message passing layers.
    radial_features
        The number of radial basis functions to use.
    mlp_layers
        The number of layers in the MLPs.
    mlp_hidden_dim
        The hidden dimension of the MLPs.
    l_max
        The maximum degree of spherical harmonics to use.
    edge_outer_product
        If ``True``, use the outer product of radial and angular features for
        edge embeddings. If ``False``, concatenate radial and angular features.
    activation
        The activation function to use in the MLPs.
    norm_type
        The type of normalization to use in the MLPs. Either ``"layer"`` for
        :class:`torch.nn.LayerNorm` or ``"rms"`` for :class:`torch.nn.RMSNorm`.
    attention_gate
        The type of attention gating to use in message passing layers. Either
        ``"sigmoid"`` for element-wise sigmoid gating or ``"softmax"`` for
        normalising attention weights over neighbours.
    distance_smoothing
        If ``True``, apply a polynomial envelope to attention weights based on
        interatomic distances. If ``False``, do not apply any distance-based
        smoothing.
    max_neighbours
        If set, limit the number of neighbours per atom to this value by
        keeping only the closest ones.
    """
    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        conservative: bool = False,
        channels: int = 256,
        layers: int = 5,
        radial_features: int = 8,
        mlp_layers: int = 2,
        mlp_hidden_dim: int = 1024,
        l_max: int = 3,
        edge_outer_product: bool = True,
        activation: str = "silu",
        norm_type: NormType = "layer",
        attention_gate: AttentionGate = "sigmoid",
        distance_smoothing: bool = True,
        max_neighbours: int | None = None,
    ):
        props: list[PropertyKey] = (
            ["local_energies"] if conservative else ["local_energies", "forces"]
        )
        super().__init__(implemented_properties=props, cutoff=cutoff)

        self.max_neighbours = max_neighbours

        # backbone
        self._encoder = OrbEncoder(
            cutoff=cutoff,
            channels=channels,
            radial_features=radial_features,
            l_max=l_max,
            edge_outer_product=edge_outer_product,
            mlp_layers=mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            activation=activation,
            norm_type=norm_type,
        )
        self._gnn_layers = UniformModuleList(
            [
                OrbMessagePassingLayer(
                    channels=channels,
                    mlp_layers=mlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                    activation=activation,
                    norm_type=norm_type,
                    attention_gate=attention_gate,
                    distance_smoothing=distance_smoothing,
                    cutoff=cutoff,
                )
                for _ in range(layers)
            ]
        )

        # readouts
        self._energy_readout = MLP(
            [channels] + [mlp_hidden_dim] * mlp_layers + [1],
            activation=ShiftedSoftplus(),
        )
        self.scaler = LocalEnergiesScaler()
        if conservative:
            self._force_readout = None
        else:
            self._force_readout = MLP(
                [channels] + [mlp_hidden_dim] * mlp_layers + [3],
                activation=ShiftedSoftplus(),
                bias=False,
            )

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, Tensor]:
        if self.max_neighbours is not None:
            graph = keep_at_most_k_neighbours(graph, self.max_neighbours)

        # embed the graph
        node_emb, edge_emb = self._encoder(graph)

        # message passing
        for layer in self._gnn_layers:
            node_emb, edge_emb = layer(node_emb, edge_emb, graph)

        # readout
        raw_energies = self._energy_readout(node_emb)
        preds: dict[PropertyKey, Tensor] = {
            "local_energies": self.scaler(raw_energies, graph)
        }

        if self._force_readout is not None:
            raw_forces = self._force_readout(node_emb)
            preds["forces"] = remove_mean_and_net_torque(raw_forces, graph)

        return preds
