import torch
from torch import Tensor

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, neighbour_vectors
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
# TODO: rotational augmentations


class OrbEncoder(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        channels: int,
        radial_features: int,
        l_max: int,
        edge_outer_product: bool,
        mlp_layers: int,
        mlp_hidden_dim: int,
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
            activation=ShiftedSoftplus(),
        )
        self.edge_layer_norm = torch.nn.LayerNorm(channels)

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
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(
        self, node_emb: Tensor, edge_emb: Tensor, graph: AtomicGraph
    ) -> tuple[Tensor, Tensor]:
        return node_emb, edge_emb


class Orb(GraphPESModel):
    def __init__(
        self,
        cutoff: float,
        conservative: bool = False,
        channels: int = 128,
        layers: int = 5,
        radial_features: int = 8,
        mlp_layers: int = 2,
        mlp_hidden_dim: int = 128,
        l_max: int = 3,
        edge_outer_product: bool = True,
    ):
        props: list[PropertyKey] = (
            ["local_energies"] if conservative else ["local_energies", "forces"]
        )
        super().__init__(implemented_properties=props, cutoff=cutoff)

        # backbone
        self._encoder = OrbEncoder(
            cutoff,
            channels,
            radial_features,
            l_max,
            edge_outer_product,
            mlp_layers,
            mlp_hidden_dim,
        )
        self._gnn_layers = UniformModuleList(
            [OrbMessagePassingLayer(channels) for _ in range(layers)]
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
