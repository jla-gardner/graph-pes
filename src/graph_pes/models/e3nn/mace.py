from __future__ import annotations

from typing import Callable, Union

import e3nn.util.jit
import torch
from e3nn import o3
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import neighbour_distances, neighbour_vectors
from graph_pes.models.distances import (
    Bessel,
    DistanceExpansion,
    PolynomialEnvelope,
)
from graph_pes.models.scaling import AutoScaledPESModel
from graph_pes.nn import (
    AtomicOneHot,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)
from mace_layer import MACE_layer


class LinearReadOut(o3.Linear):
    def __init__(self, input_irreps: str):
        super().__init__(input_irreps, "1x0e")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class NonLinearReadOut(torch.nn.Module):
    def __init__(self, input_irreps: str):
        super().__init__()
        hidden_dim = o3.Irreps(input_irreps).count(o3.Irrep("0e"))
        self.layers = torch.nn.Sequential(
            o3.Linear(input_irreps, f"{hidden_dim}x0e"),
            torch.nn.SiLU(),
            o3.Linear(f"{hidden_dim}x0e", "1x0e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


ReadOut = Union[LinearReadOut, NonLinearReadOut]


@e3nn.util.jit.compile_mode("script")
class _BaseMACE(AutoScaledPESModel):
    def __init__(
        self,
        # radial things
        cutoff: float,
        n_radial: int,
        radial_expansion_type: type[DistanceExpansion],
        # node attributes
        z_embed_dim: int,
        z_embedding: Callable[[torch.Tensor], torch.Tensor],
        # message passing
        layers: int,
        max_ell: int,
        correlation: int,
        hidden_irreps: str,
        neighbour_scaling: float,
        use_self_connection: bool,
    ):
        super().__init__()

        self.radial_expansion = HaddamardProduct(
            radial_expansion_type(
                n_features=n_radial, cutoff=cutoff, trainable=True
            ),
            PolynomialEnvelope(cutoff=cutoff),
        )

        self.z_embedding = z_embedding
        self.starting_linear = o3.Linear(
            irreps_in=f"{z_embed_dim}x0e", irreps_out=hidden_irreps
        )

        self.layers = UniformModuleList(
            MACE_layer(
                max_ell=max_ell,
                correlation=correlation,
                n_dims_in=z_embed_dim,
                hidden_irreps=hidden_irreps,
                node_feats_irreps=hidden_irreps,
                edge_feats_irreps=f"{n_radial}x0e",
                avg_num_neighbors=neighbour_scaling,
                use_sc=use_self_connection,
            )
            for _ in range(layers)
        )

        self.readouts: UniformModuleList[ReadOut] = UniformModuleList(
            [LinearReadOut(hidden_irreps) for _ in range(layers - 1)]
            + [NonLinearReadOut(hidden_irreps)]
        )

    def predict_unscaled_energies(self, graph: AtomicGraph) -> torch.Tensor:
        vectors = neighbour_vectors(graph)
        Z_embedding = self.z_embedding(graph["atomic_numbers"])

        node_features = self.starting_linear(Z_embedding)
        edge_features = self.radial_expansion(
            neighbour_distances(graph).view(-1, 1)
        )

        per_atom_energies = []
        for layer, readout in zip(self.layers, self.readouts):
            node_features = layer(
                vectors,
                node_features,
                Z_embedding,
                edge_features,
                graph["neighbour_index"],
            )
            per_atom_energies.append(readout(node_features))

        return torch.sum(torch.stack(per_atom_energies), dim=0)


@e3nn.util.jit.compile_mode("script")
class MACE(_BaseMACE):
    def __init__(
        self,
        elements: list[str],
        # radial things
        cutoff: float = 5.0,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] = Bessel,
        # message passing
        layers: int = 2,
        max_ell: int = 3,
        correlation: int = 3,
        hidden_irreps: str = "128x0e + 128x1o",
        neighbour_scaling: float = 10.0,
        use_self_connection: bool = True,
    ):
        z_embed_dim = len(elements)
        z_embedding = AtomicOneHot(elements)

        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            z_embed_dim=z_embed_dim,
            z_embedding=z_embedding,
            layers=layers,
            max_ell=max_ell,
            correlation=correlation,
            hidden_irreps=hidden_irreps,
            neighbour_scaling=neighbour_scaling,
            use_self_connection=use_self_connection,
        )


@e3nn.util.jit.compile_mode("script")
class ZEmbeddingMACE(_BaseMACE):
    def __init__(
        self,
        # radial things
        cutoff: float = 5.0,
        n_radial: int = 8,
        radial_expansion_type: type[DistanceExpansion] = Bessel,
        # node attributes
        z_embed_dim: int = 16,
        # message passing
        layers: int = 2,
        max_ell: int = 3,
        correlation: int = 3,
        hidden_irreps: str = "128x0e + 128x1o",
        neighbour_scaling: float = 10.0,
        use_self_connection: bool = True,
    ):
        super().__init__(
            cutoff=cutoff,
            n_radial=n_radial,
            radial_expansion_type=radial_expansion_type,
            z_embed_dim=z_embed_dim,
            z_embedding=PerElementEmbedding(z_embed_dim),
            layers=layers,
            max_ell=max_ell,
            correlation=correlation,
            hidden_irreps=hidden_irreps,
            neighbour_scaling=neighbour_scaling,
            use_self_connection=use_self_connection,
        )
