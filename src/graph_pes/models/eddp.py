from __future__ import annotations

import torch
from ase.data import atomic_numbers

from ..atomic_graph import (
    AtomicGraph,
    PropertyKey,
    neighbour_distances,
    sum_over_central_atom_index,
)
from ..graph_pes_model import GraphPESModel
from ..utils.misc import uniform_repr
from ..utils.nn import MLP, UniformModuleList
from ..utils.threebody import triplet_edge_pairs


class RadialExpansion(torch.nn.Module):
    """
    Expansion of the EDDP potential.
    """

    def __init__(
        self, cutoff: float = 5.0, features: int = 8, max_power: float = 8
    ):
        super().__init__()
        self.cutoff = cutoff
        self.exponents = torch.nn.Parameter(
            2 + (max_power - 2) * torch.linspace(0, 1, features)
        )

    def forward(
        self,
        r: torch.Tensor,  # of shape (E,)
    ) -> torch.Tensor:  # of shape (E, F)
        # apply the linear function
        f = torch.clamp(2 * (1 - r / self.cutoff), min=0)  # (E,)
        # repeat f for each exponent
        f = f.view(-1, 1).repeat(1, self.exponents.shape[0])
        # raise to the power of the exponents
        f = f ** torch.clamp(self.exponents, min=2)

        return f

    @torch.no_grad()
    def __repr__(self):
        return uniform_repr(self.__class__.__name__, cutoff=self.cutoff)


class TwoBodyDescriptor(torch.nn.Module):
    """
    Two-body term of the EDDP potential.
    """

    def __init__(
        self,
        Z1: int,
        Z2: int,
        cutoff: float = 5.0,
        features: int = 8,
        max_power: float = 8,
    ):
        super().__init__()
        self.Z1 = Z1
        self.Z2 = Z2

        self.expansion = RadialExpansion(cutoff, features, max_power)

    def forward(self, r: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        # select the relevant edges
        i = graph.neighbour_list[0]
        j = graph.neighbour_list[1]
        mask = (graph.Z[i] == self.Z1) & (graph.Z[j] == self.Z2)
        r = r[mask]

        # expand each edge
        f_edge = self.expansion(r)

        # sum over central atom
        f_atom = sum_over_central_atom_index(f_edge, i[mask], graph)
        return f_atom


class ThreeBodyDescriptor(torch.nn.Module):
    """
    Three-body term of the EDDP potential.
    """

    def __init__(
        self,
        Z1: int,
        Z2: int,
        Z3: int,
        cutoff: float = 5.0,
        features: int = 8,
        max_power: float = 8,
    ):
        super().__init__()
        self.Z1 = Z1
        self.Z2 = Z2
        self.Z3 = Z3
        self.cutoff = cutoff

        self.central_atom_expansion = RadialExpansion(
            cutoff, features, max_power
        )
        self.neighbour_atom_expansion = RadialExpansion(
            cutoff, features, max_power
        )

    def forward(
        self, i, j, k, r_ij, r_ik, r_jk, graph: AtomicGraph
    ) -> torch.Tensor:
        mask = (
            (graph.Z[i] == self.Z1)
            & (graph.Z[j] == self.Z2)
            & (graph.Z[k] == self.Z3)
        )
        r_ij = r_ij[mask]
        r_ik = r_ik[mask]
        r_jk = r_jk[mask]

        central_atom_basis = (  # (E, F)
            self.central_atom_expansion(r_ij)
            * self.central_atom_expansion(r_ik)
        )

        neighbour_atom_basis = (  # (E, F)
            self.neighbour_atom_expansion(r_ij)
            * self.neighbour_atom_expansion(r_ik)
        )

        # outer product to get (E, FxF)
        E = r_ij.shape[0]
        F = central_atom_basis.shape[1]
        prod = torch.einsum(
            "ij,ik->ijk", central_atom_basis, neighbour_atom_basis
        ).view(E, F * F)

        # sum over central atoms
        return sum_over_central_atom_index(prod, i[mask], graph)


class EDDP(GraphPESModel):
    """
    Ephemeral data-derived potential.
    """

    def __init__(
        self,
        elements: list[str],
        two_body_cutoff: float = 5.0,
        two_body_features: int = 8,
        two_body_max_power: float = 4,
        three_body_cutoff: float | None = None,
        three_body_features: int = 8,
        three_body_max_power: float = 4,
        mlp_features: int = 128,
        mlp_layers: int = 3,
    ):
        if three_body_cutoff is None:
            three_body_cutoff = two_body_cutoff

        super().__init__(
            cutoff=max(two_body_cutoff, three_body_cutoff),
            implemented_properties=["local_energies"],
        )
        self.three_body_cutoff = three_body_cutoff

        self.elements = elements

        Zs = [atomic_numbers[Z] for Z in elements]

        # two body terms
        self.Z_pairs = [(Z1, Z2) for Z1 in Zs for Z2 in Zs]
        self.two_body_descriptors = UniformModuleList(
            [
                TwoBodyDescriptor(
                    Z1,
                    Z2,
                    cutoff=two_body_cutoff,
                    features=two_body_features,
                    max_power=two_body_max_power,
                )
                for Z1, Z2 in self.Z_pairs
            ]
        )

        # three body terms
        self.Z_triplets = [
            (Z1, Z2, Z3)
            for Z1 in Zs
            for Z2 in Zs
            for Z3 in Zs
            if Z2 <= Z3  # skip identical triplets (A,B,C) and (A,C,B)
        ]
        self.three_body_descriptors = UniformModuleList(
            [
                ThreeBodyDescriptor(
                    Z1,
                    Z2,
                    Z3,
                    cutoff=three_body_cutoff,
                    features=three_body_features,
                    max_power=three_body_max_power,
                )
                for Z1, Z2, Z3 in self.Z_triplets
            ]
        )

        # read out head
        input_features = (
            len(self.two_body_descriptors) * two_body_features
            + len(self.three_body_descriptors) * three_body_features**2
        )
        layers = [input_features] + [mlp_features] * mlp_layers + [1]
        self.mlp = MLP(layers, activation="CELU")

    def featurise(self, graph: AtomicGraph) -> torch.Tensor:
        central_atom_features = []

        # two body terms
        rs = neighbour_distances(graph)
        for descriptor in self.two_body_descriptors:
            central_atom_features.append(descriptor(rs, graph))

        # three body terms
        A = triplet_edge_pairs(graph, self.three_body_cutoff)
        a = A[:, 0]
        b = A[:, 1]
        i = graph.neighbour_list[0, a]
        j = graph.neighbour_list[1, a]
        k = graph.neighbour_list[1, b]
        r_ij = rs[a]
        r_ik = rs[b]
        r_jk = rs[b]

        for descriptor in self.three_body_descriptors:
            central_atom_features.append(
                descriptor(i, j, k, r_ij, r_ik, r_jk, graph)
            )

        central_atom_features = torch.cat(central_atom_features, dim=1)
        return central_atom_features

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        features = self.featurise(graph)
        return {"local_energies": self.mlp(features).view(-1)}
