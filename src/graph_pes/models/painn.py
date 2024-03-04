from __future__ import annotations

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import (
    AtomicGraph,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
)
from graph_pes.nn import MLP, HaddamardProduct, PerSpeciesEmbedding
from torch import Tensor, nn

from .distances import Bessel, PolynomialEnvelope


class Interaction(nn.Module):
    r"""
    The interaction block of the :class:`PaiNN` model.

    Continuous filters generated from neighbour distances are convolved with
    existing scalar embeddings to create messages :math:`x_{j \rightarrow i}`
    for each neighbour :math:`j` of atom :math:`i`.

    Scalar total messages, :math:`\Delta s_i`, are created by summing over
    neighbours, while vector total messages, :math:`\Delta v_i`,
    incorporate directional information from neighbour unit vectors and
    existing vector embeddings.

    The code aims to follow **Figure 2b** of the `PaiNN paper
    <https://arxiv.org/abs/2102.03150>`_ as closely as possible.

    Parameters
    ----------
    radial_features
        The number of radial features to expand bond distances into.
    internal_dim
        The dimension of the internal representations.
    cutoff
        The cutoff distance for the radial features.
    """

    def __init__(
        self,
        radial_features: int,
        internal_dim: int,
        cutoff: float,
    ):
        super().__init__()
        self.internal_dim = internal_dim

        self.filter_generator = HaddamardProduct(
            nn.Sequential(
                Bessel(radial_features, cutoff),
                nn.Linear(radial_features, internal_dim * 3),
            ),
            PolynomialEnvelope(cutoff),
        )

        self.φ = MLP(
            [internal_dim, internal_dim, internal_dim * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self,
        vector_embeddings: Tensor,
        scalar_embeddings: Tensor,
        graph: AtomicGraph,
    ) -> tuple[Tensor, Tensor]:
        neighbours = graph["neighbour_index"][1]  # (E,)
        d = neighbour_distances(graph).unsqueeze(-1)  # (E, 1)
        unit_vectors = neighbour_vectors(graph) / d  # (E, 3)

        # continous filter message creation
        x_ij = self.filter_generator(d) * self.φ(scalar_embeddings)[neighbours]
        a, b, c = torch.split(x_ij, self.internal_dim, dim=-1)  # (E, D)

        # simple sum over neighbours to get scalar messages
        Δs = torch.zeros_like(scalar_embeddings)  # (N, D)
        Δs.scatter_add_(0, neighbours.view(-1, 1).expand_as(a), a)

        # create vector messages
        v_ij = b.unsqueeze(-1) * unit_vectors.unsqueeze(1)  # (E, D, 3)
        v_ij = v_ij + c.unsqueeze(-1) * vector_embeddings[neighbours]

        Δv = torch.zeros_like(vector_embeddings)  # (N, D, 3)
        Δv.scatter_add_(
            0, neighbours.unsqueeze(-1).unsqueeze(-1).expand_as(v_ij), v_ij
        )

        return Δv, Δs  # (N, D, 3), (N, D)


class VectorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # a hack to swap the vector and channel dimensions
        return self._linear(x.transpose(-1, -2)).transpose(-1, -2)


class Update(nn.Module):
    r"""
    The update block of the :class:`PaiNN` model.

    Projections of vector embeddings are used to update the scalar embeddings,
    and vice versa. The code aims to follow **Figure 2c** of the `PaiNN paper
    <https://arxiv.org/abs/2102.03150>`_ as closely as possible.

    Parameters
    ----------
    internal_dim
        The dimension of the internal representations.
    """

    def __init__(self, internal_dim: int):
        super().__init__()
        self.internal_dim = internal_dim
        self.U = VectorLinear(internal_dim, internal_dim)
        self.V = VectorLinear(internal_dim, internal_dim)
        self.mlp = MLP(
            [internal_dim * 2, internal_dim, internal_dim * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self, vector_embeddings: Tensor, scalar_embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        u = self.U(vector_embeddings)  # (N, D, 3)
        v = self.V(vector_embeddings)  # (N, D, 3)

        # stack scalar message and the norm of v
        m = torch.cat(
            [scalar_embeddings, torch.linalg.norm(v, dim=-1)],
            dim=-1,
        )  # (N, 2D)
        m = self.mlp(m)  # (N, 3D)

        # split the update into 3 parts
        a, b, c = torch.split(m, self.internal_dim, dim=-1)  # (N, D)

        # vector update:
        Δv = u * a.unsqueeze(-1)  # (N, D, 3)

        # scalar update:
        dot = torch.sum(u * v, dim=-1)  # (N, D)
        Δs = b + c * dot  # (N, D)

        return Δv, Δs


class PaiNN(GraphPESModel):
    r"""
    The `Polarizable Atom Interaction Neural Network (PaiNN)
    <https://arxiv.org/abs/2102.03150>`_ model.

    Alternating :class:`Interaction` and :class:`Update` blocks
    are used to residually update both vector and scalar per-atom embeddings.

    Citation:

    .. code-block:: bibtex

        @misc{Schutt-21-06,
            title = {Equivariant Message Passing for the Prediction of Tensorial Properties and Molecular Spectra},
            author = {Sch{\"u}tt, Kristof T. and Unke, Oliver T. and Gastegger, Michael},
            year = {2021},
            doi = {10.48550/arXiv.2102.03150},
        }

    Parameters
    ----------
    internal_dim
        The dimension of the internal representations.
    radial_features
        The number of radial features to expand bond distances into.
    layers
        The number of (interaction + update) layers to use.
    cutoff
        The cutoff distance for the radial features.
    """  # noqa: E501

    def __init__(
        self,
        internal_dim: int = 32,
        radial_features: int = 20,
        layers: int = 3,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.layers = layers
        self.interactions: list[Interaction] = nn.ModuleList(
            [
                Interaction(radial_features, internal_dim, cutoff)
                for _ in range(layers)
            ]
        )  # type: ignore
        self.updates: list[Update] = nn.ModuleList(
            [Update(internal_dim) for _ in range(layers)]
        )  # type: ignore
        self.z_embedding = PerSpeciesEmbedding(internal_dim)
        self.read_out = MLP(
            [internal_dim, internal_dim, 1],
            activation=nn.SiLU(),
        )

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        vector_embeddings = torch.zeros(
            (number_of_atoms(graph), self.internal_dim, 3),
            device=graph["atomic_numbers"].device,
        )
        scalar_embeddings = self.z_embedding(graph["atomic_numbers"])

        for interaction, update in zip(self.interactions, self.updates):
            Δv, Δs = interaction(vector_embeddings, scalar_embeddings, graph)
            vector_embeddings = vector_embeddings + Δv
            scalar_embeddings = scalar_embeddings + Δs

            Δv, Δs = update(vector_embeddings, scalar_embeddings)
            vector_embeddings = vector_embeddings + Δv
            scalar_embeddings = scalar_embeddings + Δs

        return self.read_out(scalar_embeddings)
