from __future__ import annotations

import torch
from torch import Tensor, nn

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    index_over_neighbours,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    sum_over_neighbours,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.scaling import LocalEnergiesScaler
from graph_pes.utils.nn import (
    MLP,
    HaddamardProduct,
    PerElementEmbedding,
    UniformModuleList,
)

from .components.distances import Bessel, PolynomialEnvelope


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
    channels
        The number of channels of the internal representations.
    cutoff
        The cutoff distance for the radial features.
    """

    def __init__(
        self,
        radial_features: int,
        channels: int,
        cutoff: float,
    ):
        super().__init__()
        self.channels = channels

        self.filter_generator = HaddamardProduct(
            nn.Sequential(
                Bessel(radial_features, cutoff),
                nn.Linear(radial_features, channels * 3),
            ),
            PolynomialEnvelope(cutoff),
        )

        self.Phi = MLP(
            [channels, channels, channels * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self,
        vector_embeddings: Tensor,
        scalar_embeddings: Tensor,
        graph: AtomicGraph,
    ) -> tuple[Tensor, Tensor]:
        d = neighbour_distances(graph).unsqueeze(-1)  # (E, 1)
        unit_vectors = neighbour_vectors(graph) / d  # (E, 3)

        # continous filter message creation
        new_features = self.Phi(scalar_embeddings)  # (N, 3D)
        edge_embeddings = self.filter_generator(d)  # (E, 3D)
        x_ij = index_over_neighbours(new_features, graph) * edge_embeddings
        a, b, c = torch.split(x_ij, self.channels, dim=-1)  # (E, D)

        # simple sum over neighbours to get scalar messages
        delta_s = sum_over_neighbours(a, graph)

        # create vector messages
        v_ij = b.unsqueeze(-1) * unit_vectors.unsqueeze(1)  # (E, D, 3)
        v_ij = v_ij + c.unsqueeze(-1) * index_over_neighbours(
            vector_embeddings, graph
        )
        delta_v = sum_over_neighbours(v_ij, graph)

        return delta_v, delta_s  # (N, D, 3), (N, D)


class VectorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias=False)

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
    channels
        The number of channels of the internal representations.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.U = VectorLinear(channels, channels)
        self.V = VectorLinear(channels, channels)
        self.mlp = MLP(
            [channels * 2, channels, channels * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self, vector_embeddings: Tensor, scalar_embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        u = self.U(vector_embeddings)  # (N, D, 3)
        v = self.V(vector_embeddings)  # (N, D, 3)

        # stack scalar message and the norm of v
        vnorm = torch.sqrt(torch.sum(v**2, dim=-1) + 1e-8)
        m = torch.cat([scalar_embeddings, vnorm], dim=-1)  # (N, 2D)
        m = self.mlp(m)  # (N, 3D)

        # split the update into 3 parts
        a, b, c = torch.split(m, self.channels, dim=-1)  # (N, D)

        # vector update:
        delta_v = u * a.unsqueeze(-1)  # (N, D, 3)

        # scalar update:
        dot = torch.sum(u * v, dim=-1)  # (N, D)
        delta_s = b + c * dot  # (N, D)

        return delta_v, delta_s


class PaiNN(GraphPESModel):
    r"""
    The `Polarizable Atom Interaction Neural Network (PaiNN)
    <https://arxiv.org/abs/2102.03150>`_ model.

    Alternating :class:`~graph_pes.models.painn.Interaction` and
    :class:`~graph_pes.models.painn.Update` blocks
    are used to residually update both vector and scalar per-atom embeddings.

    Citation:

    .. code-block:: bibtex

        @misc{Schutt-21-06,
            title = {
                Equivariant Message Passing for the Prediction
                of Tensorial Properties and Molecular Spectra
            },
            author = {
                Sch{\"u}tt, Kristof T. and Unke, Oliver T.
                and Gastegger, Michael
            },
            year = {2021},
            doi = {10.48550/arXiv.2102.03150},
        }

    Parameters
    ----------
    channels
        The number of channels of the internal representations.
    radial_features
        The number of radial features to expand bond distances into.
    layers
        The number of (interaction + update) layers to use.
    cutoff
        The cutoff distance for the radial features.

    Examples
    --------

    Configure a PaiNN model for use with ``graph-pes-train``:

    .. code:: yaml

        model:
          +PaiNN:
            channels: 32
            layers: 3
            cutoff: 5.0
    """  # noqa: E501

    def __init__(
        self,
        channels: int = 32,
        radial_features: int = 20,
        layers: int = 3,
        cutoff: float = DEFAULT_CUTOFF,
    ):
        super().__init__(
            cutoff=cutoff,
            implemented_properties=["local_energies"],
        )

        self.channels = channels
        self.z_embedding = PerElementEmbedding(channels)
        self.interactions = UniformModuleList(
            Interaction(radial_features, channels, cutoff)
            for _ in range(layers)
        )
        self.updates = UniformModuleList(
            Update(channels) for _ in range(layers)
        )
        self.read_out = MLP(
            [channels, channels, 1],
            activation=nn.SiLU(),
        )

        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, Tensor]:
        # initialise embbedings:
        # - scalars as an embedding of the atomic numbers
        scalars = self.z_embedding(graph.Z)
        # - vectors as all 0s:
        vectors = torch.zeros(
            (number_of_atoms(graph), self.channels, 3), device=graph.Z.device
        )

        # iteratively interact and update the scalar and vector embeddings
        for interaction, update in zip(self.interactions, self.updates):
            dv, ds = interaction(vectors, scalars, graph)
            vectors = vectors + dv
            scalars = scalars + ds

            dv, ds = update(vectors, scalars)
            vectors = vectors + dv
            scalars = scalars + ds

        # mlp read out
        local_energies = self.read_out(scalars).squeeze()

        # scaling
        local_energies = self.scaler(local_energies, graph)
        return {"local_energies": local_energies}
