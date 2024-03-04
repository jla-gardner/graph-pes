from __future__ import annotations

from typing import Literal

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import AtomicGraph, neighbour_distances
from graph_pes.nn import MLP, PerSpeciesEmbedding, ShiftedSoftplus
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .distances import (
    Bessel,
    DistanceExpansion,
    GaussianSmearing,
    PolynomialEnvelope,
)


class CFConv(MessagePassing):
    r"""
    CFConv: The Continous Filter Convolution

    Originally proposed as part of `SchNet <https://arxiv.org/abs/1706.08566>`_,
    this message passing layer creates messages of the form:

    .. math::
        m_{ij} = x_j \odot \mathbb{F}(r_{ij})

    where :math:`\mathbb{F}` is some learnable filter function which
    expands the radial distance :math:`r_{ij}` into a set of features
    that match the dimensionality of the node features :math:`x_j`.

    Example
    -------
    .. code::

        from graph_pes.models.distances import GaussianSmearing
        from graph_pes.nn import MLP
        from torch import nn

        filter_generator = nn.Sequential(
            GaussianSmearing(8, 5.0),
            MLP([8, 16, 16, 8]),
        )
        cfconv = CFConv(filter_generator)

    Parameters
    ----------
    filter_generator : nn.Module
        The filter function :math:`\mathbb{F}`.
    """

    def __init__(self, filter_generator: nn.Module):
        super().__init__(aggr="add")
        self.filter_generator = filter_generator

    def message(
        self, x_j: torch.Tensor, neighbour_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the filter function to the distances and multiply by the
        node features.

        Parameters
        ----------
        x_j : torch.Tensor
            The node features of the neighbors.
        neighbour_distances : torch.Tensor
            The distances to the neighbors.
        """

        return x_j * self.filter_generator(neighbour_distances)

    def update(self, inputs: Tensor) -> Tensor:
        """
        Identity update function.
        """
        return inputs

    def forward(
        self,
        neighbour_index: torch.Tensor,
        neighbour_distances: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.propagate(
            neighbour_index,
            neighbour_distances=neighbour_distances,
            x=node_features,
        )

    def __repr__(self):
        rep = f"CFConv({self.filter_generator})"
        if len(rep) > 40:
            indented = "  " + "\n  ".join(
                str(self.filter_generator).split("\n")
            )
            rep = f"CFConv(\n{indented}\n)"
        return rep


class SchNetInteraction(nn.Module):
    r"""
    SchNet interaction block.

    Updates the embedding of each atom, :math:`x_i` by sequentially
    applying the following:

    1. a linear transform of each node's features :math:`x_i \leftarrow W x_i`
    2. message creation :math:`m_{ij} = x_j \odot \mathbb{F}(r_{ij})`
    3. message aggregation
       :math:`m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}`
    4. a multi-layer perceptron that further embeds these new node features
       :math:`x_i \leftarrow \mathrm{MLP}(h_i)`

    Parameters
    ----------
    n_features
        Number of features per node.
    expansion_features
        Number of features used for the radial basis expansion.
    cutoff
        Neighborhood cutoff radius.
    basis_type
        The type of radial basis expansion to use.
    """

    def __init__(
        self,
        n_features: int,
        expansion_features: int,
        cutoff: float,
        basis_type: type[DistanceExpansion],
    ):
        super().__init__()

        # schnet interaction block's are composed of 3 elements

        # 1. linear transform to get new node features
        self.linear = nn.Linear(n_features, n_features, bias=False)

        # 2. cfconv to mix these new features with distances information,
        # and aggregate over neighbors to create completely new node features
        filter_generator = nn.Sequential(
            basis_type(expansion_features, cutoff),
            MLP(
                [expansion_features, n_features, n_features],
                activation=ShiftedSoftplus(),
            ),
        )
        self.cfconv = CFConv(filter_generator)

        # 3. mlp to further embed these new node features
        self.mlp = MLP(
            [n_features, n_features, n_features],
            activation=ShiftedSoftplus(),
        )

    def forward(
        self,
        neighbour_index: torch.Tensor,
        neighbour_distances: torch.Tensor,
        node_features: torch.Tensor,
    ):
        # 1. linear transform to get new node features
        h = self.linear(node_features)
        # 2. cfconv to mix these new features with distances information,
        # and aggregate over neighbors to create completely new node features
        h = self.cfconv(neighbour_index, neighbour_distances, h)
        # 3. mlp to further embed these new node features
        return self.mlp(h)


class SchNet(GraphPESModel):
    r"""
    The `SchNet <https://arxiv.org/abs/1706.08566>`_ model: a pairwise, scalar,
    message passing GNN.

    A stack of :class:`SchNetInteraction` blocks are used to update
    the node features of each atom, :math:`x_i` by sequentially
    aggregating over neighbouring atom's features, :math:`x_j`:, and
    distances, :math:`r_{ij}`:.

    Citation:

    .. code:: bibtex

        @article{Schutt-18-03,
            title = {{{SchNet}} {\textendash} {{A}} Deep Learning Architecture for Molecules and Materials},
            author = {Sch{\"u}tt, K. T. and Sauceda, H. E. and Kindermans, P.-J. and Tkatchenko, A. and M{\"u}ller, K.-R.},
            year = {2018},
            journal = {The Journal of Chemical Physics},
            volume = {148},
            number = {24},
            pages = {241722},
            doi = {10.1063/1.5019779},
        }

    Parameters
    ----------
    node_features
        Number of features per node.
    expansion_features
        Number of features used for the radial basis expansion.
    cutoff
        Neighborhood cutoff radius.
    num_interactions
        Number of interaction blocks to apply.
    expansion
        The type of radial basis expansion to use. Defaults to
        :class:`GaussianSmearing <graph_pes.models.distances.GaussianSmearing>`
        as in the original paper.
    """  # noqa: E501

    def __init__(
        self,
        node_features: int = 64,
        expansion_features: int = 50,
        cutoff: float = 5.0,
        num_interactions: int = 3,
        expansion: type[DistanceExpansion] | None = None,
    ):
        super().__init__()

        if expansion is None:
            expansion = GaussianSmearing

        self.chemical_embedding = PerSpeciesEmbedding(node_features)

        self.interactions = nn.ModuleList(
            SchNetInteraction(
                node_features, expansion_features, cutoff, expansion
            )
            for _ in range(num_interactions)
        )

        self.read_out = MLP(
            [node_features, node_features // 2, 1],
            activation=ShiftedSoftplus(),
        )

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        h = self.chemical_embedding(graph["atomic_numbers"])

        for interaction in self.interactions:
            h = h + interaction(
                graph["neighbour_index"], neighbour_distances(graph), h
            )

        return self.read_out(h)


class ImprovedSchNetInteraction(nn.Module):
    def __init__(
        self,
        cutoff: float,
        radial_features: int = 8,
        radial_expansion: type[DistanceExpansion] = Bessel,
        atom_features: int = 4,
        mlp_height: int = 16,
        mlp_depth: int = 3,
        activation: type[nn.Module] = nn.CELU,
        pre_embed: bool = True,
    ):
        super().__init__()

        # radial filter components
        self.envelope = PolynomialEnvelope(cutoff)
        self.raw_filter_generator = nn.Sequential(
            radial_expansion(radial_features, cutoff),
            MLP(
                [radial_features, mlp_height, mlp_height, atom_features],
                activation=activation(),
            ),
        )

        # atom embedding components
        self.pre_embed = pre_embed
        if pre_embed:
            self.pre_embedding = MLP(
                [atom_features] + [mlp_height] * mlp_depth + [atom_features],
                activation=activation(),
            )
        self.post_embedding = MLP(
            [atom_features] + [mlp_height] * mlp_depth + [atom_features],
            activation=activation(),
        )

    def get_radial_embeddings(self, distances: torch.Tensor) -> torch.Tensor:
        envelope = self.envelope(distances).unsqueeze(-1)
        radial_embedding = self.raw_filter_generator(distances)
        return radial_embedding * envelope

    def forward(
        self,
        central_atom_embeddings: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        # get radial features
        radial_embedding = self.get_radial_embeddings(
            neighbour_distances(graph)
        )

        if self.pre_embed:
            # update atomic features
            central_atom_embeddings = self.pre_embedding(
                central_atom_embeddings
            )

        # haddamard product to mix atomic features with radial features
        i, j = graph["neighbour_index"]
        neighbour_atom_embeddings = central_atom_embeddings[j]
        messages = neighbour_atom_embeddings * radial_embedding

        # aggregate over neighbours to get new central atom embeddings
        central_atom_embeddings = scatter(messages, i, reduce="add")

        # update atomic features
        return self.post_embedding(central_atom_embeddings)


class ImprovedSchNet(GraphPESModel):
    def __init__(
        self,
        cutoff: float,
        radial_features: int = 8,
        radial_expansion: type[DistanceExpansion] = Bessel,
        atom_features: int = 4,
        mlp_height: int = 16,
        mlp_depth: int = 3,
        activation: type[nn.Module] = nn.CELU,
        num_blocks: int = 3,
        residual_messages: bool = True,
        pre_embed: Literal["none", "all", "2+"] = "2+",
    ):
        super().__init__()

        self.residual_messages = residual_messages

        self.chemical_embedding = PerSpeciesEmbedding(dim=atom_features)

        def _rule(i):
            return {
                "none": False,
                "all": True,
                "2+": i > 0,
            }[pre_embed]

        self.interactions = nn.ModuleList(
            ImprovedSchNetInteraction(
                cutoff,
                radial_features,
                radial_expansion,
                atom_features,
                mlp_height,
                mlp_depth,
                activation,
                pre_embed=_rule(i),
            )
            for i in range(num_blocks)
        )

        self.read_outs = nn.ModuleList(
            MLP(
                [atom_features] + [mlp_height] * mlp_depth + [1],
                activation=activation(),
            )
            for _ in range(num_blocks)
        )

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        return self.per_block_predictions(graph).sum(dim=1)

    def per_block_predictions(self, graph: AtomicGraph) -> torch.Tensor:
        h = self.chemical_embedding(graph["atomic_numbers"])

        predictions = []
        for block, read_out in zip(self.interactions, self.read_outs):
            if self.residual_messages:
                h = block(h, graph) + h
            else:
                h = block(h, graph)
            predictions.append(read_out(h).squeeze())

        return torch.stack(predictions, dim=1)
