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
from ..utils.nn import MLP, AtomicOneHot, UniformModuleList
from ..utils.threebody import triplet_edges


class RadialExpansion(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        features: int = 8,
        max_power: float = 8,
        learnable_powers: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff
        beta = (max_power / 2) ** (1.0 / (features - 1))
        powers = torch.tensor(
            [2 * (beta**i) for i in range(features)], dtype=torch.float
        )
        if learnable_powers:
            self.exponents = torch.nn.Parameter(powers)
        else:
            self.register_buffer("exponents", powers)

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
        learnable_powers: bool = False,
    ):
        super().__init__()
        self.Z1 = Z1
        self.Z2 = Z2

        self.expansion = RadialExpansion(
            cutoff, features, max_power, learnable_powers
        )

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
        learnable_powers: bool = False,
    ):
        super().__init__()
        self.Z1 = Z1
        self.Z2 = Z2
        self.Z3 = Z3
        self.cutoff = cutoff

        self.central_atom_expansion = RadialExpansion(
            cutoff, features, max_power, learnable_powers
        )
        self.neighbour_atom_expansion = RadialExpansion(
            cutoff, features, max_power, learnable_powers
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

        e_ij = self.central_atom_expansion(r_ij)
        e_ik = self.central_atom_expansion(r_ik)
        e_jk = self.neighbour_atom_expansion(r_jk)

        E = r_ij.shape[0]
        F = e_ij.shape[1]
        prod = (e_ij * e_ik)[:, None, :] * e_jk[:, :, None]
        prod = prod.view(E, F * F)

        # sum over central atoms
        return sum_over_central_atom_index(prod, i[mask], graph)


class EDDP(GraphPESModel):
    r"""
    The Ephemeral Data Derived Potential (EDDP) architecture, from
    `Ephemeral data derived potentials for random structure 
    search. Phys. Rev. B 106, 014102 <https://doi.org/10.1103/PhysRevB.106.014102>`_.

    This model makes local energy predictions by featurising the local
    atomic environment and passing the result through an MLP:

    .. math::

        \varepsilon_i = \text{MLP}(\mathbf{F}_i)

    where :math:`\mathbf{F}_i` is a a concatenation of one-, two- and
    three-body features:

    .. math::

        \mathbf{F}_i = \mathbf{F}_i^{(1)} \oplus \mathbf{F}_i^{(2)}
        \oplus \mathbf{F}_i^{(3)}

    as per Eq. 14. The m'th component of the two-body feature is given by:

    .. math::

        \mathbf{F}_i^{(2)}[m] = \sum_{j \in \mathcal{N}(i)} f(r_{ij})^{p_m}

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    :math:`\mathcal{N}(i)` is the set of neighbours of atom :math:`i`,
    :math:`p_m \geq 2` is some power, and :math:`f(r_{ij})` is defined as:

    .. math::

        f(r_{ij}) = \begin{cases}
            2 (1 - r_{ij} / r_{\text{cutoff}}) & \text{if } r_{ij} \leq 
            r_{\text{cutoff}} \\
            0 & \text{otherwise}
        \end{cases}
    
    as per Eq. 7. Finally, the :math:`(m, o)`'th component of the three-body
    feature is given by:

    .. math::

        \mathbf{F}_i^{(3)}[m, o] = \sum_{j \in \mathcal{N}(i)}
        \sum_{k > j \in \mathcal{N}(i)} f(r_{ij})^{p_m} f(r_{ik})^{p_m}
        f(r_{jk})^{p_o}

    If you use this model, please cite the original paper:

    .. code-block:: bibtex

        @article{Pickard-22-07,
            title = {
                Ephemeral Data Derived Potentials for Random Structure Search
            },
            author = {Pickard, Chris J.},
            year = {2022},
            journal = {Physical Review B},
            volume = {106},
            number = {1},
            pages = {014102},
            doi = {10.1103/PhysRevB.106.014102},
        }

    Parameters
    ----------
    elements:
        The elements the model will be applied to (e.g. ``["C", "H"]``).
    cutoff:
        The maximum distance between pairs of atoms to consider their
        interaction.
    features:
        The number of features used to describe the two body interactions.
    max_power:
        The maximum power of the interaction expansion for two body terms.
    mlp_width:
        The width of the MLP (i.e. number of neurons in each hidden layer).
    mlp_layers:
        The number of hidden layers in the MLP.
    activation:
        The activation function to use in the MLP.
    three_body_cutoff:
        The radial cutoff for three body interactions. If ``None``, the
        same as ``cutoff``.
    three_body_features:
        The number of features used to describe the three body interactions.
        If ``None``, the same as for the two body terms.
    three_body_max_power:
        The maximum power of the interaction expansion for three body terms.
        If ``None``, the same as for the two body terms.
    learnable_powers:
        If ``True``, the powers of the interaction expansion are learnable.
        If ``False``, the powers are fixed to the values given by ``max_power``.  
    """  # noqa: E501

    def __init__(
        self,
        elements: list[str],
        cutoff: float = 5.0,
        features: int = 8,
        max_power: float = 8,
        mlp_width: int = 16,
        mlp_layers: int = 1,
        activation: str = "CELU",
        three_body_cutoff: float | None = None,
        three_body_features: int | None = None,
        three_body_max_power: float | None = None,
        learnable_powers: bool = False,
    ):
        if three_body_cutoff is None:
            three_body_cutoff = cutoff
        if three_body_features is None:
            three_body_features = features
        if three_body_max_power is None:
            three_body_max_power = max_power

        super().__init__(
            cutoff=max(cutoff, three_body_cutoff),
            implemented_properties=["local_energies"],
            three_body_cutoff=three_body_cutoff,
        )

        self.elements = elements
        Zs = [atomic_numbers[Z] for Z in elements]

        # one body terms
        self.one_hot = AtomicOneHot(elements)

        # two body terms
        self.Z_pairs = [(Z1, Z2) for Z1 in Zs for Z2 in Zs]
        self.two_body_descriptors = UniformModuleList(
            [
                TwoBodyDescriptor(
                    Z1,
                    Z2,
                    cutoff=cutoff,
                    features=features,
                    max_power=max_power,
                    learnable_powers=learnable_powers,
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
                    learnable_powers=learnable_powers,
                )
                for Z1, Z2, Z3 in self.Z_triplets
            ]
        )

        # read out head
        input_features = (
            len(elements)
            + len(self.two_body_descriptors) * features
            + len(self.three_body_descriptors) * three_body_features**2
        )
        layers = [input_features] + [mlp_width] * mlp_layers + [1]
        self.mlp = MLP(layers, activation=activation)

        self.shifts = torch.nn.Parameter(torch.zeros(input_features))
        self.scales = torch.nn.Parameter(torch.ones(input_features))

    def featurise(self, graph: AtomicGraph) -> torch.Tensor:
        # one body terms
        central_atom_features = [self.one_hot(graph.Z)]

        # two body terms
        rs = neighbour_distances(graph)
        for descriptor in self.two_body_descriptors:
            central_atom_features.append(descriptor(rs, graph))

        # three body terms
        i, j, k, r_ij, r_ik, r_jk = triplet_edges(
            graph, self.three_body_cutoff.item()
        )

        for descriptor in self.three_body_descriptors:
            central_atom_features.append(
                descriptor(i, j, k, r_ij, r_ik, r_jk, graph)
            )

        # concatenate all features
        f = torch.cat(central_atom_features, dim=1)
        f = (f - self.shifts) / self.scales
        return f

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        features = self.featurise(graph)
        return {"local_energies": self.mlp(features).view(-1)}

    def pre_fit(self, graphs: AtomicGraph) -> None:
        X = self.featurise(graphs)
        _mean = X.mean(dim=0)
        _std = X.std(dim=0)
        # replace nans
        _std[torch.isnan(_std) | (_std < 1e-6)] = 1
        self.shifts.data = _mean
        self.scales.data = _std
