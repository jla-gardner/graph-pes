from __future__ import annotations

import torch

from ..atomic_graph import (
    AtomicGraph,
    PropertyKey,
    neighbour_distances,
    sum_over_central_atom_index,
    sum_over_neighbours,
)
from ..graph_pes_model import GraphPESModel
from ..utils.threebody import triplet_bond_descriptors


class StillingerWeber(GraphPESModel):
    r"""
    The `Stillinger-Weber potential <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262>`__
    predicts the total energy as a sum over two-body and three-body interactions:

    .. math::

        E = \epsilon \sum_i \sum_{j>i} \varphi_2(r_{ij}) + \zeta \epsilon  \cdot \sum_i \sum_{j \ne i} \sum_{k>j} \varphi_3(r_{ij}, r_{ik}, \theta_{ijk})

    where:

    .. math::

        \varphi_2(r) = A \left[ B \left( \frac{\sigma}{r} \right)^p - \left( \frac{\sigma}{r} \right)^q \right] \exp\left( \frac{\sigma}{r-a\sigma} \right)

    .. math::

        \varphi_3(r, s, \theta) = [\cos \theta - \cos \theta_0]^2 \exp\left( \frac{\gamma \sigma}{r - a\sigma} \right) \exp\left( \frac{\gamma \sigma}{s - a\sigma} \right)

    Default parameters are given for the original Stillinger-Weber potential,
    and hence are appropriate for modelling Si.

    When a parameter expects a ``tuple[float, bool]``, the first element is the
    value of the parameter and the second is a boolean indicating whether the
    parameter should be trainable.

    Parameters
    ----------
    lambda_
        The weight of the three-body term. Usually in the range 20-25.
    epsilon
        The energy scale.
    sigma
        The energy minimum
    A
        The two-body prefactor
    B
        The two-body exponent
    a
        Sets the cutoff as :math:`a \sigma`
    p
        The repulsive two-body exponent
    q
        The attractive two-body exponent
    gamma
        The three-body exponent
    """  # noqa: E501

    def __init__(
        self,
        lambda_: float = 21.0,
        epsilon: float = 2.1682,
        sigma: float = 2.0951,
        A: tuple[float, bool] = (7.049556277, False),
        B: tuple[float, bool] = (0.6022245584, False),
        a: tuple[float, bool] = (1.8, False),
        p: tuple[float, bool] = (4, False),
        q: tuple[float, bool] = (0, False),
        gamma: tuple[float, bool] = (1.2, False),
    ):
        cutoff = a[0] * sigma
        super().__init__(
            cutoff,
            implemented_properties=["local_energies"],
            three_body_cutoff=cutoff,
        )

        self.lambda_ = torch.nn.Parameter(torch.tensor(lambda_))
        self.epsilon = torch.nn.Parameter(torch.tensor(epsilon))
        self.sigma = torch.nn.Parameter(torch.tensor(sigma))

        self.A: torch.Tensor
        self.B: torch.Tensor
        self.a: torch.Tensor
        self.p: torch.Tensor
        self.q: torch.Tensor
        self.gamma: torch.Tensor
        self.theta_0: torch.Tensor

        def add_param(name: str, value: float, trainable: bool = False) -> None:
            if trainable:
                self.register_parameter(
                    name, torch.nn.Parameter(torch.tensor(value).float())
                )
            else:
                self.register_buffer(name, torch.tensor(value))

        add_param("A", *A)
        add_param("B", *B)
        add_param("a", *a)
        add_param("p", *p)
        add_param("q", *q)
        add_param("gamma", *gamma)

        self.register_buffer(
            "theta_0", torch.deg2rad(torch.tensor(109.4712206344))
        )

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        r_ij = neighbour_distances(graph)
        r_ij = r_ij[r_ij < self.cutoff]  # strictly less than cutoff
        x = self.sigma / r_ij
        phi_2 = (
            self.A
            * (self.B * x**self.p - x**self.q)
            * torch.exp(self.sigma / (r_ij - self.cutoff))
        ) / 2  # divide by 2 to account for double counting
        local_e = sum_over_neighbours(phi_2, graph)

        triplet_idxs, theta, r_ij, r_ik = triplet_bond_descriptors(graph)
        # strictly less than three body cutoff
        mask = (r_ij < self.three_body_cutoff) & (r_ik < self.three_body_cutoff)
        triplet_idxs = triplet_idxs[mask]
        theta = theta[mask]
        r_ij = r_ij[mask]
        r_ik = r_ik[mask]

        phi_3 = self.lambda_ * (torch.cos(theta) - torch.cos(self.theta_0)) ** 2
        phi_3 *= torch.exp(self.gamma * self.sigma / (r_ij - self.cutoff))
        phi_3 *= torch.exp(self.gamma * self.sigma / (r_ik - self.cutoff))
        phi_3 /= 2  # double counting

        local_e += sum_over_central_atom_index(phi_3, triplet_idxs[:, 0], graph)

        return {"local_energies": local_e * self.epsilon}

    @classmethod
    def monatomic_water(cls) -> StillingerWeber:
        r"""
        The Stillinger-Weber potential for `monatomic water <https://pubs.acs.org/doi/10.1021/jp805227c>`__
        with :math:`\epsilon = 0.268381`, :math:`\sigma = 2.3925`, and
        :math:`\lambda = 23.15`.

        This potential expects as input a structure containing just the oxygen
        atoms.
        """
        return cls(epsilon=0.268381, sigma=2.3925, lambda_=23.15)
