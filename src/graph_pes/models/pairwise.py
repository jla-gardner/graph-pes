from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from graph_pes.core import ConservativePESModel
from graph_pes.graphs import (
    AtomicGraph,
    AtomicGraphBatch,
    keys,
)
from graph_pes.graphs.operations import (
    neighbour_distances,
    sum_over_neighbours,
)
from graph_pes.nn import PerElementParameter
from graph_pes.util import to_significant_figures, uniform_repr


class PairPotential(ConservativePESModel, ABC):
    r"""
    An abstract base class for PES models that calculate system energy as
    a sum over pairwise interactions:

    .. math::
        E = \sum_{i, j} V(r_{ij}, Z_i, Z_j)

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`Z_i` and :math:`Z_j` are their atomic numbers.
    This can be recast as a sum over local energy contributions,
    :math:`E = \sum_i \varepsilon_i`, according to:

    .. math::
        \varepsilon_i = \frac{1}{2} \sum_j V(r_{ij}, Z_i, Z_j)

    Subclasses should implement :meth:`interaction`.
    """

    @abstractmethod
    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the interactions between pairs of atoms, given their
        distances and atomic numbers.

        Parameters
        ----------
        r
            The pair-wise distances between the atoms.
        Z_i
            The atomic numbers of the central atoms.
        Z_j
            The atomic numbers of the neighbours.

        Returns
        -------
        V: torch.Tensor
            The pair-wise interactions.
        """

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        """
        Predict the local energies as half the sum of the pair-wise
        interactions that each atom participates in.
        """

        # avoid tuple unpacking to keep torchscript happy
        central_atoms = graph[keys.NEIGHBOUR_INDEX][0]
        neighbours = graph[keys.NEIGHBOUR_INDEX][1]
        distances = neighbour_distances(graph)

        Z_i = graph[keys.ATOMIC_NUMBERS][central_atoms]
        Z_j = graph[keys.ATOMIC_NUMBERS][neighbours]

        V = self.interaction(distances, Z_i, Z_j)  # (E) / (E, 1)

        # sum over the neighbours
        energies = sum_over_neighbours(V.squeeze(), graph)

        # divide by 2 to avoid double counting
        return energies / 2


class LennardJones(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = 4 \varepsilon \left[ \left(
        \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}}
        \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`\varepsilon` and :math:`\sigma` are strictly positive
    paramters that control the depth and width of the potential well,

    Parameters
    ----------
    epsilon:
        The maximum depth of the potential.
    sigma:
        The distance at which the potential is zero.

    Example
    -------
    .. code-block:: python

        from graph_pes.analysis import dimer_curve
        from graph_pes.models import LennardJones

        dimer_curve(LennardJones(), system="H2", rmax=3.5)

    .. image:: lj-dimer.svg
        :align: center
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        sigma: float = 1.0,
    ):
        # TODO: add optional cutoff
        super().__init__(cutoff=None, auto_scale=False)

        self._log_epsilon = torch.nn.Parameter(torch.tensor(epsilon).log())
        self._log_sigma = torch.nn.Parameter(torch.tensor(sigma).log())

    @property
    def epsilon(self):
        return self._log_epsilon.exp()

    @property
    def sigma(self):
        return self._log_sigma.exp()

    # don't use Z_i and Z_j, but include them for consistency with the
    # abstract method
    def interaction(
        self,
        r: torch.Tensor,
        Z_i: Optional[torch.Tensor] = None,  # noqa: UP007
        Z_j: Optional[torch.Tensor] = None,  # noqa: UP007
    ):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r
            The pair-wise distances between the atoms.
        """

        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)

    def model_specific_pre_fit(self, graphs: AtomicGraphBatch):
        # set the distance at which the potential is zero to be
        # close to the minimum pair-wise distance
        d = torch.quantile(neighbour_distances(graphs), 0.01)
        self._log_sigma = torch.nn.Parameter(d.log())

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            epsilon=to_significant_figures(self.epsilon.item(), 3),
            sigma=to_significant_figures(self.sigma.item(), 3),
        )


class Morse(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = D (1 - e^{-a(r_{ij} - r_0)})^2

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`D`, :math:`a` and :math:`r_0` are strictly positive parameters
    that control the depth, width and center of the potential well respectively.

    Parameters
    ----------
    D:
        The maximum depth of the potential.
    a:
        A measure of the width of the potential.
    r0:
        The distance at which the potential is at its minimum.

    Example
    -------
    .. code-block:: python

        from graph_pes.analysis import dimer_curve
        from graph_pes.models import Morse

        dimer_curve(Morse(), system="H2", rmax=3.5)

    .. image:: morse-dimer.svg
        :align: center
    """

    def __init__(self, D: float = 0.1, a: float = 5.0, r0: float = 1.5):
        super().__init__(cutoff=None, auto_scale=False)

        self._log_D = torch.nn.Parameter(torch.tensor(D).log())
        self._log_a = torch.nn.Parameter(torch.tensor(a).log())
        self._log_r0 = torch.nn.Parameter(torch.tensor(r0).log())

    @property
    def D(self):
        return self._log_D.exp()

    @property
    def a(self):
        return self._log_a.exp()

    @property
    def r0(self):
        return self._log_r0.exp()

    def interaction(
        self,
        r: torch.Tensor,
        Z_i: Optional[torch.Tensor] = None,  # noqa: UP007
        Z_j: Optional[torch.Tensor] = None,  # noqa: UP007
    ):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of the central atoms. (unused)
        Z_j : torch.Tensor
            The atomic numbers of the neighbours. (unused)
        """
        return self.D * (1 - torch.exp(-self.a * (r - self.r0))) ** 2

    def model_specific_pre_fit(self, graphs: AtomicGraphBatch):
        # set the center of the well to be close to the minimum pair-wise
        # distance: the 10th percentile plus a small offset
        d = torch.quantile(neighbour_distances(graphs), 0.1) + 0.1
        self._log_r0 = torch.nn.Parameter(d.log())

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            D=to_significant_figures(self.D.item(), 3),
            a=to_significant_figures(self.a.item(), 3),
            r0=to_significant_figures(self.r0.item(), 3),
        )


# TODO: improve this class
class LennardJonesMixture(PairPotential):
    r"""
    An extension of the simple :class:`LennardJones` potential to
    account for multiple atomic species.

    Each element is associated with a unique pair of parameters, $\sigma_i$ and
    $\varepsilon_i$, which control the width and depth of the potential well
    for that element.

    Interactions between atoms of different elements are calculated using
    effective parameters $\sigma_{i\neq j}$ and $\varepsilon_{i\neq j}$,
    which are calculated as:

    * :math:`\sigma_{i\neq j} = \nu_{ij} \cdot (\sigma_i + \sigma_j) / 2`
    * :math:`\varepsilon_{i\neq j} = \zeta_{ij} \cdot \sqrt{\varepsilon_i
      \cdot \varepsilon_j}`

    where $\nu_{ij}$ is a mixing parameter that controls the width of the

    For more details, see `wikipedia <https://en.wikipedia.org/wiki/Lennard-Jones_potential#Mixtures_of_Lennard-Jones_substances>`_.
    """

    def __init__(self, modulate_distances: bool = True):
        super().__init__(cutoff=None, auto_scale=False)

        self.modulate_distances: Tensor
        self.register_buffer(
            "modulate_distances", torch.tensor(modulate_distances)
        )

        self.epsilon = PerElementParameter.of_length(1, default_value=0.1)
        self.sigma = PerElementParameter.covalent_radii(scaling_factor=0.9)
        self.nu = PerElementParameter.of_length(
            1, index_dims=2, default_value=1
        )
        self.zeta = PerElementParameter.of_length(
            1, index_dims=2, default_value=1
        )

    def interaction(self, r: Tensor, Z_i: Tensor, Z_j: Tensor) -> Tensor:
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of the central atoms.
        Z_j : torch.Tensor
            The atomic numbers of the neighbours.
        """
        cross_interaction = Z_i != Z_j  # (E)

        sigma_j = self.sigma[Z_j].squeeze()  # (E)
        sigma_i = self.sigma[Z_i].squeeze()  # (E)
        nu = (
            self.nu[Z_i, Z_j].squeeze()
            if self.modulate_distances
            else torch.tensor(1)
        )
        sigma = torch.where(
            cross_interaction,
            nu * (sigma_i + sigma_j) / 2,
            sigma_i,
        ).clamp(min=0.2)  # (E)

        epsilon_i = self.epsilon[Z_i].squeeze()
        epsilon_j = self.epsilon[Z_j].squeeze()
        zeta = self.zeta[Z_i, Z_j].squeeze()
        epsilon = torch.where(
            cross_interaction,
            zeta * (epsilon_i * epsilon_j).sqrt(),
            epsilon_i,
        ).clamp(min=0.00)  # (E)

        x = sigma / r
        return 4 * epsilon * (x**12 - x**6)

    def __repr__(self):
        kwargs = {
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "zeta": self.zeta,
        }
        if self.modulate_distances:
            kwargs["nu"] = self.nu

        return uniform_repr(
            self.__class__.__name__,
            **kwargs,
            max_width=60,
            stringify=False,
        )
