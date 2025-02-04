from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from graph_pes.atomic_graph import (
    DEFAULT_CUTOFF,
    AtomicGraph,
    PropertyKey,
    neighbour_distances,
    sum_over_neighbours,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models.components.distances import SmoothOnsetEnvelope
from graph_pes.utils.misc import to_significant_figures, uniform_repr
from graph_pes.utils.nn import PerElementParameter


class PairPotential(GraphPESModel, ABC):
    r"""
    An abstract base class for PES models that calculate total energy as
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

    def __init__(self, cutoff: float):
        super().__init__(
            cutoff=cutoff,
            implemented_properties=["local_energies"],
        )

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

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, Tensor]:
        """
        Predict the local energies as half the sum of the pair-wise
        interactions that each atom participates in.
        """

        # avoid tuple unpacking to keep torchscript happy
        central_atoms = graph.neighbour_list[0]
        neighbours = graph.neighbour_list[1]
        distances = neighbour_distances(graph)

        Z_i = graph.Z[central_atoms]
        Z_j = graph.Z[neighbours]

        V = self.interaction(distances, Z_i, Z_j)  # (E) / (E, 1)

        # sum over the neighbours
        energies = sum_over_neighbours(V.squeeze(), graph)

        # divide by 2 to avoid double counting
        return {"local_energies": energies / 2}


class SmoothedPairPotential(PairPotential):
    r"""
    A wrapper around a :class:`~graph_pes.models.PairPotential` that
    applies a smooth cutoff function, :math:`f` =
    :class:`~graph_pes.models.components.distances.SmoothOnsetEnvelope`,
    to the potential to ensure a continous energy surface:

    .. math::

        V(r) = f(r, r_o, r_c) \cdot V_{\text{wrapped}}(r)

    where

    .. math::

        f(r, r_o, r_c) = \begin{cases}
        \hfill 1 \hfill & \text{if } r < r_o \\
        \frac{(r_c - r)^2 (r_c + 2r - 3r_o)}{(r_c - r_o)^3} & \text{if } r_o \leq r < r_c \\
        \hfill 0 \hfill & \text{if } r \geq r_c
        \end{cases}

    and :math:`r_o` and :math:`r_c` are the onset and cutoff radii respectively.

    Parameters
    ----------
    potential
        The potential to wrap.
    smoothing_onset
        The radius at which the smooth cutoff function begins.
        Defaults to :math:`2r_c / 3`.
    """  # noqa: E501

    def __init__(
        self,
        potential: PairPotential,
        smoothing_onset: float | None = None,
    ):
        cutoff = potential.cutoff.item()
        super().__init__(cutoff=cutoff)
        if not smoothing_onset:
            smoothing_onset = 2 * cutoff / 3
        self.envelope = SmoothOnsetEnvelope(cutoff, smoothing_onset)
        self.potential = potential

    def interaction(self, r: Tensor, Z_i: Tensor, Z_j: Tensor) -> Tensor:
        raw_v = self.potential.interaction(r, Z_i, Z_j)
        return raw_v * self.envelope(r)


class LennardJones(PairPotential):
    r"""
    A pair potential with an interaction term of the form:

    .. math::

        V_{LJ}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
        \left( \frac{\sigma}{r} \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`\varepsilon` and :math:`\sigma` are strictly positive
    paramters that control the depth and width of the potential well.

    To ensure a continous energy surface, the final interaction is shifted
    by the value of the potential at the cutoff to remove the discontinuity:

    .. math::

        V(r) = V_{LJ}(r) - V_{LJ}(r_c)

    .. warning::

        The derivative of the potential is not continuous at the cutoff,
        irrespective of whether the potential is shifted or not.
        This leads to discontinuities in this models's forces. Consider wrapping
        this potential in :class:`~graph_pes.models.SmoothedPairPotential`
        to obtain a smooth and continuous potential and force field.

    Parameters
    ----------
    cutoff:
        The cutoff radius.
    epsilon:
        The maximum depth of the potential.
    sigma:
        The distance at which the potential is zero.

    Example
    -------
    .. code-block:: python

        from graph_pes.utils.analysis import dimer_curve
        from graph_pes.models import LennardJones

        dimer_curve(LennardJones(), system="H2", rmax=3.5)

    .. image:: lj-dimer.svg
        :align: center
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        epsilon: float = 0.1,
        sigma: float = 1.0,
        shift: bool = False,
    ):
        super().__init__(cutoff=cutoff)

        self._log_epsilon = torch.nn.Parameter(torch.tensor(epsilon).log())
        self._log_sigma = torch.nn.Parameter(torch.tensor(sigma).log())

        # register buffers for loading from state dict
        self.register_buffer(
            "_offset", self.V_LJ(torch.tensor(cutoff)).detach()
        )
        # save as int to avoid issues with torchscript
        self.register_buffer("_shift", torch.tensor(int(shift)))

    @property
    def epsilon(self) -> torch.Tensor:
        return self._log_epsilon.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self._log_sigma.exp()

    # don't use Z_i and Z_j, but include them for consistency with the
    # abstract method
    def interaction(
        self,
        r: torch.Tensor,
        Z_i: Optional[torch.Tensor] = None,  # noqa: UP007
        Z_j: Optional[torch.Tensor] = None,  # noqa: UP007
    ):
        v_lj = self.V_LJ(r)
        if self._shift.item():
            return v_lj - self._offset
        return v_lj

    def V_LJ(self, r: torch.Tensor) -> torch.Tensor:
        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            epsilon=to_significant_figures(self.epsilon.item(), 3),
            sigma=to_significant_figures(self.sigma.item(), 3),
            cutoff=to_significant_figures(self.cutoff.item(), 3),
        )

    @staticmethod
    def from_ase(
        sigma: float = 1.0,
        epsilon: float = 1.0,
        rc: float | None = None,
        ro: float | None = None,
        smooth: bool = False,
    ):
        """
        Create a :class:`LennardJones` potential with an interface
        identical to the ASE :class:`ase.calculators.lj.LennardJones`
        calculator.

        Please refer to the ASE documentation for more details.

        Parameters
        ----------
        sigma
            The distance at which the potential is zero.
        epsilon
            The maximum depth of the potential.
        rc
            The cutoff radius. If not given, the default value is 3 * sigma.
        ro
            The radius at which the smooth cutoff function begins.
        """
        if rc is None:
            rc = 3 * sigma
        if ro is None:
            ro = 2 * rc / 3

        if not smooth:
            return LennardJones(rc, epsilon, sigma, shift=True)
        return SmoothedPairPotential(
            LennardJones(rc, epsilon, sigma, shift=False), ro
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

        from graph_pes.utils.analysis import dimer_curve
        from graph_pes.models import Morse

        dimer_curve(Morse(), system="H2", rmax=3.5)

    .. image:: morse-dimer.svg
        :align: center
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        D: float = 0.1,
        a: float = 5.0,
        r0: float = 1.5,
    ):
        super().__init__(cutoff=cutoff)

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

    def pre_fit(self, graphs: AtomicGraph):
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


class LennardJonesMixture(PairPotential):
    r"""
    An extension of the simple :class:`LennardJones` potential to
    account for multiple atomic species.

    Each element is associated with a unique pair of parameters, $\sigma_i$ and
    $\varepsilon_i$, which control the width and depth of the potential well
    for that element.

    Interactions between atoms of different elements are calculated using
    effective parameters :math:`\sigma_{i\neq j}` and :math:`\varepsilon_{i
    \neq j}`, which are calculated as:

    * :math:`\sigma_{i\neq j} = \nu_{ij} \cdot (\sigma_i + \sigma_j) / 2`
    * :math:`\varepsilon_{i\neq j} = \zeta_{ij} \cdot \sqrt{\varepsilon_i
      \cdot \varepsilon_j}`

    where :math:`\nu_{ij}` is a mixing parameter that controls the width of the
    potential well.

    For more details, see `wikipedia <https://en.wikipedia.org/wiki/Lennard-Jones_potential#Mixtures_of_Lennard-Jones_substances>`_.
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_CUTOFF,
        modulate_distances: bool = True,
    ):
        super().__init__(cutoff=cutoff)

        self.modulate_distances: Tensor
        self.register_buffer(
            "modulate_distances", torch.tensor(int(modulate_distances))
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
            indent_width=2,
            max_width=60,
            stringify=False,
        )


class ZBLCoreRepulsion(PairPotential):
    r"""
    The `ZBL <https://en.wikipedia.org/wiki/Stopping_power_(particle_radiation)#Repulsive_interatomic_potentials>`__ repulsive potential:

    .. math::

        V(r, Z_i, Z_j) = \frac{e^2}{4 \pi \epsilon_0} \cdot \frac{Z_i Z_j}{r} \cdot \phi(r / a)

    where :math:`\phi(x)` is a dimensionless function given by:

    .. math::

        \phi(x) = 0.1818e^{-3.2x} + 0.5099e^{-0.9423x} + 0.2802e^{-0.4029x} + 0.02817e^{-0.2016x}

    and :math:`a` is the screening length:

    .. math::

        a = \frac{\lambda_p \cdot a_0}{Z_i^{\lambda_e} + Z_j^{\lambda_e}}

    where :math:`a_0` is the Bohr radius, :math:`\lambda_p = 0.8854` and
    :math:`\lambda_e = 0.23`.

    Parameters
    ----------
    cutoff : float, optional
        The cutoff radius for the potential. Default is DEFAULT_CUTOFF.
    trainable : bool, optional
        If True, the pre-factor (:math:`\lambda_p`) and exponent
        (:math:`\lambda_e`) of the screening length are trainable parameters.
        Default is False.

    Example
    -------
    .. code-block:: python

        import matplotlib.pyplot as plt
        from graph_pes.utils.analysis import dimer_curve
        from graph_pes.models import ZBL

        dimer_curve(ZBL(), system="H2", rmin=0.1, rmax=3.5)
        plt.xlim(0, 3.5)
        plt.ylim(0.01, 100)
        plt.yscale("log")

    .. image:: zbl-dimer.svg
        :align: center
    """  # noqa: E501

    def __init__(self, cutoff: float = DEFAULT_CUTOFF, trainable: bool = False):
        super().__init__(cutoff=cutoff)

        if trainable:
            self.pre_factor = torch.nn.Parameter(torch.tensor(0.8854))
            self.exponent = torch.nn.Parameter(torch.tensor(0.23))
        else:
            self.register_buffer("pre_factor", torch.tensor(0.8854))
            self.register_buffer("exponent", torch.tensor(0.23))

        ZBL_CONSTANTS = {
            "coeff": [0.1818, 0.5099, 0.2802, 0.02817],
            "exp": [-3.2, -0.9423, -0.4029, -0.2016],
        }
        self.register_buffer("ZBL_coeff", torch.tensor(ZBL_CONSTANTS["coeff"]))
        self.register_buffer("ZBL_exp", torch.tensor(ZBL_CONSTANTS["exp"]))

    def interaction(self, r: Tensor, Z_i: Tensor, Z_j: Tensor) -> Tensor:
        BOHR_RADIUS = 0.529177249  # Å
        COULOMB_CONSTANT = 14.3996  # eV Å

        a = (
            self.pre_factor
            * BOHR_RADIUS
            / (torch.pow(Z_i, self.exponent) + torch.pow(Z_j, self.exponent))
        )
        x = r / a
        phi = (
            self.ZBL_coeff[0] * torch.exp(self.ZBL_exp[0] * x)
            + self.ZBL_coeff[1] * torch.exp(self.ZBL_exp[1] * x)
            + self.ZBL_coeff[2] * torch.exp(self.ZBL_exp[2] * x)
            + self.ZBL_coeff[3] * torch.exp(self.ZBL_exp[3] * x)
        )

        return COULOMB_CONSTANT * Z_i * Z_j * phi / r
