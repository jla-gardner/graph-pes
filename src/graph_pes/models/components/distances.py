from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod

import torch

from graph_pes.utils.misc import to_significant_figures


class DistanceExpansion(torch.nn.Module, ABC):
    r"""
    Abstract base class for an expansion function, :math:`\phi(r) :
    [0, r_{\text{cutoff}}] \rightarrow \mathbb{R}^{n_\text{features}}`.

    Subclasses should implement :meth:`expand`, which must also work over
    batches:

    .. math::
        \phi(r) : [0, r_{\text{cutoff}}]^{n_\text{batch} \times 1}
        \rightarrow \mathbb{R}^{n_\text{batch} \times n_\text{features}}

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__()
        self.n_features = n_features
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.trainable = trainable

    @abstractmethod
    def expand(self, r: torch.Tensor) -> torch.Tensor:
        r"""
        Perform the expansion.

        Parameters
        ----------
        r : torch.Tensor
            The distances to expand. Guaranteed to have shape :math:`(..., 1)`.
        """

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Call the expansion as normal in PyTorch.

        Parameters
        ----------
        r
            The distances to expand.
        """
        if r.shape[-1] != 1:
            r = r.unsqueeze(-1)
        return self.expand(r)

    # for mypy etc.
    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return super().__call__(r)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_features={self.n_features}, "
            f"cutoff={to_significant_figures(self.cutoff.item(), 3)}, "
            f"trainable={self.trainable})"
        )


def get_distance_expansion(
    thing: str | type[DistanceExpansion],
) -> type[DistanceExpansion]:
    """
    Get a distance expansion class by it's name.

    Parameters
    ----------
    name
        The name of the distance expansion class.

    Example
    -------
    >>> get_distance_expansion("Bessel")
    <class 'graph_pes.models.components.distances.Bessel'>
    """
    if isinstance(thing, type) and issubclass(thing, DistanceExpansion):
        return thing

    try:
        klass = getattr(sys.modules[__name__], thing)
    except AttributeError:
        raise ValueError(f"Unknown distance expansion type: {thing}") from None

    if not isinstance(klass, type) or not issubclass(klass, DistanceExpansion):
        raise ValueError(f"{thing} is not a DistanceExpansion") from None

    return klass


class Bessel(DistanceExpansion):
    r"""
    The Bessel expansion:

    .. math::
        \phi_{n}(r) = \sqrt{\frac{2}{r_{\text{cut}}}}
        \frac{\sin(n \pi \frac{r}{r_\text{cut}})}{r} \quad n
        \in [1, n_\text{features}]

    where :math:`r_\text{cut}` is the cutoff radius and :math:`n` is the order
    of the Bessel function, as introduced in `Directional Message Passing for
    Molecular Graphs <http://arxiv.org/abs/2003.03123>`_.

    .. code::

        import torch
        from graph_pes.models.components.distances import Bessel
        import matplotlib.pyplot as plt

        cutoff = 5.0
        bessel = Bessel(n_features=4, cutoff=cutoff)
        r = torch.linspace(0, cutoff, 101) # (101,)

        with torch.no_grad():
            embedding = bessel(r) # (101, 4)

        plt.plot(r / cutoff, embedding)
        plt.xlabel(r"$r / r_c$")

    .. image:: bessel.svg
        :align: center

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    frequencies
        :math:`n`, the frequencies of the Bessel functions.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__(n_features, cutoff, trainable)
        self.frequencies = torch.nn.Parameter(
            torch.arange(1, n_features + 1) * math.pi / cutoff,
            requires_grad=trainable,
        )
        self.pre_factor = torch.sqrt(torch.tensor(2 / cutoff))

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        numerator = self.pre_factor * torch.sin(r * self.frequencies)
        # we avoid dividing by zero by replacing any zero elements with 1
        denominator = torch.where(r == 0, torch.tensor(1.0, device=r.device), r)
        return numerator / denominator


class GaussianSmearing(DistanceExpansion):
    r"""
    A Gaussian smearing expansion:

    .. math::
        \phi_{n}(r) = \exp\left(-\frac{(r - \mu_n)^2}{2\sigma^2}\right)
        \quad n \in [1, n_\text{features}]

    where :math:`\mu_n` is the center of the :math:`n`'th Gaussian
    and :math:`\sigma` is a width shared across all the Gaussians.

    .. code::

        import torch
        from graph_pes.models.components.distances import GaussianSmearing
        import matplotlib.pyplot as plt

        cutoff = 5.0
        gaussian = GaussianSmearing(n_features=4, cutoff=cutoff)
        r = torch.linspace(0, cutoff, 101)  # (101,)

        with torch.no_grad():
            embedding = gaussian(r)  # (101, 4)

        plt.plot(r / cutoff, embedding)
        plt.xlabel(r"$r / r_c$")

    .. image:: gaussian.svg
        :align: center

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    centers
        :math:`\mu_n`, the centers of the Gaussians.
    coef
        :math:`\frac{1}{2\sigma^2}`, the coefficient of the exponent.
    """

    def __init__(
        self,
        n_features: int,
        cutoff: float,
        trainable: bool = True,
    ):
        super().__init__(n_features, cutoff, trainable)

        sigma = cutoff / n_features
        self.coef = torch.nn.Parameter(
            torch.tensor(-1 / (2 * sigma**2)),
            requires_grad=trainable,
        )
        self.centers = torch.nn.Parameter(
            torch.linspace(0, cutoff, n_features),
            requires_grad=trainable,
        )

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        offsets = r - self.centers
        return torch.exp(self.coef * offsets**2)


class SinExpansion(DistanceExpansion):
    r"""
    A sine expansion:

    .. math::
        \phi_{n}(r) = \sin\left(\frac{n \pi r}{r_\text{cut}}\right)
        \quad n \in [1, n_\text{features}]

    where :math:`r_\text{cut}` is the cutoff radius and :math:`n` is the
    frequency of the sine function.

    .. code::

        import torch
        from graph_pes.models.components.distances import SinExpansion
        import matplotlib.pyplot as plt

        cutoff = 5.0
        sine = SinExpansion(n_features=4, cutoff=cutoff)
        r = torch.linspace(0, cutoff, 101)  # (101,)
        with torch.no_grad():
            embedding = sine(r)  # (101, 4)

        plt.plot(r / cutoff, embedding)
        plt.xlabel(r"$r / r_c$")

    .. image:: sin.svg
        :align: center

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    frequencies
        :math:`n`, the frequencies of the sine functions.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__(n_features, cutoff, trainable)
        self.frequencies = torch.nn.Parameter(
            torch.arange(1, n_features + 1) * math.pi / cutoff,
            requires_grad=trainable,
        )

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        return torch.sin(r * self.frequencies)


class ExponentialRBF(DistanceExpansion):
    r"""
    The exponential radial basis function expansion, as introduced in
    `PhysNet: A Neural Network for Predicting Energies, Forces, Dipole
    Moments and Partial Charges <https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181>`_:

    .. math::
        \phi_{n}(r) = \exp\left(-\beta_n \cdot(\exp(-r_{ij}) - \mu_n)^2 \right)
        \quad n \in [1, n_\text{features}]

    where :math:`\beta_n` and :math:`\mu_n` are the (inverse) width and
    center of the :math:`n`'th expansion, respectively.

    Following `PhysNet <https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181>`_,
    :math:`\mu_n` are evenly spaced between :math:`\exp(-r_{\text{cut}})` and
    :math:`1`, and:

    .. math::
        \left( \frac{1}{\sqrt{2}\beta_n} \right)^2 =
        \frac{1 - \exp(-r_{\text{cut}})}{n_\text{features}}

    .. code::

        import torch
        from graph_pes.models.components.distances import ExponentialRBF
        import matplotlib.pyplot as plt

        cutoff = 5.0
        rbf = ExponentialRBF(n_features=10, cutoff=cutoff)
        r = torch.linspace(0, cutoff, 101)  # (101,)
        with torch.no_grad():
            embedding = rbf(r)  # (101, 10)

        plt.plot(r / cutoff, embedding)
        plt.xlabel(r"$r / r_c$")

    .. image:: erbf.svg
        :align: center

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    β
        :math:`\beta_n`, the (inverse) widths of each basis.
    centers
        :math:`\mu_n`, the centers of each basis.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__(n_features, cutoff, trainable)

        c = torch.exp(-torch.tensor(cutoff))
        self.beta = torch.nn.Parameter(
            torch.ones(n_features) / (2 * (1 - c) / n_features) ** 2,
            requires_grad=trainable,
        )
        self.centers = torch.nn.Parameter(
            torch.linspace(c.item(), 1, n_features),
            requires_grad=trainable,
        )

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        offsets = torch.exp(-r) - self.centers
        return torch.exp(-self.beta * offsets**2)


class Envelope(torch.nn.Module):
    """
    Any envelope function, :math:`E(r)`, for smoothing potentials
    must implement a forward method that takes in a tensor of distances
    and returns a tensor of the same shape, where the values outside the
    cutoff are set to zero.
    """

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Perform the envelope function.

        Parameters
        ----------
        r : torch.Tensor
            The distances to envelope.
        """
        ...

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return super().__call__(r)


class PolynomialEnvelope(Envelope):
    r"""
    A thrice differentiable envelope function.

    .. math::
        E_p(r) = 1 - \frac{(p+1)(p+2)}{2}\cdot r^p + p(p+2) \cdot r^{p+1}
        - \frac{p(p+1)}{2}\cdot d^{p+2}

    where :math:`r_\text{cut}` is the cutoff radius,
    and :math:`p \in \mathbb{N}`.

    Parameters
    ----------
    cutoff : float
        The cutoff radius.
    p: int
        The order of the envelope function.
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
        self.register_buffer(
            "coefficients",
            torch.tensor(
                [
                    -(p + 1) * (p + 2) / 2,
                    p * (p + 2),
                    -(p * (p + 1)) / 2,
                ]
            ),
        )
        self.register_buffer("powers", torch.arange(p, p + 3))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        powers = (r.unsqueeze(-1) / self.cutoff) ** self.powers
        envelope = 1 + (powers * self.coefficients).sum(dim=-1)
        return torch.where(
            r <= self.cutoff, envelope, torch.tensor(0.0, device=r.device)
        )

    def __repr__(self):
        return f"PolynomialEnvelope(cutoff={self.cutoff}, p={self.p})"


class CosineEnvelope(Envelope):
    r"""
    A cosine envelope function.

    .. math::
        E_c(r) = \frac{1}{2}\left(1 + \cos\left(\frac{\pi r}{r_\text{cut}}
        \right)\right)

    where :math:`r_\text{cut}` is the cutoff radius.

    Parameters
    ----------
    cutoff : float
        The cutoff radius.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        cos = 0.5 * (1 + torch.cos(math.pi * r / self.cutoff))
        return torch.where(r <= self.cutoff, cos, torch.tensor(0.0))

    def __repr__(self):
        return f"CosineEnvelope(cutoff={self.cutoff})"


class SmoothOnsetEnvelope(Envelope):
    r"""
    A smooth cutoff function with an onset.

    .. math::

        f(r, r_o, r_c) = \begin{cases}
        \hfill 1 \hfill & \text{if } r < r_o \\
        \frac{(r_c - r)^2 (r_c + 2r - 3r_o)}{(r_c - r_o)^3} & \text{if } r_o \leq r < r_c \\
        \hfill 0 \hfill & \text{if } r \geq r_c
        \end{cases}

    where :math:`r_o` is the onset radius and :math:`r_c` is the cutoff radius.

    Parameters
    ----------
    cutoff : float
        The cutoff radius.
    onset : float
        The onset radius.
    """  # noqa: E501

    def __init__(self, cutoff: float, onset: float):
        super().__init__()
        if onset >= cutoff:
            raise ValueError("Onset must be less than cutoff")

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("onset", torch.tensor(onset))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.where(
            r < self.onset,
            torch.tensor(1.0, device=r.device),
            torch.where(
                r < self.cutoff,
                (self.cutoff - r) ** 2
                * (self.cutoff + 2 * r - 3 * self.onset)
                / (self.cutoff - self.onset) ** 3,
                torch.tensor(0.0, device=r.device),
            ),
        )

    def __repr__(self):
        return f"SmoothOnsetEnvelope(cutoff={self.cutoff}, onset={self.onset})"
