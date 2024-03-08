from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    is_local_property,
    number_of_structures,
    structure_sizes,
    sum_per_structure,
)
from graph_pes.nn import (
    PerSpeciesParameter,
    left_aligned_add,
    left_aligned_div,
    left_aligned_mul,
    left_aligned_sub,
)


class Transform(nn.Module, ABC):
    r"""
    Abstract base class for shape-preserving transformations, :math:`T`, of
    source property, :math:`x`, to target property :math:`y`, defined on an
    :class:`~graph_pes.data.AtomicGraph`, :math:`\mathcal{G}`\ :

    .. math::
        T: (x; \mathcal{G}) \mapsto y, \quad x, y \in \mathbb{R}^n

    Subclasses should implement :meth:`forward` and :meth:`inverse`. Optionally,
    implement :meth:`fit_to_source`, and :meth:`fit_to_target` methods to
    create a "fittable" transform.

    Parameters
    ----------
    trainable
        whether the transform should be trainable.
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable

    # add type hints to play nicely with mypy
    def __call__(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return super().__call__(x, graph)

    @abstractmethod
    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the forward transformation, :math:`y = T(x; \mathcal{G})`.

        Parameters
        ----------
        x
            the input data.
        graph
            the graph to condition the transformation on.

        Returns
        -------
        Tensor
            y: the transformed data.
        """

    @abstractmethod
    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the inverse transformation,
        :math:`x = T^{-1}(y; \mathcal{G})`.

        Parameters
        ----------
        y
            The input data.
        graph
            The graph to condition the transformation on.

        Returns
        -------
        Tensor
            x: the reverse-transformed data.
        """

    @torch.no_grad()
    def fit_to_source(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Fits the transform :math:`T: (x; \mathcal{G}) \rightarrow y` to property
        ``x`` defined on :code:`graphs`.

        Parameters
        ----------
        x
            The property to fit to.
        graphs
            The graphs :math:`\mathcal{G}` that the data originates from.
        """

    @torch.no_grad()
    def fit_to_target(self, y: Tensor, graphs: AtomicGraphBatch):
        r"""
        Fits the transform :math:`T: (x; \mathcal{G}) \rightarrow y` to property
        ``y`` defined on :code:`graphs`.

        Parameters
        ----------
        y
            The property to fit to.
        graphs
            The graphs :math:`\mathcal{G}` that the data originates from.
        """


class Identity(Transform):
    r"""
    The identity transform :math:`T(x; \mathcal{G}) = x` (provided
    for convenience).
    """

    def __init__(self):
        super().__init__(trainable=False)

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        return y


class Chain(Transform):
    r"""
    A chain of transformations, :math:`T_n \circ \dots \circ T_2 \circ T_1`.

    The forward transformation is applied sequentially from left to right,
    as originally defined by the order of the transformations:
    
    .. math::
        \begin{align}
        &T = \text{Chain}(T_1, T_2, \ldots, T_n) \\
        & \implies y = T_n ( \; \dots \; T_2( \; T_1(x; 
        \mathcal{G});\mathcal{G}) \; \dots \;;\mathcal{G}))
        \end{align}

    Parameters
    ----------
    transforms
        The transformations to chain together.
    trainable
        Whether the chain should be trainable.
    """

    def __init__(self, transforms: list[Transform], trainable: bool = True):
        super().__init__(trainable)
        for t in transforms:
            t.trainable = trainable
        self.transforms: list[Transform] = nn.ModuleList(transforms)  # type: ignore
        # stored here for torchscript compatibility
        self._reversed: list[Transform] = nn.ModuleList(reversed(transforms))  # type: ignore

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        for transform in self.transforms:
            x = transform(x, graph)
        return x

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        for transform in self._reversed:
            y = transform.inverse(y, graph)
        return y

    @torch.no_grad()
    def fit_to_source(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        for transform in self.transforms:
            transform.fit_to_source(x, graphs)
            x = transform(x, graphs)
        return self

    @torch.no_grad()
    def fit_to_target(self, y: Tensor, graphs: AtomicGraphBatch) -> Transform:
        for transform in self._reversed:
            transform.fit_to_target(y, graphs)
            y = transform.inverse(y, graphs)
        return self

    def __repr__(self):
        return nn.ModuleList.__repr__(self.transforms).replace(  # type: ignore
            self.transforms.__class__.__name__, self.__class__.__name__
        )


class PerAtomShift(Transform):
    r"""
    Applies a (species-dependent) per-atom shift to :math:`x`.

    When :math:`x` is a local (per-atom) property, we add atom-wise:

    .. math::

        T: y_i = x_i + \text{shift}[z_i]

    When :math:`x` is a global property, per-atom shifts are summed per
    structure, :math:`S`, in the batch:

    .. math::

        T: Y_S = X_S + \sum_{i \in S} \text{shift}[z_i]

    Parameters
    ----------
    trainable
        Whether the shift should be trainable. If ``True``, the fitted
        shift can be changed during training. If ``False``, the fitted
        shift is fixed.
    """

    def __init__(self, trainable: bool = True):
        super().__init__(trainable=trainable)
        self.shift = PerSpeciesParameter.of_dim(
            1, requires_grad=trainable, generator=0
        )
        """The fitted, per-species shifts."""

    @torch.no_grad()
    def fit_to_source(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Calculate the :math:`\text{shift}[z]` from the input data,
        :math:`x`, such that the tranformed output, :math:`y`, is expected
        to be centered about 0.

        Where :math:`x` is a local, per-atom property, we set the shift
        to be the mean of :math:`x` per species.

        Where :math:`x` is a global property, we assume that this property
        is a sum of local properties, and so perform linear regression to
        get the shift per species.

        Parameters
        ----------
        x
            The input data.
        graphs
            The atomic graphs that x originates from.
        """
        self._fit(x, graphs, "source")

    @torch.no_grad()
    def fit_to_target(self, y: Tensor, graphs: AtomicGraphBatch):
        r"""
        Calculate the :math:`\text{shift}[z]` from sample output data,
        :math:`y`, such that an input, :math:`x`, centered about 0 will
        be transformed to have similar (per-atom) centering as :math:`y`.

        Where :math:`y` is a local, per-atom property, we set the shift
        to be the mean of :math:`y` per species.

        Where :math:`y` is a global property, we assume that this property
        is a sum of local properties, and so perform linear regression to
        get the shift per species.

        Parameters
        ----------
        y
            The input data.
        graphs
            The atomic graphs that y originates from.
        """
        self._fit(y, graphs, "target")

    def _fit(
        self, t: Tensor, graphs: AtomicGraphBatch, to: str
    ) -> PerAtomShift:
        if to == "source":
            mult = 1
        elif to == "target":
            mult = -1
        else:
            raise ValueError(f"Invalid value for 'to': {to}")

        zs = torch.unique(graphs["atomic_numbers"])

        if is_local_property(t, graphs):
            # we have one data point per atom in the batch
            # we therefore fit the shift to be the mean of x
            # per unique species
            for z in zs:
                self.shift[z] = t[graphs["atomic_numbers"] == z].mean() * mult

        else:
            # we have a single data point per structure in the batch
            # we assume that x is produced as a sum of local properties
            # and do linear regression to guess the shift per species
            N = torch.zeros(number_of_structures(graphs), len(zs))
            for idx, z in enumerate(zs):
                N[:, idx] = sum_per_structure(
                    (graphs["atomic_numbers"] == z).float(), graphs
                )
            shift_vec = torch.linalg.lstsq(N, t).solution
            for idx, z in enumerate(zs):
                self.shift[z] = shift_vec[idx] * mult

        return self

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        shifts = self.shift[graph["atomic_numbers"]].squeeze()
        if not is_local_property(x, graph):
            shifts = sum_per_structure(shifts, graph)

        return left_aligned_sub(x, shifts)

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        shifts = self.shift[graph["atomic_numbers"]].squeeze()
        if not is_local_property(y, graph):
            shifts = sum_per_structure(shifts, graph)

        return left_aligned_add(y, shifts)

    def __repr__(self):
        return self.shift.__repr__().replace(
            self.shift.__class__.__name__, self.__class__.__name__
        )


class PerAtomScale(Transform):
    r"""
    Applies a (species-dependent) per-atom scale to :math:`x`.

    When :math:`x` is a local (per-atom) property, we scale atom-wise:

    .. math::

        T: y_i = x_i \times \text{scale}[z_i]

    When :math:`x` is a global property, squared per-atom scales are summed per
    structure, :math:`S`, in the batch, before being applied (in a manner
    consistent with e.g. the variance of a sum of independent random variables):

    .. math::

        T: Y_S = X_S \times \left(
            \sum_{i \in S} \text{scale}[z_i]^2
            \right)^{1/2}

    Parameters
    ----------
    trainable
        Whether the scale should be trainable. If ``True``, the fitted
        scale can be changed during training. If ``False``, the fitted
        scale is fixed.
    """

    def __init__(self, trainable: bool = True):
        super().__init__(trainable=trainable)
        self.scales = PerSpeciesParameter.of_dim(
            dim=1, requires_grad=trainable, generator=1
        )
        """The fitted, per-species scales."""

    @torch.no_grad()
    def fit_to_source(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Calculate the :math:`\text{scale}[z]` from the input data,
        :math:`x`, such that the tranformed output, :math:`y`, is expected
        to have unit variance.

        Where :math:`x` is a local, per-atom property, we fit the scale
        to be the standard-deviation of :math:`x` per species.

        Where :math:`x` is a global property, we assume that this property
        is a sum of local properties, such that the variance of the total
        property is the sum of variances of the local properties:
        :math:`\sigma^2(x) = \sum_{i \in \text{atoms}} \sigma^2(x_i)`.

        Parameters
        ----------
        x
            The input data.
        graphs
            The atomic graphs that x originates from.
        """
        self._fit(x, graphs, "source")

    @torch.no_grad()
    def fit_to_target(self, y: Tensor, graphs: AtomicGraphBatch):
        r"""
        Calculate the :math:`\text{scale}[z]` from sample output data,
        :math:`y`, such that an input, :math:`x`, with unit variance will
        be transformed to have similar (per-atom) variance as :math:`y`.

        Where :math:`y` is a local, per-atom property, we fit the scale
        to be the standard-deviation of :math:`y` per species.

        Where :math:`y` is a global property, we assume that this property
        is a sum of local properties, such that the variance of the total
        property is the sum of variances of the local properties:
        :math:`\sigma^2(x) = \sum_{i \in \text{atoms}} \sigma^2(x_i)`.

        Parameters
        ----------
        y
            The input data.
        graphs
            The atomic graphs that x originates from.
        """
        self._fit(y, graphs, "target")

    def _fit(
        self, x: Tensor, graphs: AtomicGraphBatch, to: str
    ) -> PerAtomScale:
        if to == "source":
            func = lambda x: x  # noqa
        elif to == "target":
            func = lambda x: 1 / x  # noqa
        else:
            raise ValueError(f"Invalid value for 'to': {to}")

        zs = torch.unique(graphs["atomic_numbers"])

        if is_local_property(x, graphs):
            # we have one data point per atom in the batch
            # we therefore fit the scale to be the variance of x
            # per unique species
            for z in zs:
                self.scales[z] = func(x[graphs["atomic_numbers"] == z].std())

        else:
            # TODO: this is very tricky + needs more work
            # for now, we just get a single scale for all species
            scale = (x / structure_sizes(graphs) ** 0.5).std()
            self.scales[zs] = func(scale)

        return self

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        scales = self.scales[graph["atomic_numbers"]].squeeze()
        if not is_local_property(x, graph):
            scales = sum_per_structure(scales, graph)

        return left_aligned_div(x, scales**0.5)

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        scales = self.scales[graph["atomic_numbers"]].squeeze()
        if not is_local_property(y, graph):
            scales = sum_per_structure(scales, graph)

        return left_aligned_mul(y, scales**0.5)

    def __repr__(self):
        return self.scales.__repr__().replace(
            self.scales.__class__.__name__, self.__class__.__name__
        )


def PerAtomStandardScaler(trainable: bool = True) -> Transform:
    r"""
    A convenience function for a :class:`Chain` of :class:`PerAtomShift` and
    :class:`PerAtomScale` transforms.
    """
    return Chain([PerAtomShift(trainable), PerAtomScale(trainable)])


class Scale(Transform):
    def __init__(self, trainable: bool = True, scale: float | int = 1.0):
        super().__init__(trainable=trainable)
        self.scale = nn.Parameter(
            torch.tensor(float(scale)), requires_grad=trainable
        )

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x / self.scale

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        return y * self.scale

    @torch.no_grad()
    def fit_to_source(self, x: Tensor, graphs: AtomicGraphBatch):
        self.scale.data = x.std()

    @torch.no_grad()
    def fit_to_target(self, y: Tensor, graphs: AtomicGraphBatch):
        self.scale.data = 1 / y.std()


class DividePerAtom(Transform):
    def __init__(self):
        super().__init__(trainable=False)

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return left_aligned_div(x, structure_sizes(graph))

    def inverse(self, y: Tensor, graph: AtomicGraph) -> Tensor:
        return left_aligned_mul(y, structure_sizes(graph))
