from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    is_batch,
    is_local_property,
    number_of_atoms,
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
from torch import Tensor, nn


class Transform(nn.Module, ABC):
    r"""
    Abstract base class for shape-preserving transformations of properties,
    :math:`x`, defined on :class:`AtomicGraph <graph_pes.data.AtomicGraph>`s,
    :math:`\mathcal{G}`.

    :math:`T: (x; \mathcal{G}) \mapsto y, \quad x, y \in \mathbb{R}^n`

    Subclasses should implement :meth:`forward`, :meth:`inverse`,
    and :meth:`fit`.

    Parameters
    ----------
    trainable
        Whether the transform should be trainable.
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable

    @abstractmethod
    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the forward transformation, :math:`y = T(x; \mathcal{G})`.

        Parameters
        ----------
        x
            The input data.
        graph
            The graph to condition the transformation on.

        Returns
        -------
        y: Tensor
            The transformed data.
        """

    # add type hints to play nicely with mypy
    def __call__(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return super().__call__(x, graph)

    @abstractmethod
    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the inverse transformation,
        :math:`x = T^{-1}(y; \mathcal{G})`.

        Parameters
        ----------
        x
            The input data.
        graph
            The graph to condition the inverse transformation on.

        Returns
        -------
        x: Tensor
            The inversely-transformed data.
        """

    @abstractmethod
    @torch.no_grad()
    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        r"""
        Fits the transform to property `x` defined on `graphs`.

        Parameters
        ----------
        x
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

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x

    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        return self


class Chain(Transform):
    r"""
    A chain of transformations, :math:`T_n \circ \dots \circ T_2 \circ T_1`.

    The forward transformation is applied sequentially from left to right,
    :math:`y = T_n \circ \dots \circ T_2 \circ T_1(x; \mathcal{G})`.

    The inverse transformation is applied sequentially from right to left,
    :math:`x = T_1^{-1} \circ T_2^{-1} \circ \dots
    \circ T_n^{-1}(y; \mathcal{G})`.

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
        self.transforms = nn.ModuleList(transforms)  # type: ignore
        # stored here for torchscript compatibility
        self._reversed = nn.ModuleList(reversed(transforms))  # type: ignore

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        for transform in self.transforms:
            x = transform(x, graph)
        return x

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        for transform in self._reversed:
            x = transform.inverse(x, graph)
        return x

    @torch.no_grad()
    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        for transform in self.transforms:
            transform.fit(x, graphs)
            x = transform(x, graphs)
        return self

    def __repr__(self):
        return nn.ModuleList.__repr__(self.transforms).replace(  # type: ignore
            self.transforms.__class__.__name__, self.__class__.__name__
        )


class PerAtomShift(Transform):
    """
    Applies a (species-dependent) per-atom shift to either a local
    or global property.

    Within :meth:`fit`, we calculate the per-species shift that center the
    input data about 0.

    Within :meth:`forward`, we apply the fitted shift to the input data, and
    hence expect the output to be centered about 0.

    Within :meth:`inverse`, we apply the inverse of the fitted shift to the
    input data. If this input is centered about 0, we expect the output to be
    centered about the fitted shift.

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
            dim=1, requires_grad=trainable, generator=0
        )
        """The fitted, per-species shifts."""

    @torch.no_grad()
    def fit(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Fit the shift to the data, :math:`x`.

        Where :math:`x` is a local, per-atom property, we fit the shift
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
        # reset the shift
        zs = torch.unique(graphs["atomic_numbers"])

        if is_local_property(x, graphs):
            # we have one data point per atom in the batch
            # we therefore fit the shift to be the mean of x
            # per unique species
            for z in zs:
                self.shift[z] = x[graphs["atomic_numbers"] == z].mean()

        else:
            # we have a single data point per structure in the batch
            # we assume that x is produced as a sum of local properties
            # and do linear regression to guess the shift per species
            N = torch.zeros(number_of_structures(graphs), len(zs))
            for idx, z in enumerate(zs):
                N[:, idx] = sum_per_structure(
                    (graphs["atomic_numbers"] == z).float(), graphs
                )
            shift_vec = torch.linalg.lstsq(N, x).solution
            for idx, z in enumerate(zs):
                self.shift[z] = shift_vec[idx]

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Subtract the learned shift from :math:`x` such that the output
        is expected to be centered about 0 if :math:`x` is centered similarly
        to the data used to fit the shift.

        If :math:`x` is a local property, we subtract the shift from
        each element: :math:`x_i \rightarrow x_i - \text{shift}_i`.

        If :math:`x` is a global property, we subtract the shift from
        each structure: :math:`x_i \rightarrow x_i - \sum_{j \in i}
        \text{shift}_j`.

        Parameters
        ----------
        x
            The input data.
        batch
            The batch of atomic graphs.

        Returns
        -------
        Tensor
            The input data, shifted by the learned shift.
        """

        shifts = self.shift[graph["atomic_numbers"]].squeeze()
        if not is_local_property(x, graph):
            shifts = sum_per_structure(shifts, graph)

        return left_aligned_sub(x, shifts)

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Add the learned shift to :math:`x`, such that the output
        is expected to be centered about the learned shift if :math:`x`
        is centered about 0.

        If :math:`x` is a local property, we add the shift to
        each element: :math:`x_i \rightarrow x_i + \text{shift}_i`.

        If :math:`x` is a global property, we add the shift to
        each structure: :math:`x_i \rightarrow x_i + \sum_{j \in i}
        \text{shift}_j`.

        Parameters
        ----------
        x
            The input data.
        batch
            The batch of atomic graphs.
        """
        shifts = self.shift[graph["atomic_numbers"]].squeeze()
        if not is_local_property(x, graph):
            shifts = sum_per_structure(shifts, graph)

        return left_aligned_add(x, shifts)

    def __repr__(self):
        return self.shift.__repr__().replace(
            self.shift.__class__.__name__, self.__class__.__name__
        )


class PerAtomScale(Transform):
    r"""
    Applies a (species-dependent) per-atom scale to either a local
    or global property.

    Within :meth:`fit`, we calculate the per-species scale that
    transforms the input data to have unit variance.

    Within :meth:`forward`, we apply the fitted scale to the input data,
    and hence expect the output to have unit variance.

    Within :meth:`inverse`, we apply the inverse of the fitted scale to the
    input data. If this input has unit variance, we expect the output to
    have variance equal to the fitted scale.

    Parameters
    ----------
    trainable
        Whether the scale should be trainable. If ``True``, the fitted
        scale can be changed during training. If ``False``, the fitted
        scale is fixed.
    """

    def __init__(self, trainable: bool = True, act_on_norms: bool = False):
        super().__init__(trainable=trainable)
        self.scales = PerSpeciesParameter.of_dim(
            dim=1, requires_grad=trainable, generator=1
        )
        """The fitted, per-species scales (variances)."""
        self.act_on_norms = act_on_norms

    @torch.no_grad()
    def fit(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Fit the scale to the data, :math:`x`.

        Where :math:`x` is a local, per-atom property, we fit the scale
        to be the variance of :math:`x` per species.

        Where :math:`x` is a global property, we assume that this property
        is a sum of local properties, with the variance of this property
        being the sum of variances of the local properties:
        :math:`\sigma^2(x) = \sum_{i \in \text{atoms}} \sigma^2(x_i)`.

        Parameters
        ----------
        x
            The input data.
        graphs
            The atomic graphs that x originates from.
        """
        # reset the scale
        zs = torch.unique(graphs["atomic_numbers"])

        if self.act_on_norms:
            x = x.norm(dim=-1)

        if is_local_property(x, graphs):
            # we have one data point per atom in the batch
            # we therefore fit the scale to be the variance of x
            # per unique species
            for z in zs:
                self.scales[z] = x[graphs["atomic_numbers"] == z].var()

        else:
            # TODO: this is very tricky + needs more work
            # for now, we just get a single scale for all species
            scale = (x / structure_sizes(graphs) ** 0.5).var()
            self.scales[zs] = scale

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Scale the input data, :math:`x`, by the learned scale such that the
        output is expected to have unit variance if :math:`x` has unit
        variance.

        If :math:`x` is a local property, we scale each element:
        :math:`x_i \rightarrow x_i / \text{scale}_i`.

        If :math:`x` is a global property, we scale each structure:
        :math:`x_i \rightarrow x_i / \sqrt{\text{scale}_i}`.

        Parameters
        ----------
        x
            The input data.
        batch
            The batch of atomic graphs.

        Returns
        -------
        Tensor
            The input data, scaled by the learned scale.
        """
        scales = self.scales[graph["atomic_numbers"]].squeeze()
        if not is_local_property(x, graph):
            scales = sum_per_structure(scales, graph)

        # ndims = len(x.shape)
        # return x / scales.view(-1, *([1] * (ndims - 1))) ** 0.5

        return left_aligned_div(x, scales**0.5)

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Scale the input data, :math:`x`, by the inverse of the learned scale
        such that the output is expected to have variance equal to the
        learned scale if :math:`x` has unit variance.

        If :math:`x` is a local property, we scale each element:
        :math:`x_i \rightarrow x_i \times \text{scale}_i`.

        If :math:`x` is a global property, we scale each structure:
        :math:`x_i \rightarrow x_i \times \sqrt{\text{scale}_i}`.

        Parameters
        ----------
        x
            The input data.
        batch
            The batch of atomic graphs.

        Returns
        -------
        Tensor
            The input data, scaled by the inverse of the learned scale.
        """
        scales = self.scales[graph["atomic_numbers"]]
        if not is_local_property(x, graph):
            scales = sum_per_structure(scales, graph)

        # ndims = len(x.shape)
        # return x * scales.view(-1, *([1] * (ndims - 1))) ** 0.5
        return left_aligned_mul(x, scales**0.5)

    def __repr__(self):
        return self.scales.__repr__().replace(
            self.scales.__class__.__name__, self.__class__.__name__
        )


def PerAtomStandardScaler(trainable: bool = True) -> Transform:
    r"""
    A convenience function for a chain of :class:`PerAtomShift` and
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
        return x * self.scale**0.5

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x / self.scale**0.5

    @torch.no_grad()
    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        self.scale.data = x.var()
        return self


class DividePerAtom(Transform):
    """
    A convenience transform for dividing a property by the number of atoms
    in the structure.
    """

    def __init__(self):
        super().__init__(trainable=False)

    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        return self

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        sizes = (
            structure_sizes(graph)  # type: ignore
            if is_batch(graph)
            else number_of_atoms(graph)
        )
        return x / sizes

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        sizes = (
            structure_sizes(graph)  # type: ignore
            if is_batch(graph)
            else number_of_atoms(graph)
        )
        return x * sizes
