from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    GlobalProperty,
    LocalProperty,
    sum_per_structure,
)
from graph_pes.nn import PerSpeciesParameter
from jaxtyping import Shaped
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
    def forward(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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

    @abstractmethod
    def inverse(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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

    def forward(self, x, graph):
        return x

    def inverse(self, x, graph):
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
        self.transforms: list[Transform] = nn.ModuleList(transforms)  # type: ignore

    def forward(self, x, graph):
        for transform in self.transforms:
            x = transform(x, graph)
        return x

    def inverse(self, x, graph):
        for transform in reversed(self.transforms):
            x = transform.inverse(x, graph)
        return x

    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        for transform in self.transforms:
            transform.fit(x, graphs)
            x = transform(x, graphs)
        return self


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
    def fit(self, x: LocalProperty | GlobalProperty, graphs: AtomicGraphBatch):
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
        self.shift = PerSpeciesParameter.of_dim(
            1, requires_grad=self.trainable, generator=0
        )

        if graphs.is_local_property(x):
            # we have one data point per atom in the batch
            # we therefore fit the shift to be the mean of x
            # per unique species
            zs = torch.unique(graphs.Z)
            for z in zs:
                self.shift[z] = x[graphs.Z == z].mean()

        else:
            # we have a single data point per structure in the batch
            # we assume that x is produced as a sum of local properties
            # and do linear regression to guess the shift per species
            zs = torch.unique(graphs.Z)
            N = torch.zeros(graphs.n_structures, len(zs))
            for idx, z in enumerate(zs):
                N[:, idx] = sum_per_structure((graphs.Z == z).float(), graphs)
            shift_vec = torch.linalg.lstsq(N, x).solution
            for idx, z in enumerate(zs):
                self.shift[z] = shift_vec[idx]

    def forward(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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
        Shaped[Tensor, "shape ..."]
            The input data, shifted by the learned shift.
        """
        return -self._add_per_species_shift(-x, graph)

    def inverse(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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
        return self._add_per_species_shift(x, graph)

    def _add_per_species_shift(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
        if graph.is_local_property(x):
            return x + self.shift[graph.Z].squeeze()
        else:
            return x + sum_per_structure(self.shift[graph.Z].squeeze(), graph)

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

    def __init__(self, trainable: bool = True):
        super().__init__(trainable=trainable)
        self.scales = PerSpeciesParameter.of_dim(
            dim=1, requires_grad=trainable, generator=1
        )
        """The fitted, per-species scales (variances)."""

    def fit(self, x: LocalProperty | GlobalProperty, graphs: AtomicGraphBatch):
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
        self.scales = PerSpeciesParameter.of_dim(
            1, requires_grad=self.trainable, generator=1
        )

        if graphs.is_local_property(x):
            # we have one data point per atom in the batch
            # we therefore fit the scale to be the variance of x
            # per unique species
            zs = torch.unique(graphs.Z)
            for z in zs:
                self.scales[z] = x[graphs.Z == z].var()

        else:
            # TODO: this is very tricky + needs more work
            # just get a single scale for all species
            scale = (x / graphs.structure_sizes**0.5).var().item()
            self.scales = PerSpeciesParameter.of_dim(
                1, requires_grad=self.trainable, generator=scale
            )
            # hack to register the Zs we have used
            _ = self.scales[graphs.Z]

    def forward(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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
        Shaped[Tensor, "shape ..."]
            The input data, scaled by the learned scale.
        """
        if graph.is_local_property(x):
            return x / self.scales[graph.Z].squeeze() ** 0.5
        else:
            var = sum_per_structure(self.scales[graph.Z].squeeze(), graph)
            return x / var**0.5

    def inverse(
        self,
        x: Shaped[Tensor, "shape ..."],
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> Shaped[Tensor, "shape ..."]:
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
        Shaped[Tensor, "shape ..."]
            The input data, scaled by the inverse of the learned scale.
        """
        if graph.is_local_property(x):
            return x * self.scales[graph.Z].squeeze() ** 0.5
        else:
            var = sum_per_structure(self.scales[graph.Z].squeeze(), graph)
            return x * var**0.5

    def __repr__(self):
        return self.scales.__repr__().replace(
            self.scales.__class__.__name__, self.__class__.__name__
        )


class Scale(Transform):
    def __init__(self, trainable: bool = True, scale: float = 1.0):
        super().__init__(trainable=trainable)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=trainable)

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x * self.scale**0.5

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x / self.scale**0.5

    def fit(self, x: Tensor, graphs: AtomicGraphBatch) -> Transform:
        self.scale = nn.Parameter(x.var(), requires_grad=self.trainable)
        return self
