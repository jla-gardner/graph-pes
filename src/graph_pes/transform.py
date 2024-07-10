# TODO: work with iterators over graphs, not batches: for the big mem PR
# TODO: do we need these transforms now? how about just a type hint for the
# __call__ method of a transform

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from graph_pes.graphs import AtomicGraph, AtomicGraphBatch
from graph_pes.graphs.operations import structure_sizes
from graph_pes.nn import left_aligned_div


class Transform(nn.Module, ABC):
    r"""
    Abstract base class for shape-preserving transformations, :math:`T`, of a
    source property, :math:`x`, to target property :math:`y`, conditioned on an
    :class:`~graph_pes.data.AtomicGraph`, :math:`\mathcal{G}`\ :

    .. math::
        T: (x; \mathcal{G}) \mapsto y, \quad x, y \in \mathbb{R}^n

    Implement :meth:`forward` to apply the transform.

    Optionally, implement :meth:`fit_implementation`, to create a "fittable"
    transform. :class:`Transform` instances can only be fitted once.
    Further attempts to fit will warn the user, and do nothing.
    """

    def __init__(self):
        super().__init__()

        # save as buffer for de/serialization compatibility
        self._fitted: Tensor
        self.register_buffer("_fitted", torch.tensor(False))

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

    def fit(self, x: Tensor, graphs: AtomicGraphBatch):
        r"""
        Fits the transform :math:`T: (x; \mathcal{G}) \rightarrow y`
        to the data.

        Parameters
        ----------
        x
            The property to fit to.
        graphs
            The graphs :math:`\mathcal{G}` that the property is defined on.
        """
        if self._fitted:
            warnings.warn(
                "This transform has already been fitted. "
                "Further attempts to fit will do nothing.",
                stacklevel=2,
            )

        self.fit_implementation(x, graphs)
        self._fitted.fill_(True)

    @torch.no_grad()
    def fit_implementation(self, x: Tensor, graphs: AtomicGraph):
        """Class specific fitting implementation."""


class Identity(Transform):
    r"""
    The identity transform :math:`T(x; \mathcal{G}) = x` (provided
    for convenience).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x


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
    """

    def __init__(self, *transforms: Transform):
        super().__init__()
        self.transforms: list[Transform] = nn.ModuleList(transforms)  # type: ignore
        # TODO: uniform typed module list

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        for transform in self.transforms:
            x = transform(x, graph)
        return x

    @torch.no_grad()
    def fit_implementation(
        self,
        x: Tensor,
        graphs: AtomicGraphBatch,
    ):
        for transform in self.transforms:
            transform.fit_to_data(x, graphs)
            x = transform(x, graphs)

    def __repr__(self):
        return nn.ModuleList.__repr__(self.transforms).replace(  # type: ignore
            self.transforms.__class__.__name__, self.__class__.__name__
        )


class Scale(Transform):
    r"""
    Scale the data by a constant factor:

    .. math::
        T: y = \text{scale} \cdot x

    During fitting, the scale is set to the inverse of the standard deviation
    of the passed data.

    Parameters
    ----------
    trainable
        Whether the scale should be updated during training.
    """

    def __init__(self, trainable: bool = True, scale: float | int = 1.0):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(float(scale)),
            requires_grad=trainable,
        )

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x * self.scale

    @torch.no_grad()
    def fit_implementation(self, x: Tensor, graphs: AtomicGraphBatch):
        self.scale.data = 1 / x.std()


class DividePerAtom(Transform):
    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return divide_per_atom(x, graph)


def divide_per_atom(x: Tensor, graph: AtomicGraph) -> Tensor:
    return left_aligned_div(x, structure_sizes(graph))
