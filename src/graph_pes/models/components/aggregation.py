from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_atoms,
    number_of_edges,
    number_of_neighbours,
    sum_over_neighbours,
)
from graph_pes.utils.misc import (
    is_being_documented,
    left_aligned_div,
    uniform_repr,
)

if TYPE_CHECKING or is_being_documented():
    NeighbourAggregationMode = Literal[
        "sum", "mean", "constant_fixed", "constant_learnable", "sqrt"
    ]
else:
    NeighbourAggregationMode = str


class NeighbourAggregation(ABC, torch.nn.Module):
    r"""
    An abstract base class for aggregating values over neighbours:

    .. math::

        X_i^\prime = \text{Agg}_{j \in \mathcal{N}_i} \left[X_j\right]

    where :math:`\mathcal{N}_i` is the set of neighbours of atom :math:`i`,
    :math:`X` has shape ``(E, ...)``, :math:`X^\prime` has shape ``(N, ...)``
    and ``E`` and ``N`` are the number of edges and atoms in the graph,
    respectively.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        """Aggregate x over neighbours."""

    def pre_fit(self, graphs: AtomicGraph) -> None:
        """
        Calculate any quantities that are dependent on the graph structure
        that should be fixed before prediction.

        Default implementation does nothing.

        Parameters
        ----------
        graphs
            A batch of graphs to pre-fit to.
        """

    # type hints for mypy etc.
    def __call__(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return super().__call__(x, graph)

    @staticmethod
    def parse(
        mode: Literal[
            "sum", "mean", "constant_fixed", "constant_learnable", "sqrt"
        ],
    ) -> NeighbourAggregation:
        """
        Evaluates the following map:

        .. list-table::
           :widths: 30 70
           :header-rows: 1

           * - Mode
             - Aggregation
           * - ``"sum"``
             - :class:`SumNeighbours() <SumNeighbours>`
           * - ``"mean"``
             - :class:`MeanNeighbours() <MeanNeighbours>`
           * - ``"constant_fixed"``
             - :class:`ScaledSumNeighbours(learnable=False) <ScaledSumNeighbours>`
           * - ``"constant_learnable"``
             - :class:`ScaledSumNeighbours(learnable=True) <ScaledSumNeighbours>`
           * - ``"sqrt"``
             - :class:`VariancePreservingSumNeighbours() <VariancePreservingSumNeighbours>`

        Parameters
        ----------
        mode
            The neighbour aggregation mode to parse.

        Returns
        -------
        NeighbourAggregation
            The parsed neighbour aggregation mode.
        """  # noqa: E501
        if mode == "sum":
            return SumNeighbours()
        elif mode == "mean":
            return MeanNeighbours()
        elif mode == "constant_fixed":
            return ScaledSumNeighbours(learnable=False)
        elif mode == "constant_learnable":
            return ScaledSumNeighbours(learnable=True)
        elif mode == "sqrt":
            return VariancePreservingSumNeighbours()
        else:
            raise ValueError(f"Unknown neighbour aggregation mode: {mode}")


class SumNeighbours(NeighbourAggregation):
    r"""
    Sum over neighbours:

    .. math::

        X_i^\prime = \sum_{j \in \mathcal{N}_i} X_j
    """

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return sum_over_neighbours(x, graph)


class MeanNeighbours(NeighbourAggregation):
    r"""
    Take an average over neighbours:

    .. math::

        X_i^\prime = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} X_j

    where :math:`|\mathcal{N}_i|` is the number of neighbours of atom :math:`i`
    (including the central atom).

    .. note::

        This aggregation can lead to un-physical discontinuities in the PES
        as neighbours enter or leave the radial cutoff.
    """

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return left_aligned_div(
            sum_over_neighbours(x, graph),
            number_of_neighbours(graph, include_central_atom=True),
        )


class ScaledSumNeighbours(NeighbourAggregation):
    r"""
    Scale the sum over neighbours by a learnable or fixed constant,
    :math:`s`:

    .. math::

        X_i^\prime = \frac{1}{s} \sum_{j \in \mathcal{N}_i} X_j

    :math:`s` defaults to 1.0, but is set to the average number of neighbours
    of each atom in the training set passed to :meth:`pre_fit`.

    Parameters
    ----------
    learnable
        If ``True``, the scale is a learnable parameter. If ``False``, the
        scale is a fixed constant.
    """

    def __init__(self, learnable: bool = False):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=learnable
        )

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return sum_over_neighbours(x, graph) / self.scale

    def pre_fit(self, graphs: AtomicGraph) -> None:
        """
        Set the scale equal to the average number of neighbours in the
        training set.
        """
        avg_neighbours = number_of_edges(graphs) / number_of_atoms(graphs)
        self.scale.data = torch.tensor(avg_neighbours)

    def __repr__(self) -> str:
        return uniform_repr(
            self.__class__.__name__,
            scale=f"{self.scale.item():.3f}",
            learnable=self.scale.requires_grad,
        )


class VariancePreservingSumNeighbours(NeighbourAggregation):
    r"""
    Scale the sum over neighbours by the square root of the number of
    neighbours:

    .. math::

        X_i^\prime = \frac{1}{\sqrt{|\mathcal{N}_i|}} \sum_{j \in \mathcal{N}_i} X_j

    where :math:`|\mathcal{N}_i|` is the number of neighbours of atom :math:`i`
    (including the central atom).

    .. note::

        This aggregation can lead to un-physical discontinuities in the PES
        as neighbours enter or leave the radial cutoff.
    """  # noqa: E501

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return left_aligned_div(
            sum_over_neighbours(x, graph),
            torch.sqrt(number_of_neighbours(graph, include_central_atom=True)),
        )
