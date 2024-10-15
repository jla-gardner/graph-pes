from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from graph_pes.graphs import AtomicGraph, LabelledBatch
from graph_pes.graphs.operations import (
    number_of_atoms,
    number_of_edges,
    number_of_neighbours,
    sum_over_neighbours,
)
from graph_pes.util import _is_being_documented, left_aligned_div

if TYPE_CHECKING or _is_being_documented():
    NeighbourAggregationMode = Literal[
        "sum", "mean", "constant_fixed", "constant_learnable", "sqrt"
    ]
else:
    NeighbourAggregationMode = str


class NeighbourAggregation(ABC, torch.nn.Module):
    r"""
    An abstract base class for aggregating values over neighbours:

    .. math::

        X_i^' = \text{Agg}_{j \in \mathcal{N}_i} (X_j)

    where :math:`\mathcal{N}_i` is the set of neighbours of atom :math:`i`,
    :math:`X` has shape ``(E, ...)``, :math:`X^'` has shape ``(N, ...)`` and
    ``E`` and ``N`` are the number of edges and atoms in the graph,
    respectively.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        """Aggregate x over neighbours."""

    def pre_fit(self, graphs: LabelledBatch) -> None:
        """
        Calculate any quantities that are dependent on the graph structure
        that should be fixed before prediction.

        Call this in the model's ``model_specific_pre_fit`` method.

        Default implementation does nothing.

        Parameters
        ----------
        graphs
            A batch of graphs to pre-fit to.
        """

    @staticmethod
    def parse(
        mode: Literal[
            "sum", "mean", "constant_fixed", "constant_learnable", "sqrt"
        ],
    ) -> NeighbourAggregation:
        """
        Parse a neighbour aggregation mode.

        Parameters
        ----------
        mode
            The neighbour aggregation mode to parse.

        Returns
        -------
        NeighbourAggregation
            The parsed neighbour aggregation mode.
        """
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
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return sum_over_neighbours(x, graph)


class MeanNeighbours(NeighbourAggregation):
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return left_aligned_div(
            sum_over_neighbours(x, graph),
            number_of_neighbours(graph),
        )


class ScaledSumNeighbours(NeighbourAggregation):
    def __init__(self, learnable: bool = False):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=learnable
        )

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return sum_over_neighbours(x, graph) / self.scale

    def pre_fit(self, graphs: LabelledBatch) -> None:
        avg_neighbours = number_of_edges(graphs) / number_of_atoms(graphs)
        self.scale.data = torch.tensor(avg_neighbours)


class VariancePreservingSumNeighbours(NeighbourAggregation):
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return left_aligned_div(
            sum_over_neighbours(x, graph),
            torch.sqrt(number_of_neighbours(graph, include_central_atom=True)),
        )
