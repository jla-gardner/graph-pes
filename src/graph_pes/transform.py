from __future__ import annotations

from typing import Callable

from torch import Tensor

from graph_pes.graphs import AtomicGraph
from graph_pes.graphs.operations import structure_sizes
from graph_pes.nn import left_aligned_div

Transform = Callable[[Tensor, AtomicGraph], Tensor]
r"""
``Transform``s map a property, :math:`x`, to a target property, :math:`y`,
conditioned on an :class:`~graph_pes.graphs.AtomicGraph`, :math:`\mathcal{G}`:

.. math::

    T: (x; \mathcal{G}) \mapsto y
"""


def identity(x: Tensor, graph: AtomicGraph) -> Tensor:
    return x


def divide_per_atom(x: Tensor, graph: AtomicGraph) -> Tensor:
    return left_aligned_div(x, structure_sizes(graph))
