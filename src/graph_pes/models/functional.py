from __future__ import annotations

from typing import Callable

import torch

from graph_pes.core import LocalEnergyModel
from graph_pes.graphs import AtomicGraph


class FunctionalModel(LocalEnergyModel):
    """
    Wrap a function that returns a local energy prediction into a model
    that can be used in the same way as other
    :class:`~graph_pes.core.GraphPESModel` subclasses.

    Parameters
    ----------
    func
        The function to wrap.
    """

    def __init__(
        self,
        func: Callable[[AtomicGraph], torch.Tensor],
    ):
        super().__init__(auto_scale=False, cutoff=0)
        self.func = func

    def predict_raw_energies(self, graph: AtomicGraph) -> torch.Tensor:
        return self.func(graph)
