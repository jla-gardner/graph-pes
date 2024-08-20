from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from graph_pes.core import ConservativePESModel
from graph_pes.data.dataset import LabelledGraphDataset
from graph_pes.graphs import AtomicGraph, LabelledGraph
from graph_pes.nn import UniformModuleDict
from graph_pes.util import uniform_repr


class AdditionModel(ConservativePESModel):
    """
    A wrapper that makes predictions as the sum of the predictions
    of its constituent models.

    Parameters
    ----------
    models
        the models (given with arbitrary names) to sum.

    Examples
    --------
    Create a model with explicit two-body and multi-body terms:

    .. code-block:: python

        from graph_pes.models import LennardJones, SchNet
        from graph_pes.core import AdditionModel

        model = AdditionModel(pair_wise=LennardJones(), many_body=SchNet())
    """

    def __init__(self, **models: ConservativePESModel):
        cutoffs = [
            m.cutoff.view(-1) for m in models.values() if m.cutoff is not None
        ]
        max_cutoff = None if not cutoffs else torch.cat(cutoffs).max().item()
        super().__init__(cutoff=max_cutoff)
        self.models = UniformModuleDict(**models)

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        predictions = torch.stack(
            [
                model.predict_local_energies(graph).squeeze()
                for model in self.models.values()
            ]
        )  # (models, atoms)
        return torch.sum(predictions, dim=0)  # (atoms,)

    def pre_fit(self, graphs: LabelledGraphDataset | Sequence[LabelledGraph]):
        for model in self.models.values():
            model.pre_fit(graphs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            **self.models,
            stringify=True,
            max_width=80,
        )
