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
    Create a model with an explicit offset, two-body and multi-body terms:

    .. code-block:: python

        from graph_pes.models import LennardJones, SchNet, FixedOffset
        from graph_pes.core import AdditionModel

        model = AdditionModel(
            offset=FixedOffset(C=-45.6, H=-1.23),
            pair_wise=LennardJones(cutoff=5.0),
            many_body=SchNet(cutoff=3.0),
        )
    """

    def __init__(self, **models: ConservativePESModel):
        max_cutoff = max([m.cutoff.item() for m in models.values()])
        super().__init__(cutoff=max_cutoff, auto_scale=False)
        self.models = UniformModuleDict(**models)

    def predict_scaled_local_energies(self, graph: AtomicGraph) -> Tensor:
        predictions = torch.stack(
            [
                model.predict_scaled_local_energies(graph).squeeze()
                for model in self.models.values()
            ]
        )  # (models, atoms)
        return torch.sum(predictions, dim=0)  # (atoms,)

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        predictions = torch.stack(
            [
                model.predict_local_energies(graph).squeeze()
                for model in self.models.values()
            ]
        )  # (models, atoms)
        return torch.sum(predictions, dim=0)  # (atoms,)

    def pre_fit(self, graphs: LabelledGraphDataset | Sequence[LabelledGraph]):
        """
        Dispatches the graphs to all ``pre_fit`` methods of the
        constituent models.
        """

        for model in self.models.values():
            model.pre_fit(graphs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            **self.models,
            stringify=True,
            max_width=80,
        )

    def __getitem__(self, key: str) -> ConservativePESModel:
        """Get a named component of the model."""
        return self.models[key]
