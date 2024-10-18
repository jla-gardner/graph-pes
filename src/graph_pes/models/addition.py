from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from graph_pes.core import GraphPESModel
from graph_pes.data.dataset import LabelledGraphDataset
from graph_pes.graphs import AtomicGraph, LabelledGraph, keys
from graph_pes.graphs.operations import (
    is_batch,
    number_of_atoms,
    number_of_structures,
    trim_edges,
)
from graph_pes.nn import UniformModuleDict
from graph_pes.util import uniform_repr


class AdditionModel(GraphPESModel):
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

    def __init__(self, **models: GraphPESModel):
        max_cutoff = max([m.cutoff.item() for m in models.values()])
        implemented_properties = list(
            set().union(*[m.implemented_properties for m in models.values()])
        )
        super().__init__(
            cutoff=max_cutoff,
            implemented_properties=implemented_properties,
            auto_scale_local_energies=False,
        )
        self.models = UniformModuleDict(**models)

    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, Tensor]:
        device = graph["atomic_numbers"].device
        N = number_of_atoms(graph)

        if is_batch(graph):
            S = number_of_structures(graph)
            zeros = {
                "energy": torch.zeros((S), device=device),
                "forces": torch.zeros((N, 3), device=device),
                "stress": torch.zeros((S, 3, 3), device=device),
                "local_energies": torch.zeros((N), device=device),
            }
        else:
            zeros = {
                "energy": torch.zeros((), device=device),
                "forces": torch.zeros((N, 3), device=device),
                "stress": torch.zeros((3, 3), device=device),
                "local_energies": torch.zeros((N), device=device),
            }

        predictions: dict[keys.LabelKey, Tensor] = {
            k: zeros[k] for k in self.implemented_properties
        }
        for model in self.models.values():
            trimmed = trim_edges(graph, model.cutoff.item())
            preds = model.predict(
                trimmed, properties=self.implemented_properties
            )
            for key in self.implemented_properties:
                predictions[key] += preds[key]

        return predictions

    def pre_fit_all_components(
        self, graphs: LabelledGraphDataset | Sequence[LabelledGraph]
    ):
        for model in self.models.values():
            model.pre_fit_all_components(graphs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            **self.models,
            stringify=True,
            max_width=80,
        )

    def __getitem__(self, key: str) -> GraphPESModel:
        """
        Get a component by name.

        Examples
        --------
        >>> model = AdditionModel(model1=model1, model2=model2)
        >>> model["model1"]
        """
        return self.models[key]
