from __future__ import annotations

from typing import Sequence

import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    has_cell,
    is_batch,
    number_of_atoms,
    number_of_structures,
)
from graph_pes.data.datasets import GraphDataset
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import uniform_repr
from graph_pes.utils.nn import UniformModuleDict


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
        )
        self.models = UniformModuleDict(**models)

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        device = graph.Z.device
        N = number_of_atoms(graph)

        if is_batch(graph):
            S = number_of_structures(graph)
            zeros = {
                "energy": torch.zeros((S), device=device),
                "forces": torch.zeros((N, 3), device=device),
                "local_energies": torch.zeros((N), device=device),
            }
        else:
            zeros = {
                "energy": torch.zeros((), device=device),
                "forces": torch.zeros((N, 3), device=device),
                "local_energies": torch.zeros((N), device=device),
            }

        # only predict stresses if the graph has a cell!
        # no list comprehension here due to TorchScript
        properties: list[PropertyKey] = []
        for prop in self.implemented_properties:
            if prop != "stress":
                properties.append(prop)

        if has_cell(graph):
            zeros["stress"] = torch.zeros_like(graph.cell)
            properties.append("stress")

        total_predictions: dict[PropertyKey, torch.Tensor] = {
            k: zeros[k] for k in properties
        }
        for model in self.models.values():
            preds = model.predict(graph, properties=properties)
            for key, value in preds.items():
                total_predictions[key] += value

        return total_predictions

    def pre_fit_all_components(
        self, graphs: GraphDataset | Sequence[AtomicGraph]
    ):
        for model in self.models.values():
            model.pre_fit_all_components(graphs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            **self.models,
            stringify=True,
            max_width=80,
            indent_width=2,
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
