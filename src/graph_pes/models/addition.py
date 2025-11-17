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
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.graph_property_model import GraphTensorModel
from graph_pes.models.offsets import TensorOffset
from graph_pes.utils.misc import all_equal, uniform_repr
from graph_pes.utils.nn import UniformModuleDict


class AdditionModel(GraphPESModel):
    """
    A utility class for combining the predictions of multiple models.

    This is particularly useful for e.g. combining an many-body model with an
    :class:`~graph_pes.models.offsets.EnergyOffset` model to account for the
    arbitrary per-atom energy offsets produced by labelling codes.

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
        three_bodies = set(
            [
                round(m.three_body_cutoff.item(), 3)
                for m in models.values()
                if m.three_body_cutoff.item() > 0
            ]
        )
        if len(three_bodies) == 0:
            three_body_cutoff = None
        elif len(three_bodies) == 1:
            three_body_cutoff = three_bodies.pop()
        else:
            three_body_cutoff = None

        super().__init__(
            cutoff=max_cutoff,
            implemented_properties=implemented_properties,
            three_body_cutoff=three_body_cutoff,
        )
        self.models = UniformModuleDict(**models)

        self.register_buffer(
            "_all_models_same_properties",
            torch.tensor(
                all_equal(
                    [set(m.implemented_properties) for m in models.values()]
                )
            ),
        )

    def predict(
        self, graph: AtomicGraph, properties: list[PropertyKey]
    ) -> dict[PropertyKey, torch.Tensor]:
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

        if has_cell(graph):
            zeros["stress"] = torch.zeros_like(graph.cell)
            zeros["virial"] = torch.zeros_like(graph.cell)

        final_predictions = {}
        for key in properties:
            if key in zeros:
                final_predictions[key] = zeros[key]

        for model in self.models.values():
            preds = model.predict(graph, properties=properties)
            for key, value in preds.items():
                # move initialization in the case of "tensor" property here
                if key not in final_predictions:
                    final_predictions[key] = torch.zeros_like(
                        value, device=device
                    )
                final_predictions[key] += value

        return final_predictions

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        if not self._all_models_same_properties.item():
            raise ValueError(
                "The forward pass of an AdditionModel is not supported for "
                "models with different implemented properties. "
                "Consider using the predict method instead."
            )

        predictions = [model(graph) for model in self.models.values()]
        return {
            k: torch.stack([p[k] for p in predictions]).sum(dim=0)
            for k in predictions[0]
        }

    def pre_fit_all_components(self, graphs: Sequence[AtomicGraph]):
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


class TensorAdditionModel(GraphTensorModel):
    """
    A utility class for combining the predictions of multiple tensor models.
    """

    def __init__(self, **models: GraphTensorModel):
        targets = set([m.target_tensor_irreps for m in models.values()])
        if len(targets) != 1:
            raise ValueError(
                "The target tensor irreps of the models must be the same."
            )
        target_tensor_irreps = targets.pop()

        # TODO: allow for adding models with different target method
        target_method = set(
            [
                m.target_method
                for m in models.values()
                if not isinstance(m, TensorOffset)
            ]
        )
        if len(target_method) != 1:
            raise ValueError(
                "Only models with similar target methods can be added."
            )
        target_method = target_method.pop()

        number_of_tps = set(
            [
                m.number_of_tps
                for m in models.values()
                if not isinstance(m, TensorOffset)
            ]
        )
        if len(number_of_tps) != 1:
            raise ValueError(
                "The number of tensor products of the models must be the same."
            )
        number_of_tps = number_of_tps.pop()

        super().__init__(
            cutoff=max([m.cutoff.item() for m in models.values()]),
            implemented_properties=["tensor"],
            target_tensor_irreps=target_tensor_irreps,
            target_method=target_method,
            number_of_tps=number_of_tps,
        )
        self.models = UniformModuleDict(**models)

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        preds = [model(graph)["tensor"] for model in self.models.values()]
        return {
            "tensor": torch.stack(preds).sum(dim=0),
        }

    def predict(
        self, graph: AtomicGraph, properties: list[PropertyKey]
    ) -> dict[PropertyKey, torch.Tensor]:
        preds = [
            model.predict(graph, properties)["tensor"]
            for model in self.models.values()
        ]
        return {
            "tensor": torch.stack(preds).sum(dim=0),
        }
