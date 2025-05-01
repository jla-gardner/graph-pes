from __future__ import annotations

import torch

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import uniform_repr


class ScriptedModel(GraphPESModel):
    def __init__(self, scripted_model: torch.jit.ScriptModule):
        super().__init__(
            cutoff=scripted_model.cutoff.item(),
            implemented_properties=scripted_model.implemented_properties,
            three_body_cutoff=scripted_model.three_body_cutoff.item(),
        )
        self.scripted_model = scripted_model

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        return self.scripted_model(graph)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            scripted_model=self.scripted_model,
            stringify=True,
            max_width=80,
            indent_width=2,
        )
