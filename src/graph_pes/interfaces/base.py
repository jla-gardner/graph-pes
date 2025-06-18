from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, is_batch
from graph_pes.graph_pes_model import GraphPESModel

T = TypeVar("T")


class InterfaceModel(GraphPESModel, ABC, Generic[T]):
    @abstractmethod
    def convert_to_underlying_input(self, graph: AtomicGraph) -> T: ...

    @abstractmethod
    def raw_forward_pass(
        self, input: T, is_batched: bool, properties: list[PropertyKey]
    ) -> dict[PropertyKey, torch.Tensor]: ...

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        input = self.convert_to_underlying_input(graph)
        return self.raw_forward_pass(
            input, is_batch(graph), self.implemented_properties
        )
