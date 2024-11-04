from __future__ import annotations

from typing import Any

import pytest
import torch
from graph_pes import GraphPESModel
from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.models import LennardJones


def test_state_dict():
    model = LennardJones()
    state_dict = model.state_dict()
    assert "_GRAPH_PES_VERSION" in state_dict["_extra_state"]
    assert "extra" in state_dict["_extra_state"]

    class CustomModel(GraphPESModel):
        def __init__(self, v: float):
            super().__init__(
                cutoff=0.0, implemented_properties=["local_energies"]
            )
            self.v = v

        def forward(
            self, graph: AtomicGraph
        ) -> dict[PropertyKey, torch.Tensor]:
            return {"local_energies": torch.zeros_like(graph.Z)}

        @property
        def extra_state(self) -> dict[str, Any]:
            return {"v": self.v}

        @extra_state.setter
        def extra_state(self, state: dict[str, Any]) -> None:
            self.v = state["v"]

    model1 = CustomModel(v=2.0)
    state_dict1 = model1.state_dict()
    assert state_dict1["_extra_state"]["extra"]["v"] == 2.0

    model2 = CustomModel(v=4.0)
    assert model2.v == 4.0
    model2.load_state_dict(state_dict1)
    assert model2.v == 2.0


def test_state_dict_warning():
    model = LennardJones()
    model._GRAPH_PES_VERSION = "0.0.0"  # type: ignore

    with pytest.warns(UserWarning):
        LennardJones().load_state_dict(model.state_dict())
