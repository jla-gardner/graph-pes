from __future__ import annotations

import pytest
import torch
from ase.build import molecule
from graph_pes.data import convert_to_atomic_graph
from graph_pes.models.painn import PaiNN
from graph_pes.models.pairwise import LennardJones
from graph_pes.models.schnet import SchNet
from graph_pes.models.tensornet import TensorNet

graph = convert_to_atomic_graph(molecule("CH3CH2OH"), cutoff=1.5)
models = [
    LennardJones(),
    PaiNN(),
    SchNet(),
    TensorNet(),
]


@pytest.mark.parametrize(
    "model",
    models,
    ids=[model.__class__.__name__ for model in models],
)
def test_model(model):
    actual_energy = model(graph)

    scripted_model: torch.jit.ScriptModule = torch.jit.script(model)  # type: ignore
    scripted_energy = scripted_model(graph)

    assert torch.allclose(actual_energy, scripted_energy)
