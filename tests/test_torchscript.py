from __future__ import annotations

import pytest
import torch
from ase.build import molecule
from graph_pes.data import batch_graphs, convert_to_atomic_graph
from graph_pes.models.zoo import LennardJones, PaiNN, SchNet, TensorNet

graph = convert_to_atomic_graph(molecule("CH3CH2OH"), cutoff=1.5)
batch = batch_graphs([graph, graph])
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
    actual_energies = model(batch)
    assert actual_energies.shape == (2,)

    scripted_model: torch.jit.ScriptModule = torch.jit.script(model)  # type: ignore
    scripted_energy = scripted_model(batch)

    assert torch.allclose(actual_energies, scripted_energy)
