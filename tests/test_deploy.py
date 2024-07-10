from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase.build import molecule
from graph_pes.core import GraphPESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.deploy import deploy_model
from graph_pes.models import ALL_MODELS, NequIP

graph = to_atomic_graph(molecule("CH3CH2OH"), cutoff=1.5)


@pytest.mark.parametrize(
    "model_klass",
    ALL_MODELS,
    ids=[model.__name__ for model in ALL_MODELS],
)
def test_deploy(model_klass: type[GraphPESModel], tmp_path: Path):
    if model_klass is NequIP:
        model: GraphPESModel = NequIP(n_elements=3)  # type: ignore
    else:
        model = model_klass()

    # register parameters so that the model can be run
    model.pre_fit([graph])

    save_path = tmp_path / f"{model.__class__.__name__}.pt"
    deploy_model(model, cutoff=4.0, path=save_path)

    # load back in
    loaded_model = torch.jit.load(save_path)
    assert isinstance(loaded_model, torch.jit.ScriptModule)
    assert loaded_model.get_cutoff() == 4.0

    with torch.no_grad():
        actual_energy = model(graph).double()

    dummy_lammps_graph = {
        **graph,
        "compute_virial": torch.tensor(True),
        "debug": torch.tensor(False),
    }
    scripted_energy = loaded_model(dummy_lammps_graph)["total_energy"].double()

    assert torch.allclose(actual_energy, scripted_energy)
