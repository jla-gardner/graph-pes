from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase.build import molecule
from graph_pes.core import GraphPESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.deploy import deploy_model
from graph_pes.graphs.operations import number_of_atoms
from graph_pes.models import ALL_MODELS, NequIP

CUTOFF = 1.5
graph = to_atomic_graph(molecule("CH3CH2OH"), cutoff=CUTOFF)


@pytest.mark.parametrize(
    "model_klass",
    ALL_MODELS,
    ids=[model.__name__ for model in ALL_MODELS],
)
def test_deploy(model_klass: type[GraphPESModel], tmp_path: Path):
    # 1. instantiate the model
    kwargs = {"n_elements": 3} if model_klass is NequIP else {}
    model = model_klass(**kwargs)
    model.pre_fit([graph])  # required by some models before making predictions

    # 2. deploy the model
    save_path = tmp_path / "model.pt"
    deploy_model(model, cutoff=CUTOFF, path=save_path)

    # 3. load the model back in
    loaded_model = torch.jit.load(save_path)
    assert isinstance(loaded_model, torch.jit.ScriptModule)
    assert loaded_model.get_cutoff() == CUTOFF

    # 4. test outputs
    outputs = loaded_model(
        # mock the graph that would be passed through from LAMMPS
        {
            **graph,
            "compute_virial": torch.tensor(True),
            "debug": torch.tensor(False),
        }
    )
    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {
        "total_energy",
        "local_energies",
        "forces",
        "virial",
    }
    assert outputs["total_energy"].shape == torch.Size([])
    assert outputs["local_energies"].shape == (number_of_atoms(graph),)
    assert outputs["forces"].shape == graph["_positions"].shape
    assert outputs["virial"].shape == (3, 3)

    # 5. test that the deployment process hasn't changed the model's predictions
    with torch.no_grad():
        original_energy = model(graph).double()
    assert torch.allclose(original_energy, outputs["total_energy"])
