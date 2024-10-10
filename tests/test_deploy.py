from __future__ import annotations

from pathlib import Path

import helpers
import pytest
import torch
from ase.build import molecule
from graph_pes.core import ConservativePESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.deploy import deploy_model
from graph_pes.graphs.operations import number_of_atoms
from graph_pes.models.pairwise import LennardJones, SmoothedPairPotential

CUTOFF = 1.5
graph = to_atomic_graph(molecule("CH3CH2OH"), cutoff=CUTOFF)


# ignore warnings about lack of energy labels for pre-fitting: not important
@pytest.mark.filterwarnings(
    "ignore:.*training data does not contain energy labels.*"
)
@helpers.parameterise_all_models(expected_elements=["C", "H", "O"])
def test_deploy(model: ConservativePESModel, tmp_path: Path):
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


def test_deploy_smoothed_pair_potential(tmp_path: Path):
    model = SmoothedPairPotential(LennardJones(cutoff=CUTOFF))
    test_deploy(model, tmp_path)
