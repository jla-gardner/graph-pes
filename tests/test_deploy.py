from __future__ import annotations

from pathlib import Path

import helpers
import pytest
import torch
from ase.build import molecule
from graph_pes.core import GraphPESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.deploy import deploy_model
from graph_pes.graphs.operations import number_of_atoms
from graph_pes.models.pairwise import LennardJones, SmoothedPairPotential


# ignore warnings about lack of energy labels for pre-fitting: not important
@pytest.mark.filterwarnings(
    "ignore:.*training data does not contain energy labels.*"
)
@helpers.parameterise_all_models(expected_elements=["C", "H", "O"])
def test_deploy(model: GraphPESModel, tmp_path: Path):
    dummy_graph = to_atomic_graph(molecule("CH3CH2OH"), cutoff=1.5)
    # required by some models before making predictions
    model.pre_fit([dummy_graph])

    model_cutoff = float(model.cutoff)
    graph = to_atomic_graph(
        molecule("CH3CH2OH", vacuum=2),
        cutoff=model_cutoff,
    )

    # 2. deploy the model
    save_path = tmp_path / "model.pt"
    deploy_model(model, path=save_path)

    # 3. load the model back in
    loaded_model = torch.jit.load(save_path)
    assert isinstance(loaded_model, torch.jit.ScriptModule)
    assert loaded_model.get_cutoff() == model_cutoff

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
        "energy",
        "local_energies",
        "forces",
        "virial",
        "stress",
    }
    assert outputs["energy"].shape == torch.Size([])
    assert outputs["local_energies"].shape == (number_of_atoms(graph),)
    assert outputs["forces"].shape == graph["_positions"].shape
    assert outputs["stress"].shape == (3, 3)
    assert outputs["virial"].shape == (6,)

    # 5. test that the deployment process hasn't changed the model's predictions
    with torch.no_grad():
        original_energy = model(graph).double()
    assert torch.allclose(original_energy, outputs["energy"])


def test_deploy_smoothed_pair_potential(tmp_path: Path):
    model = SmoothedPairPotential(LennardJones(cutoff=2.5))
    test_deploy(model, tmp_path)
