from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase.build import molecule
from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import number_of_atoms
from graph_pes.models.pairwise import LennardJones, SmoothedPairPotential
from graph_pes.utils.lammps import as_lammps_data, deploy_model

from .. import helpers


# ignore warnings about lack of energy labels for pre-fitting: not important
@pytest.mark.filterwarnings("ignore:.*No energy data found in training data.*")
@helpers.parameterise_all_models(expected_elements=["C", "H", "O"])
def test_deploy(model: GraphPESModel, tmp_path: Path):
    dummy_graph = AtomicGraph.from_ase(molecule("CH3CH2OH"), cutoff=1.5)
    # required by some models before making predictions
    model.pre_fit_all_components([dummy_graph])

    model_cutoff = float(model.cutoff)
    graph = AtomicGraph.from_ase(
        molecule("CH3CH2OH", vacuum=2),
        cutoff=model_cutoff,
    )
    outputs = {
        k: t.double() for k, t in model.get_all_PES_predictions(graph).items()
    }

    # 1. saving and unsaving works
    torch.save(model, tmp_path / "model.pt")
    loaded_model = torch.load(tmp_path / "model.pt")
    assert isinstance(loaded_model, GraphPESModel)
    torch.testing.assert_close(
        model.predict_forces(graph),
        loaded_model.predict_forces(graph),
        atol=1e-6,
        rtol=1e-6,
    )

    # 2. deploy the model
    save_path = tmp_path / "model.pt"
    deploy_model(model, path=save_path)

    # 3. load the model back in
    loaded_model = torch.jit.load(save_path)
    assert isinstance(loaded_model, torch.jit.ScriptModule)
    assert loaded_model.get_cutoff() == model_cutoff

    # 4. test outputs
    lammps_data = as_lammps_data(graph, compute_virial=True)
    loaded_outputs = loaded_model(lammps_data)
    assert isinstance(loaded_outputs, dict)
    assert set(loaded_outputs.keys()) == {
        "energy",
        "local_energies",
        "forces",
        "virial",
        "stress",
    }
    assert loaded_outputs["energy"].shape == torch.Size([])
    torch.testing.assert_close(
        outputs["energy"],
        loaded_outputs["energy"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert loaded_outputs["local_energies"].shape == (number_of_atoms(graph),)
    torch.testing.assert_close(
        outputs["local_energies"],
        loaded_outputs["local_energies"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert loaded_outputs["forces"].shape == graph.R.shape
    torch.testing.assert_close(
        outputs["forces"],
        loaded_outputs["forces"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert loaded_outputs["stress"].shape == (3, 3)
    torch.testing.assert_close(
        outputs["stress"],
        loaded_outputs["stress"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert loaded_outputs["virial"].shape == (6,)


def test_deploy_smoothed_pair_potential(tmp_path: Path):
    model = SmoothedPairPotential(LennardJones(cutoff=2.5))
    test_deploy(model, tmp_path)
