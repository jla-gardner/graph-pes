from __future__ import annotations

import helpers
import pytest
import torch
from ase import Atoms
from graph_pes.core import GraphPESModel
from graph_pes.data.io import to_atomic_graph, to_atomic_graphs
from graph_pes.graphs.operations import (
    has_cell,
    number_of_atoms,
    number_of_edges,
    to_batch,
)
from graph_pes.models import LennardJones, Morse, SchNet
from graph_pes.models.addition import AdditionModel

graphs = to_atomic_graphs(helpers.CU_TEST_STRUCTURES, cutoff=3)


def test_model():
    model = LennardJones()
    model.pre_fit_all_components(graphs[:2])

    assert sum(p.numel() for p in model.parameters()) == 2

    predictions = model.get_all_PES_predictions(to_batch(graphs))
    assert "energy" in predictions
    assert "forces" in predictions
    assert "stress" in predictions and has_cell(graphs[0])
    assert predictions["energy"].shape == (len(graphs),)
    assert predictions["stress"].shape == (len(graphs), 3, 3)


def test_isolated_atom():
    atom = Atoms("He", positions=[[0, 0, 0]])
    graph = to_atomic_graph(atom, cutoff=3)
    assert number_of_atoms(graph) == 1 and number_of_edges(graph) == 0

    model = LennardJones()
    assert model.predict_energy(graph) == 0


def test_pre_fit():
    model = LennardJones()
    model.pre_fit_all_components(graphs)

    with pytest.warns(
        UserWarning,
        match="has already been pre-fitted",
    ):
        model.pre_fit_all_components(graphs)


@helpers.parameterise_model_classes(expected_elements=["Cu"])
def test_model_serialisation(model_class: type[GraphPESModel], tmp_path):
    # 1. instantiate the model
    m1 = model_class()  # type: ignore
    m1.pre_fit_all_components(
        graphs
    )  # required by some models before making predictions

    torch.save(m1.state_dict(), tmp_path / "model.pt")

    m2 = model_class()  # type: ignore
    # check no errors occur
    m2.load_state_dict(torch.load(tmp_path / "model.pt"))

    # check predictions are the same
    assert torch.allclose(
        m1.predict_energy(graphs[0]), m2.predict_energy(graphs[0])
    )


def test_cutoff_save_and_load():
    model_4 = SchNet(cutoff=4.0)
    model_5 = SchNet(cutoff=5.0)
    model_5.load_state_dict(model_4.state_dict())
    # cutoff is part of the model's state: as such, it should be
    # saved and loaded as part of a state-dict process
    assert model_5.cutoff == 4.0


def test_addition():
    lj = LennardJones()
    m = Morse()

    # test addition of two models
    addition_model = AdditionModel(lj=lj, morse=m)
    assert torch.allclose(
        addition_model.predict_energy(graphs[0]),
        lj.predict_energy(graphs[0]) + m.predict_energy(graphs[0]),
    )

    # test pre_fit
    original_lj_sigma = lj.sigma.item()
    addition_model.pre_fit_all_components(graphs)
    assert (
        lj.sigma.item() != original_lj_sigma
    ), "component LJ model was not pre-fitted"


@helpers.parameterise_all_models(expected_elements=["Cu"])
def test_model_outputs(model: GraphPESModel):
    graph = graphs[0]
    assert has_cell(graph)
    model.pre_fit_all_components([graph])

    outputs = model.get_all_PES_predictions(graph)
    N = number_of_atoms(graph)

    assert "energy" in outputs and outputs["energy"].shape == ()
    assert "forces" in outputs and outputs["forces"].shape == (N, 3)
    assert "stress" in outputs and outputs["stress"].shape == (3, 3)
    assert "local_energies" in outputs and outputs["local_energies"].shape == (
        N,
    )

    assert torch.allclose(model.predict_energy(graph), outputs["energy"])
    assert torch.allclose(model.predict_forces(graph), outputs["forces"])
    assert torch.allclose(model.predict_stress(graph), outputs["stress"])
    assert torch.allclose(
        model.predict_local_energies(graph), outputs["local_energies"]
    )

    batch = to_batch(graphs[:2])
    batch_outputs = model.get_all_PES_predictions(batch)
    Nbatch = number_of_atoms(batch)
    assert "energy" in batch_outputs and batch_outputs["energy"].shape == (2,)
    assert "forces" in batch_outputs and batch_outputs["forces"].shape == (
        Nbatch,
        3,
    )
    assert "stress" in batch_outputs and batch_outputs["stress"].shape == (
        2,
        3,
        3,
    )
    assert "local_energies" in batch_outputs and batch_outputs[
        "local_energies"
    ].shape == (Nbatch,)

    # ensure that the predictions for the individual graphs are the same
    # as if they were predicted separately
    assert torch.allclose(
        batch_outputs["energy"][0], outputs["energy"], atol=1e-5
    )
    assert torch.allclose(
        batch_outputs["forces"][:N], outputs["forces"], atol=1e-5
    )
    assert torch.allclose(
        batch_outputs["stress"][0], outputs["stress"], atol=1e-5
    )
    assert torch.allclose(
        batch_outputs["local_energies"][:N],
        outputs["local_energies"],
        atol=1e-5,
    )
