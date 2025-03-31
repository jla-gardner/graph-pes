from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase import Atoms

import graph_pes
from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import (
    has_cell,
    number_of_atoms,
    number_of_edges,
    to_batch,
)
from graph_pes.models import (
    EDDP,
    LennardJones,
    Morse,
    SchNet,
    load_model,
    load_model_component,
)
from graph_pes.models.addition import AdditionModel

from .. import helpers

graphs = [
    AtomicGraph.from_ase(atoms, cutoff=3)
    for atoms in helpers.CU_TEST_STRUCTURES
]


def test_model():
    model = LennardJones(cutoff=3.0)
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
    graph = AtomicGraph.from_ase(atom, cutoff=3)
    assert number_of_atoms(graph) == 1 and number_of_edges(graph) == 0

    model = LennardJones(cutoff=3.0)
    assert model.predict_energy(graph) == 0


def test_pre_fit(caplog):
    model = LennardJones(cutoff=3.0)
    model.pre_fit_all_components(graphs)

    model.pre_fit_all_components(graphs)
    assert any(
        record.levelname == "WARNING"
        and "has already been pre-fitted" in record.message
        for record in caplog.records
    )


@helpers.parameterise_model_classes(expected_elements=["Cu"], cutoff=3.0)
def test_model_serialisation(model_class: type[GraphPESModel], tmp_path):
    # 1. instantiate the model
    m1 = model_class()  # type: ignore
    m1.pre_fit_all_components(
        graphs
    )  # required by some models before making predictions

    torch.save(m1.state_dict(), tmp_path / "model.pt")

    m2 = model_class()  # type: ignore
    # check no errors occur
    m2.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=False))

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
    lj = LennardJones(cutoff=3.0)
    m = Morse(cutoff=3.0)

    # test addition of two models
    addition_model = AdditionModel(lj=lj, morse=m)
    assert torch.allclose(
        addition_model.predict_energy(graphs[0]),
        lj.predict_energy(graphs[0]) + m.predict_energy(graphs[0]),
    )

    assert torch.allclose(
        addition_model.predict_forces(graphs[0]),
        lj.predict_forces(graphs[0]) + m.predict_forces(graphs[0]),
    )

    assert torch.allclose(
        addition_model.predict_stress(graphs[0]),
        lj.predict_stress(graphs[0]) + m.predict_stress(graphs[0]),
    )


def test_addition_cutoffs():
    lj = LennardJones(cutoff=4.0)
    eddp = EDDP(cutoff=3.0, elements=["Cu"])
    addition_model = AdditionModel(lj=lj, eddp=eddp)

    # check that the addition model has the largest cutoff
    assert torch.allclose(torch.tensor(4.0), addition_model.cutoff)

    #  ensure that 3 body cuttoff is the same as the eddp cutoff
    assert torch.allclose(torch.tensor(3.0), eddp.three_body_cutoff)
    assert torch.allclose(torch.tensor(3.0), addition_model.three_body_cutoff)


@helpers.parameterise_all_models(expected_elements=["Cu"], cutoff=3.0)
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


def test_load_model(tmp_path: Path):
    model = LennardJones()

    # test correctness
    path = tmp_path / "model.pt"
    torch.save(model, path)
    loaded = load_model(path)
    assert model.sigma == loaded.sigma

    # test warning
    fake_version = graph_pes.__version__ + "fake"
    model._GRAPH_PES_VERSION = fake_version  # type: ignore
    torch.save(model, path)
    with pytest.warns(UserWarning, match="different version of graph-pes"):
        loaded = load_model(path)
        assert loaded._GRAPH_PES_VERSION == fake_version

    # def test error
    torch.save({}, path)
    with pytest.raises(ValueError, match="to be a GraphPESModel but got"):
        load_model(path)


def test_load_model_component(tmp_path: Path):
    model = AdditionModel(lj=LennardJones(), schnet=SchNet())
    path = tmp_path / "model.pt"
    torch.save(model, path)

    lj = load_model_component(path, "lj")
    assert lj.sigma == model["lj"].sigma

    # test error
    torch.save(LennardJones(), path)
    with pytest.raises(ValueError, match="Expected to load an AdditionModel"):
        load_model_component(path, "str")
