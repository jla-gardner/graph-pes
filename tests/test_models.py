from __future__ import annotations

import helpers
import pytest
import torch
from ase import Atoms
from ase.io import read
from graph_pes.core import GraphPESModel, get_predictions
from graph_pes.data.io import to_atomic_graph, to_atomic_graphs
from graph_pes.graphs.operations import (
    has_cell,
    number_of_atoms,
    number_of_edges,
)
from graph_pes.models import LennardJones, Morse
from graph_pes.models.addition import AdditionModel

structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
graphs = to_atomic_graphs(structures, cutoff=3)


def test_model():
    model = LennardJones()
    model.pre_fit(graphs[:2])

    assert sum(p.numel() for p in model.parameters()) == 2

    predictions = get_predictions(model, graphs)
    assert "energy" in predictions
    assert "forces" in predictions
    assert "stress" in predictions and has_cell(graphs[0])
    assert predictions["energy"].shape == (len(graphs),)
    assert predictions["stress"].shape == (len(graphs), 3, 3)

    energy = model(graphs[0])
    assert torch.equal(
        get_predictions(model, graphs[0], property="energy"),
        energy,
    )


def test_isolated_atom():
    atom = Atoms("He", positions=[[0, 0, 0]])
    graph = to_atomic_graph(atom, cutoff=3)
    assert number_of_atoms(graph) == 1 and number_of_edges(graph) == 0

    model = LennardJones()
    assert model(graph) == 0


def test_pre_fit():
    model = LennardJones()
    model.pre_fit(graphs)

    with pytest.warns(
        UserWarning,
        match="has already been pre-fitted",
    ):
        model.pre_fit(graphs)


@helpers.parameterise_model_classes(["Cu"])
def test_model_serialisation(model_class: type[GraphPESModel], tmp_path):
    # 1. instantiate the model
    m1 = model_class()
    m1.pre_fit(graphs)  # required by some models before making predictions

    torch.save(m1.state_dict(), tmp_path / "model.pt")

    m2 = model_class()
    # check no errors occur
    m2.load_state_dict(torch.load(tmp_path / "model.pt"))

    # check predictions are the same
    assert torch.allclose(m1(graphs[0]), m2(graphs[0]))


def test_addition():
    lj = LennardJones()
    m = Morse()

    # test addition of two models
    addition_model = AdditionModel(lj=lj, morse=m)
    assert torch.allclose(
        addition_model(graphs[0]),
        lj(graphs[0]) + m(graphs[0]),
    )

    # test pre_fit
    original_lj_sigma = lj.sigma.item()
    addition_model.pre_fit(graphs)
    assert (
        lj.sigma.item() != original_lj_sigma
    ), "component LJ model was not pre-fitted"
