from __future__ import annotations

import torch
from ase import Atoms
from ase.io import read
from graph_pes.core import Ensemble, get_predictions
from graph_pes.data import (
    has_cell,
    number_of_atoms,
    number_of_edges,
    to_atomic_graph,
    to_atomic_graphs,
    to_batch,
)
from graph_pes.models.zoo import LennardJones, Morse

structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
graphs = to_atomic_graphs(structures, cutoff=3)


def test_model():
    model = LennardJones()
    model.pre_fit(to_batch(graphs[:2]))  # type: ignore

    assert sum(p.numel() for p in model.parameters()) == 3

    predictions = get_predictions(model, graphs)
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
    assert model(graph) == 0


def test_ensembling():
    lj = LennardJones()
    morse = Morse()
    addition_model = lj + morse
    assert addition_model(graphs[0]) == lj(graphs[0]) + morse(graphs[0])

    mean_model = Ensemble([lj, morse], aggregation="mean", weights=[1.2, 5.7])
    assert torch.allclose(
        mean_model(graphs[0]),
        (1.2 * lj(graphs[0]) + 5.7 * morse(graphs[0])) / (1.2 + 5.7),
    )
