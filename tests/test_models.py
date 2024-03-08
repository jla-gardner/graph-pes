from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.io import read
from graph_pes.core import Ensemble, GraphPESModel, get_predictions
from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    has_cell,
    number_of_atoms,
    number_of_edges,
    to_atomic_graph,
    to_atomic_graphs,
    to_batch,
)
from graph_pes.models.zoo import LennardJones, Morse
from graph_pes.transform import PerAtomShift

structures: list[Atoms] = read("tests/test.xyz", ":")  # type: ignore
graphs = to_atomic_graphs(structures, cutoff=3)


def test_model():
    model = LennardJones()
    model.pre_fit(graphs[:2])

    assert sum(p.numel() for p in model.parameters()) == 3

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


def test_pre_fit():
    model = LennardJones()
    model.pre_fit(graphs)

    with pytest.warns(
        UserWarning,
        match="This model has already been pre-fitted",
    ):
        model.pre_fit(graphs)

    batch = to_batch(graphs)
    batch.pop("energy")  # type: ignore
    with pytest.warns(
        UserWarning,
        match="The training data doesn't contain energies.",
    ):
        LennardJones().pre_fit(batch)

    for ret_value in True, False:
        # make sure energy transform is not called if return from _extra_pre_fit
        class DummyModel(GraphPESModel):
            def __init__(self):
                super().__init__(energy_transform=PerAtomShift())

            def predict_local_energies(
                self, graph: AtomicGraph
            ) -> torch.Tensor:
                return torch.ones(number_of_atoms(graph))

            def _extra_pre_fit(self, graphs: AtomicGraphBatch) -> bool | None:
                return ret_value  # noqa: B023

        model = DummyModel()
        assert model.energy_transform.shift[29] == 0
        model.pre_fit(graphs)
        if ret_value:
            assert model.energy_transform.shift[29] == 0
        else:
            assert model.energy_transform.shift[29] != 0
