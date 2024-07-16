from __future__ import annotations

import helpers
import numpy as np
import pytest
import torch
from ase import Atoms
from graph_pes.core import get_predictions
from graph_pes.data.io import to_atomic_graphs
from graph_pes.graphs.operations import number_of_atoms, to_batch
from graph_pes.models.offsets import EnergyOffset, FixedOffset, LearnableOffset

graphs = to_atomic_graphs(helpers.CU_TEST_STRUCTURES, cutoff=3)


@pytest.mark.parametrize(
    "offset_model,trainable",
    [
        (FixedOffset(He=1, Cu=-3), False),
        (LearnableOffset(He=1, Cu=-3), True),
    ],
)
def test_offset_behaviour(offset_model: EnergyOffset, trainable: bool):
    assert offset_model._offsets.requires_grad == trainable
    total_parameters = sum(p.numel() for p in offset_model.parameters())
    assert (
        total_parameters == 2
    ), "expected 2 parameters (energy offsets for He and Cu)"

    assert (
        offset_model._offsets[2] == 1
    ), "expected offset for He to be as specified"

    graph = graphs[0]
    n = number_of_atoms(graph)

    assert offset_model.predict_local_energies(graph).shape == (n,)

    if not trainable:
        with pytest.warns(UserWarning, match="no grad function"):
            predictions = get_predictions(offset_model, graph, training=True)
    else:
        predictions = get_predictions(offset_model, graph, training=True)

    assert "energy" in predictions
    # total energy is the sum of offsets of all atoms, which are Cu
    assert predictions["energy"].item() == n * -3
    if trainable:
        assert predictions["energy"].grad_fn is not None, (
            "expected gradients on the energy calculation "
            "due to using trainable offsets"
        )

    # no interactions between atoms, so forces are 0
    assert torch.all(predictions["forces"] == 0)


def test_energy_offset_fitting():
    # create some fake structures to test shift and scale fitting

    structures = []

    # (num H atoms, num C atoms)
    nums = [(3, 0), (4, 8), (5, 2)]
    H_energy, C_energy = -4.5, -10.0

    for n_H, n_C in nums:
        atoms = Atoms("H" * n_H + "C" * n_C)
        atoms.info["energy"] = n_H * H_energy + n_C * C_energy
        atoms.arrays["forces"] = np.zeros((n_H + n_C, 3))
        structures.append(atoms)

    graphs = to_atomic_graphs(structures, cutoff=1.5)
    batch = to_batch(graphs)
    assert "energy" in batch

    model = LearnableOffset()
    model.pre_fit(graphs)
    # check that the model has learned the correct energy offsets
    # use pytest.approx to account for numerical errors
    assert model._offsets[1].item() == pytest.approx(H_energy)
    assert model._offsets[6].item() == pytest.approx(C_energy)

    # check that initial values aren't overwritten if specified
    model = LearnableOffset(H=20)
    model.pre_fit(graphs)
    assert model._offsets[1].item() == pytest.approx(20)
    assert model._offsets[6].item() == 0

    # check suitable warning logged if no energy data
    model = LearnableOffset()
    graph = dict(graphs[0])
    del graph["energy"]
    with pytest.warns(
        UserWarning,
        match="No energy labels found in the training data",
    ):
        model.pre_fit([graph])  # type: ignore
