from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.data import chemical_symbols

from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.models import FixedOffset
from graph_pes.models.addition import AdditionModel
from graph_pes.utils.shift_and_scale import (
    add_auto_offset,
    guess_per_element_mean_and_var,
)


def _get_random_graphs(
    reference: Mapping[str, float | int],
) -> list[AtomicGraph]:
    rng = np.random.default_rng(42)
    structures = []
    for _ in range(100):
        symbols = rng.choice(list(reference.keys()), size=10)
        energy = sum(reference[Z] for Z in symbols)
        atoms = Atoms(symbols=symbols)
        atoms.info["energy"] = energy
        structures.append(atoms)
    return [AtomicGraph.from_ase(s, cutoff=0.1) for s in structures]


def test_add_auto_offset():
    # step 1: make a collection of structures with energies as the
    #         sum of known per-element energies
    reference = dict(C=-1, H=-2, O=-3)
    graphs = _get_random_graphs(reference)

    # step 2: ensure the guessed offsets are close
    means, _ = guess_per_element_mean_and_var(
        torch.tensor([g.properties["energy"] for g in graphs]),
        to_batch(graphs),
    )
    means = {chemical_symbols[Z]: float(mu) for Z, mu in means.items()}
    for k in reference:
        assert means[k] == pytest.approx(reference[k], abs=1e-5)

    # step 3: take an existing model with different offsets, and check
    #         that the guessed difference is close the actual difference
    model_reference = dict(C=2, H=3, O=4)
    starting_model = FixedOffset(**model_reference)

    final_model = add_auto_offset(starting_model, graphs)
    assert isinstance(final_model, AdditionModel)
    assert set(final_model.models.keys()) == {"auto_offset", "base"}
    final_model.eval()

    for g in graphs:
        assert final_model.predict_energy(g).item() == pytest.approx(
            g.properties["energy"].item(), abs=1e-4
        )

    starting_model = AdditionModel(offset=FixedOffset(**model_reference))
    final_model = add_auto_offset(starting_model, graphs)
    assert isinstance(final_model, AdditionModel)
    assert set(final_model.models.keys()) == {"auto_offset", "offset"}


def test_add_auto_offset_no_op():
    reference = dict(C=-1, H=-2, O=-3)
    graphs = _get_random_graphs(reference)

    starting_model = FixedOffset(**reference)
    final_model = add_auto_offset(starting_model, graphs)
    # this should be a no-op
    assert final_model is starting_model


def test_warning(caplog):
    reference = dict(C=-1, H=-2, O=-3)

    # get structures with the same composition
    rng = np.random.default_rng(42)
    graphs = []
    for _ in range(100):
        N = rng.integers(1, 10)
        symbols = ["H", "H", "C", "O"] * N
        atoms = Atoms(symbols=symbols)
        atoms.info["energy"] = sum(reference[Z] for Z in symbols)
        graphs.append(AtomicGraph.from_ase(atoms, cutoff=0.1))

    model_reference = dict(C=2, H=3, O=4)
    starting_model = FixedOffset(**model_reference)

    # check that the warning is issued
    with caplog.at_level(logging.WARNING):
        final_model = add_auto_offset(starting_model, graphs)
        assert "no unique solution is possible" in caplog.text

    # check that things still work
    final_model.eval()
    for g in graphs:
        assert final_model.predict_energy(g).item() == pytest.approx(
            g.properties["energy"].item(), abs=1e-4
        )
