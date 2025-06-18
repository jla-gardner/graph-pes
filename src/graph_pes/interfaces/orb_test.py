import ase
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from orb_models.forcefield import atomic_system
from orb_models.forcefield.base import batch_graphs

from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.interfaces._orb import orb_model
from graph_pes.utils.misc import full_3x3_to_voigt_6


@pytest.fixture(params=["orb-v3-direct-20-omat", "orb-v3-conservative-20-omat"])
def wrapped_orb(request):
    return orb_model(request.param)


def test_single_isolated_structure(wrapped_orb):
    atoms = molecule("H2O")

    g = AtomicGraph.from_ase(atoms, cutoff=wrapped_orb.cutoff.item())
    our_preds = wrapped_orb.forward(g)

    assert our_preds["energy"].shape == tuple()
    assert our_preds["forces"].shape == (3, 3)

    orb_g = atomic_system.ase_atoms_to_atom_graphs(
        atoms, wrapped_orb.orb_model.system_config
    )
    orb_preds = wrapped_orb.orb_model.predict(orb_g)

    torch.testing.assert_close(our_preds["energy"], orb_preds["energy"][0])
    torch.testing.assert_close(
        our_preds["forces"],
        orb_preds["forces"]
        if "forces" in orb_preds
        else orb_preds["grad_forces"],
    )


def test_single_periodic_structure(wrapped_orb):
    atoms = bulk("Cu")
    g = AtomicGraph.from_ase(atoms, cutoff=wrapped_orb.cutoff.item())
    our_preds = wrapped_orb.forward(g)

    assert our_preds["energy"].shape == tuple()
    assert our_preds["forces"].shape == (1, 3)
    assert our_preds["stress"].shape == (3, 3)

    orb_g = atomic_system.ase_atoms_to_atom_graphs(
        atoms, wrapped_orb.orb_model.system_config
    )
    orb_preds = wrapped_orb.orb_model.predict(orb_g)

    torch.testing.assert_close(
        our_preds["energy"],
        orb_preds["energy"][0],
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        our_preds["forces"],
        orb_preds["forces"]
        if "forces" in orb_preds
        else orb_preds["grad_forces"],
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        full_3x3_to_voigt_6(our_preds["stress"]),
        orb_preds["stress"][0]
        if "stress" in orb_preds
        else orb_preds["grad_stress"][0],
        atol=1e-3,
        rtol=1e-3,
    )


def test_batched(wrapped_orb):
    atoms = bulk("Cu").repeat(2)
    rng = np.random.RandomState(42)
    atoms.positions += rng.uniform(-0.1, 0.1, atoms.positions.shape)

    N = len(atoms)
    B = 2

    g = AtomicGraph.from_ase(atoms, cutoff=wrapped_orb.cutoff.item())
    batch = to_batch([g] * B)
    our_preds = wrapped_orb.forward(batch)

    assert our_preds["energy"].shape == (B,)
    assert our_preds["forces"].shape == (B * N, 3)
    assert our_preds["stress"].shape == (B, 3, 3)

    orb_g = atomic_system.ase_atoms_to_atom_graphs(
        atoms, wrapped_orb.orb_model.system_config
    )
    orb_g = batch_graphs([orb_g] * B)
    orb_preds = wrapped_orb.orb_model.predict(orb_g)

    torch.testing.assert_close(our_preds["energy"], orb_preds["energy"])
    torch.testing.assert_close(
        our_preds["forces"],
        orb_preds["forces"]
        if "forces" in orb_preds
        else orb_preds["grad_forces"],
    )
    torch.testing.assert_close(
        full_3x3_to_voigt_6(our_preds["stress"]),
        orb_preds["stress"]
        if "stress" in orb_preds
        else orb_preds["grad_stress"],
    )


def test_single_atom(wrapped_orb):
    wrapped_orb.ase_calculator().calculate(ase.Atoms("H"))
