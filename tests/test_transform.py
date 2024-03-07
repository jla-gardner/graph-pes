from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from graph_pes.data import (
    batch_graphs,
    convert_to_atomic_graph,
    convert_to_atomic_graphs,
    is_local_property,
)
from graph_pes.transform import (
    DividePerAtom,
    Identity,
    PerAtomScale,
    PerAtomShift,
    PerAtomStandardScaler,
    Scale,
    Transform,
)

structure = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
structure.info["energy"] = -1.0
structure.arrays["forces"] = torch.zeros((2, 3))
graph = convert_to_atomic_graph(structure, cutoff=1.5)


def test_identity():
    transform = Identity()
    x = torch.arange(10).float()

    assert transform.forward(x, graph).equal(x)
    assert transform.inverse(x, graph).equal(x)


def test_is_local_property():
    energy = graph["energy"]
    assert not is_local_property(energy, graph)

    forces = graph["forces"]
    assert is_local_property(forces, graph)


def test_per_atom_transforms():
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

    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)
    batch = batch_graphs(graphs)

    # fit shift to the total energies
    shift = PerAtomShift(trainable=False)
    total_energies: torch.Tensor = batch["energy"]  # type: ignore
    shift.fit_to_source(total_energies, batch)
    shifted_total_energies = shift(total_energies, batch)

    # shape preservation
    assert shifted_total_energies.shape == total_energies.shape
    # learn the correct shifts
    assert torch.allclose(
        shift.shift[torch.tensor([1, 6])].detach().squeeze(),
        torch.tensor([H_energy, C_energy]),
    )

    centered_energy = shift(total_energies, batch)
    assert torch.allclose(
        centered_energy,
        torch.zeros_like(centered_energy),
        atol=1e-5,
    )
    assert not centered_energy.requires_grad

    # test scaling forces
    scale = PerAtomScale(trainable=False)
    forces = batch["forces"]
    scale.fit_to_source(forces, batch)

    scaled_forces = scale(forces, batch)
    assert scaled_forces.shape == forces.shape


transforms = [
    PerAtomShift(),
    PerAtomScale(),
    DividePerAtom(),
    PerAtomStandardScaler(),
    Scale(),
]


@pytest.mark.parametrize(
    "transform",
    transforms,
    ids=[transform.__class__.__name__ for transform in transforms],
)
def test_inverse(transform: Transform):
    x = torch.tensor([1.0, 2.0])
    y = transform(x, graph)

    inverse = transform.inverse(y, graph)
    assert x.equal(inverse)
