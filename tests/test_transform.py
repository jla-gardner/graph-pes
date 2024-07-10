from __future__ import annotations

import pytest
import torch
from ase import Atoms
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs.operations import is_local_property
from graph_pes.transform import Chain, DividePerAtom, Identity, Scale

structure = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
structure.info["energy"] = -1.0
structure.arrays["forces"] = torch.zeros((2, 3))
graph = to_atomic_graph(structure, cutoff=1.5)


def test_identity():
    transform = Identity()
    x = torch.arange(10).float()

    assert transform.forward(x, graph).equal(x)


def test_is_local_property():
    energy = graph["energy"]
    assert not is_local_property(energy, graph)

    forces = graph["forces"]
    assert is_local_property(forces, graph)


def test_scale():
    x = torch.tensor([1.3, 2.1])
    std = x.std()

    transform = Scale()

    # once fitted, y should have unit std
    transform.fit(x, graph)
    y = transform(x, graph)
    assert torch.allclose(y, x / std)

    with pytest.warns(UserWarning, match="has already been fitted"):
        transform.fit(x, graph)


def test_chain():
    transform = Chain(Scale(scale=5), DividePerAtom())

    x = torch.tensor([1.0, 2.0])
    y = transform(x, graph)

    # y = (x * 5) / 2
    assert torch.allclose(y, x * 5 / 2)
