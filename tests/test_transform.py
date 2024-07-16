from __future__ import annotations

import torch
from ase import Atoms
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs.operations import is_local_property
from graph_pes.transform import divide_per_atom, identity

structure = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
structure.info["energy"] = -1.0
structure.arrays["forces"] = torch.zeros((2, 3))
graph = to_atomic_graph(structure, cutoff=1.5)


def test_identity():
    x = torch.arange(10).float()
    assert identity(x, graph).equal(x)


def test_divide_per_atom():
    y = divide_per_atom(graph["energy"], graph)
    assert y.equal(torch.tensor(-0.5))


def test_is_local_property():
    energy = graph["energy"]
    assert not is_local_property(energy, graph)

    forces = graph["forces"]
    assert is_local_property(forces, graph)
