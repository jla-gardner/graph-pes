from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from graph_pes.atomic_graph import (
    AtomicGraph,
    edges_per_graph,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
    number_of_structures,
    register_custom_batcher,
    structure_sizes,
    sum_per_structure,
    to_batch,
)
from graph_pes.data.loader import GraphDataLoader

STRUCTURES = [
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    Atoms("H3", positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2)], pbc=False),
]
GRAPHS = [AtomicGraph.from_ase(s, cutoff=1.5) for s in STRUCTURES]


def test_batching():
    batch = to_batch(GRAPHS)
    assert batch.batch is not None
    assert batch.ptr is not None

    assert number_of_atoms(batch) == 5
    assert number_of_structures(batch) == 2
    assert list(batch.ptr) == [0, 2, 5]
    assert list(batch.batch) == [0, 0, 1, 1, 1]

    assert number_of_edges(batch) == sum(number_of_edges(g) for g in GRAPHS)
    assert structure_sizes(batch).tolist() == [2, 3]
    assert neighbour_vectors(batch).shape == (number_of_edges(batch), 3)


def test_label_batching():
    structures = [s.copy() for s in STRUCTURES]
    # per-structure labels:
    structures[0].info["energy"] = 0
    structures[0].info["stress"] = np.eye(3)
    structures[1].info["energy"] = 1
    structures[1].info["stress"] = 2 * np.eye(3)

    # per-atom labels:
    structures[0].arrays["forces"] = np.zeros((2, 3))
    structures[1].arrays["forces"] = np.zeros((3, 3))

    graphs = [AtomicGraph.from_ase(s, cutoff=1.5) for s in structures]
    batch = to_batch(graphs)

    # per-structure, array-type labels are concatenated along a new batch axis
    assert batch.properties["stress"].shape == (2, 3, 3)
    # energy is a scalar, so the "new batch axis" is just concatenation:
    assert batch.properties["energy"].tolist() == [0, 1]

    # per-atom labels are concatenated along the first axis
    assert batch.properties["forces"].shape == (5, 3)


def test_pbcs():
    # structure 1: 1 atom in a (1x1x1) cell
    # structure 2: 1 atom in a (2x2x2) cell
    structures = [
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(1, 1, 1)),
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(2, 2, 2)),
    ]
    graphs = [AtomicGraph.from_ase(s, cutoff=1.2) for s in structures]

    assert number_of_edges(graphs[0]) == 6
    assert number_of_edges(graphs[1]) == 0

    batch = to_batch(graphs)
    assert number_of_edges(batch) == 6
    assert batch.cell.shape == (2, 3, 3)

    assert neighbour_vectors(batch).shape == (6, 3)
    assert neighbour_distances(batch).tolist() == [1.0] * 6


def test_sum_per_structure():
    structures = [s.copy() for s in STRUCTURES]
    graphs = [AtomicGraph.from_ase(s, cutoff=1.5) for s in structures]

    # sum per structure should work for:

    # 1. a single structure
    x = torch.tensor([1, 2])
    assert sum_per_structure(x, graphs[0]) == x.sum()

    # 2. a batch of structures
    batch = to_batch(graphs)
    x = torch.tensor([1, 2, 3, 4, 5])
    assert sum_per_structure(x, batch).tolist() == [3, 12]

    # and also for general sizes

    x = torch.ones(2, 3, 4)
    result = sum_per_structure(x, graphs[0])
    assert result.shape == (3, 4)


def test_data_loader():
    loader = GraphDataLoader(GRAPHS, batch_size=2)

    num_batches = 0

    for batch in loader:
        assert number_of_atoms(batch) == 5
        assert number_of_structures(batch) == 2
        assert number_of_edges(batch) == 6
        assert structure_sizes(batch).tolist() == [2, 3]
        assert neighbour_vectors(batch).shape == (6, 3)

        num_batches += 1

    assert num_batches == 1

    # test warning if try to pass own collate function
    with pytest.warns(UserWarning):
        GraphDataLoader(GRAPHS, batch_size=2, collate_fn=lambda x: x)


def test_number_of_structures():
    assert number_of_structures(to_batch(GRAPHS)) == len(GRAPHS)
    assert number_of_structures(GRAPHS[0]) == 1


def test_batching_of_other():
    graphs = [AtomicGraph(*g) for g in GRAPHS]
    for g in graphs:
        g.other["foo"] = torch.tensor([1, 2, 3], dtype=torch.long)

    batch = to_batch(graphs)
    assert batch.other["foo"].shape == (2, 3)
    assert batch.other["foo"].dtype == torch.long


def test_custom_batching():
    graphs = [AtomicGraph(*g) for g in GRAPHS]
    graphs[0].other["foo"] = torch.tensor([1, 2, 3], dtype=torch.long)
    graphs[1].other["foo"] = torch.tensor([4, 5, 6], dtype=torch.long)

    @register_custom_batcher("foo")
    def batcher(batch, values):
        return torch.vstack(values).max(dim=0).values

    batch = to_batch(graphs)
    assert batch.other["foo"].shape == (3,)
    assert batch.other["foo"].dtype == torch.long


def test_automatic_batching():
    graphs = [AtomicGraph.from_ase(s, cutoff=1.5) for s in STRUCTURES]
    graphs[0].other["local_property"] = torch.zeros((2, 6), dtype=torch.float)
    graphs[1].other["local_property"] = torch.zeros((3, 6), dtype=torch.float)
    graphs[0].other["global_property"] = torch.zeros(5, dtype=torch.bool)
    graphs[1].other["global_property"] = torch.zeros(5, dtype=torch.bool)

    batch = to_batch(graphs)
    assert batch.other["local_property"].shape == (5, 6)
    assert batch.other["local_property"].dtype == torch.float
    assert batch.other["global_property"].shape == (2, 5)
    assert batch.other["global_property"].dtype == torch.bool


def test_edges_per_graph():
    _0_edge = AtomicGraph.from_ase(
        Atoms("H", positions=[(0, 0, 0)], pbc=False),
        cutoff=5.0,
    )
    _2_egde = AtomicGraph.from_ase(
        Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
        cutoff=5.0,
    )
    _6_edge = AtomicGraph.from_ase(
        Atoms("H3", positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2)], pbc=False),
        cutoff=5.0,
    )
    assert number_of_edges(_0_edge) == 0
    assert number_of_edges(_2_egde) == 2
    assert number_of_edges(_6_edge) == 6

    assert edges_per_graph(_6_edge).tolist() == [6]

    batch = to_batch([_0_edge, _2_egde, _6_edge])
    assert edges_per_graph(batch).tolist() == [0, 2, 6]
    assert number_of_edges(batch) == 8
