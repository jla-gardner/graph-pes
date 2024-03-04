import numpy as np
import pytest
from ase import Atoms
from graph_pes.data import (
    AtomicDataLoader,
    batch_graphs,
    convert_to_atomic_graphs,
    keys,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
    number_of_edges,
    number_of_structures,
    structure_sizes,
    sum_per_structure,
)

STRUCTURES = [
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    Atoms("H3", positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2)], pbc=False),
]
GRAPHS = convert_to_atomic_graphs(STRUCTURES, cutoff=1.5)


def test_batching():
    batch = batch_graphs(GRAPHS)
    assert number_of_atoms(batch) == 5
    assert number_of_structures(batch) == 2
    assert list(batch["ptr"]) == [0, 2, 5]
    assert list(batch["batch"]) == [0, 0, 1, 1, 1]

    assert number_of_edges(batch) == sum(number_of_edges(g) for g in GRAPHS)
    assert structure_sizes(batch).tolist() == [2, 3]
    assert neighbour_vectors(batch).shape == (number_of_edges(batch), 3)


def test_label_batching():
    structures = [s.copy() for s in STRUCTURES]
    # per-structure labels:
    structures[0].info[keys.ENERGY] = 0
    structures[0].info[keys.STRESS] = np.eye(3)
    structures[1].info[keys.ENERGY] = 1
    structures[1].info[keys.STRESS] = 2 * np.eye(3)

    # per-atom labels:
    structures[0].arrays[keys.LOCAL_ENERGIES] = [0, 1]
    structures[0].arrays[keys.FORCES] = np.zeros((2, 3))
    structures[1].arrays[keys.LOCAL_ENERGIES] = [2, 3, 4]
    structures[1].arrays[keys.FORCES] = np.zeros((3, 3))

    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)
    batch = batch_graphs(graphs)

    # per-structure, array-type labels are concatenated along a new batch axis
    assert batch[keys.STRESS].shape == (2, 3, 3)
    # energy is a scalar, so the "new batch axis" is just concatenation:
    assert batch[keys.ENERGY].tolist() == [0, 1]

    # per-atom labels are concatenated along the first axis
    assert batch[keys.LOCAL_ENERGIES].tolist() == [0, 1, 2, 3, 4]
    assert batch[keys.FORCES].shape == (5, 3)


def test_pbcs():
    # structure 1: 1 atom in a (1x1x1) cell
    # structure 2: 1 atom in a (2x2x2) cell
    structures = [
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(1, 1, 1)),
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(2, 2, 2)),
    ]
    graphs = convert_to_atomic_graphs(structures, cutoff=1.2)

    assert number_of_edges(graphs[0]) == 6
    assert number_of_edges(graphs[1]) == 0

    batch = batch_graphs(graphs)
    assert number_of_edges(batch) == 6
    assert batch[keys.CELL].shape == (2, 3, 3)

    assert neighbour_vectors(batch).shape == (6, 3)
    assert neighbour_distances(batch).tolist() == [1.0] * 6


def test_sum_per_structure():
    structures = [s.copy() for s in STRUCTURES]
    structures[0].arrays[keys.LOCAL_ENERGIES] = [0, 1]
    structures[1].arrays[keys.LOCAL_ENERGIES] = [2, 3, 4]
    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)

    # sum per structure should work for:

    # 1. a single structure
    x = graphs[0][keys.LOCAL_ENERGIES]
    assert sum_per_structure(x, graphs[0]) == x.sum()

    # 2. a batch of structures
    batch = batch_graphs(graphs)
    x = batch[keys.LOCAL_ENERGIES]
    assert sum_per_structure(x, batch).tolist() == [1, 9]


def test_data_loader():
    loader = AtomicDataLoader(GRAPHS, batch_size=2)

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
        AtomicDataLoader(GRAPHS, batch_size=2, collate_fn=lambda x: x)
