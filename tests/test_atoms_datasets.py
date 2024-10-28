from typing import Literal

import helpers
import numpy as np
import pytest
from graph_pes.atomic_graph import number_of_atoms
from graph_pes.data import load_atoms_dataset


@pytest.mark.parametrize("split", ["random", "sequential"])
def test_shuffling(split: Literal["random", "sequential"]):
    dataset = load_atoms_dataset(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
        split=split,
    )

    if split == "sequential":
        np.testing.assert_allclose(
            dataset.train[0].R,
            helpers.CU_TEST_STRUCTURES[0].positions,
        )
    else:
        # different structures with different sizes in the first
        # position of the training set after shuffling
        assert number_of_atoms(dataset.train[0]) != len(
            helpers.CU_TEST_STRUCTURES[0]
        )


def test_dataset():
    dataset = load_atoms_dataset(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
    )

    assert len(dataset.train) == 8
    assert len(dataset.valid) == 2


def test_property_map():
    dataset = load_atoms_dataset(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
        property_map={"positions": "forces"},
        split="sequential",
    )

    assert "forces" in dataset.train[0]
    np.testing.assert_allclose(
        dataset.train[0].properties["forces"],
        helpers.CU_TEST_STRUCTURES[0].positions,
    )

    with pytest.raises(KeyError, match="Property UNKNOWN KEY not found for"):
        load_atoms_dataset(
            id=helpers.CU_STRUCTURES_FILE,
            cutoff=3.7,
            n_train=8,
            n_valid=2,
            property_map={"UNKNOWN Key": "energy"},
        )
