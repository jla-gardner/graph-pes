from typing import Literal

import helpers
import numpy as np
import pytest
from graph_pes.data import load_atoms_datasets


@pytest.mark.parametrize("split", ["random", "sequential"])
def test_shuffling(split: Literal["random", "sequential"]):
    dataset = load_atoms_datasets(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
        split=split,
    )

    if split == "sequential":
        np.testing.assert_allclose(
            dataset.train[0]["_positions"],
            helpers.CU_TEST_STRUCTURES[0].positions,
        )
    else:
        assert not np.allclose(
            dataset.train[0]["_positions"],
            helpers.CU_TEST_STRUCTURES[0].positions,
        )


def test_dataset():
    dataset = load_atoms_datasets(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
    )

    assert len(dataset.train) == 8
    assert len(dataset.valid) == 2


def test_property_map():
    dataset = load_atoms_datasets(
        id=helpers.CU_STRUCTURES_FILE,
        cutoff=3.7,
        n_train=8,
        n_valid=2,
        property_map={"forces": "positions"},
    )

    assert "forces" in dataset.train[0]
    np.testing.assert_allclose(
        dataset.train[0]["forces"],
        helpers.CU_TEST_STRUCTURES[0].positions,
    )
