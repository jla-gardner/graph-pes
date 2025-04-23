import pathlib

import ase.io
import pytest
from ase.build import molecule

from graph_pes.config.shared import (
    parse_dataset_collection,
    parse_single_dataset,
)
from graph_pes.data.datasets import (
    DatasetCollection,
    GraphDataset,
    file_dataset,
)
from graph_pes.models.pairwise import LennardJones


def test_parse_single_dataset(tmp_path: pathlib.Path):
    CUTOFF = 5.0
    model = LennardJones(cutoff=CUTOFF)
    ase.io.write(tmp_path / "test.xyz", [molecule("H2O"), molecule("CH4")])

    # test that a path to a single file works
    config = str(tmp_path / "test.xyz")
    dataset = parse_single_dataset(config, model)
    assert isinstance(dataset, GraphDataset)
    assert len(dataset) == 2
    assert dataset[0].cutoff == CUTOFF

    # test that a dict works
    config = {"path": str(tmp_path / "test.xyz"), "n": 1}
    dataset = parse_single_dataset(config, model)
    assert isinstance(dataset, GraphDataset)
    assert len(dataset) == 1
    assert dataset[0].cutoff == CUTOFF

    # test that overriding the cutoff works
    config = {"path": str(tmp_path / "test.xyz"), "cutoff": 6.0}
    dataset = parse_single_dataset(config, model)
    assert isinstance(dataset, GraphDataset)
    assert len(dataset) == 2
    assert dataset[0].cutoff == 6.0

    # test that an instance of a GraphDataset works
    raw_dataset = file_dataset(str(tmp_path / "test.xyz"), cutoff=CUTOFF)
    dataset = parse_single_dataset(raw_dataset, model)
    assert len(dataset) == 2
    assert dataset[0].cutoff == CUTOFF

    # test that a non-GraphDataset raises an error
    with pytest.raises(ValueError):
        parse_single_dataset(1, model)

    # and that a bogus file path raises file not found
    with pytest.raises(FileNotFoundError):
        parse_single_dataset("bogus/path.xyz", model)


def test_parse_dataset_collection(tmp_path: pathlib.Path):
    CUTOFF = 5.0
    model = LennardJones(cutoff=CUTOFF)
    ase.io.write(tmp_path / "test.xyz", [molecule("H2O"), molecule("CH4")])

    # test that a dict works
    config = {
        "train": str(tmp_path / "test.xyz"),
        "valid": str(tmp_path / "test.xyz"),
    }
    collection = parse_dataset_collection(config, model)
    assert isinstance(collection, DatasetCollection)
    assert len(collection.train) == 2
    assert len(collection.valid) == 2
    assert collection.test is None

    # tests should be picked up
    config = {
        "train": str(tmp_path / "test.xyz"),
        "valid": str(tmp_path / "test.xyz"),
        "test": str(tmp_path / "test.xyz"),
    }
    collection = parse_dataset_collection(config, model)
    assert isinstance(collection, DatasetCollection)
    assert len(collection.train) == 2
    assert len(collection.valid) == 2
    assert collection.test is not None
    assert isinstance(collection.test, GraphDataset)

    config = {
        "train": str(tmp_path / "test.xyz"),
        "valid": str(tmp_path / "test.xyz"),
        "test": {
            "bulk": str(tmp_path / "test.xyz"),
            "surface": {
                "path": str(tmp_path / "test.xyz"),
                "n": 1,
            },
        },
    }
    collection = parse_dataset_collection(config, model)
    assert isinstance(collection, DatasetCollection)
    assert len(collection.train) == 2
    assert len(collection.valid) == 2
    assert collection.test is not None
    assert isinstance(collection.test, dict)
    assert len(collection.test) == 2
    assert isinstance(collection.test["bulk"], GraphDataset)
    assert isinstance(collection.test["surface"], GraphDataset)
    assert len(collection.test["surface"]) == 1

    # test that a DatasetCollection instance is returned unchanged
    raw_collection = collection
    new_collection = parse_dataset_collection(raw_collection, model)
    assert new_collection == collection

    # no train or valid keys should raise an error
    with pytest.raises(ValueError):
        parse_dataset_collection({"test": str(tmp_path / "test.xyz")}, model)

    # file not found for test set should raise an error
    with pytest.raises(FileNotFoundError):
        parse_dataset_collection(
            {
                "train": str(tmp_path / "test.xyz"),
                "valid": str(tmp_path / "test.xyz"),
                "test": "bogus/path.xyz",
            },
            model,
        )

    # not a valid dict
    with pytest.raises(ValueError):
        parse_dataset_collection(
            {
                "train": str(tmp_path / "test.xyz"),
                "valid": str(tmp_path / "test.xyz"),
                "test": {"bulk": 1},
            },
            model,
        )
