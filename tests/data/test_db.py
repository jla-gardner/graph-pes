from pathlib import Path
from typing import Sequence

import ase
import numpy as np
import pytest

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.data.ase_db import ASE_Database
from graph_pes.data.datasets import file_dataset

DB_FILE = Path(__file__).parent / "schnetpack_data.db"
# the dataset available at ./schnetpack_data.db
# was created with the following code:
#
#     import numpy as np
#     from schnetpack.data import ASEAtomsData
#     from ase.build import molecule
#
#
#     structures = [molecule(s) for s in "H2O CO2 CH4 C2H4 C2H2".split()]
#     properties = [
#         {
#             "energy": np.random.rand(),
#             "forces": np.random.rand(len(s), 3),
#             "other_key": np.random.rand(),
#         }
#         for s in structures
#     ]
#
#     new_dataset = ASEAtomsData.create(
#         "schnetpack_data.db",
#         distance_unit="Ang",
#         property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
#     )
#     new_dataset.add_systems(properties, structures)


def test_ASE_Database():
    db = ASE_Database(DB_FILE)
    assert len(db) == 5
    assert isinstance(db[0].info["energy"], float)
    assert isinstance(db[0].arrays["forces"], np.ndarray) and db[0].arrays[
        "forces"
    ].shape == (3, 3)

    assert isinstance(db[0:2], Sequence)
    assert isinstance(db[0:2][0], ase.Atoms)

    with pytest.raises(IndexError):
        db[10]

    with pytest.raises(FileNotFoundError):
        ASE_Database("non_existent_file.db")


def test_file_dataset_with_db():
    from_file = file_dataset(DB_FILE, cutoff=2.5, n=3, shuffle=False, seed=42)
    assert len(from_file) == 3
    assert isinstance(from_file[0], AtomicGraph)
