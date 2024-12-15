from __future__ import annotations

import pathlib
from typing import Sequence, overload

import ase
import ase.db
import numpy as np

from graph_pes.utils.misc import slice_to_range


class ASEDatabase(Sequence[ase.Atoms]):
    """
    A class that wraps an ASE database file, allowing for indexing into the
    database to obtain :class:`ase.Atoms` objects.

    We assume that each row contains labels in the ``data`` attribute,
    as a mapping from property names to values, and that units are "standard"
    ASE units, e.g. ``eV``, ``eV/Å``, etc.

    Fully compatible with `SchNetPack Dataset Files <https://schnetpack.readthedocs.io/en/latest/tutorials/tutorial_01_preparing_data.html>`__.

    See the `ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`__
    for more details about this file format.

    .. warning::

        This dataset indexes into a database, performing many random access
        reads from disk. This can be very slow! If you are using a distributed
        compute cluster, ensure you copy your database file to somewhere with
        fast local storage (as opposed to network-attached storage).

        Similarly, consider using several workers when loading the dataset,
        e.g. ``fitting/loader_kwargs/num_workers=8``.

    Parameters
    ----------
    path: str | pathlib.Path
        The path to the database.
    """  # noqa: E501

    def __init__(self, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database file {path} does not exist")
        self.path = path
        self.db = ase.db.connect(path, use_lock_file=False)

    @overload
    def __getitem__(self, index: int) -> ase.Atoms: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[ase.Atoms]: ...
    def __getitem__(
        self, index: int | slice
    ) -> ase.Atoms | Sequence[ase.Atoms]:
        if isinstance(index, slice):
            indices = slice_to_range(index, len(self))
            return [self[i] for i in indices]

        atoms = self.db.get_atoms(index + 1, add_additional_information=True)
        data = atoms.info.pop("data", {})
        arrays = {
            k: v
            for k, v in data.items()
            if isinstance(v, np.ndarray) and v.shape[0] == len(atoms)
        }
        info = {k: v for k, v in data.items() if k not in arrays}
        atoms.arrays.update(arrays)
        atoms.info.update(info)
        return atoms

    def __len__(self) -> int:
        return self.db.count()
