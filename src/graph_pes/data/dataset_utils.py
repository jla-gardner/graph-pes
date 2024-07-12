from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from load_atoms import load_dataset

from graph_pes.data.dataset import ASEDataset, FittingData


def load_atoms_datasets(
    id: str | Path,
    cutoff: float,
    n_train: int,
    n_val: int,
    split: Literal["random", "sequential"] = "random",
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, str] | None = None,
) -> FittingData:
    """
    Load an ``ASE``/``load-atoms`` dataset and split into train and valid sets.

    Parameters
    ----------
    id:
        The dataset identifier. Can be a ``load-atoms`` id, or a path to an
        ``ase``-readable data file.
    cutoff:
        The cutoff radius for the neighbor list.
    n_train:
        The number of training structures.
    n_val:
        The number of validation structures.
    split:
        The split method. ``"random"`` shuffles the structures before
        choosing a non-overlapping split, while ``"sequential"`` takes the
        first ``n_train`` structures for training and the next ``n_val``
        structures for validation.
    seed:
        The random seed.
    pre_transform:
        Whether to pre-calculate the neighbour lists for each structure.
    root:
        The root directory
    property_map:
        A mapping from properties expected in ``graph-pes`` to their names
        in the dataset.

    Returns
    -------
    FittingData
        A tuple of training and validation datasets.

    Examples
    --------
    Load a subset of the QM9 dataset. Ensure that the ``U0`` property is
    mapped to ``energy``:

    >>> load_atoms_datasets(
    ...     "QM9",
    ...     cutoff=5.0,
    ...     n_train=1_000,
    ...     n_val=100,
    ...     property_map={"energy": "U0"},
    ... )
    """
    structures = list(load_dataset(id))

    if split == "random":
        idxs = np.random.default_rng(seed).permutation(len(structures))
        structures = [structures[i] for i in idxs]

    if property_map:
        for i in range(n_train + n_val):
            structure = structures[i]
            for key, value in property_map.items():
                structure.info[key] = structure.info[value]

    train_structures = structures[:n_train]
    val_structures = structures[n_train : n_train + n_val]

    return FittingData(
        ASEDataset(train_structures, cutoff, pre_transform),
        ASEDataset(val_structures, cutoff, pre_transform),
    )
