from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from load_atoms import load_dataset

from graph_pes.data.dataset import ASEDataset, FittingData
from graph_pes.graphs import keys


def load_atoms_datasets(
    id: str | Path,
    cutoff: float,
    n_train: int,
    n_valid: int,
    split: Literal["random", "sequential"] = "random",
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[keys.LabelKey, str] | None = None,
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
    n_valid:
        The number of validation structures.
    split:
        The split method. ``"random"`` shuffles the structures before
        choosing a non-overlapping split, while ``"sequential"`` takes the
        first ``n_train`` structures for training and the next ``n_valid``
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
    ...     n_valid=100,
    ...     property_map={"energy": "U0"},
    ... )
    """
    structures = list(load_dataset(id))

    if split == "random":
        idxs = np.random.default_rng(seed).permutation(len(structures))
        structures = [structures[i] for i in idxs]

    train_structures = structures[:n_train]
    val_structures = structures[n_train : n_train + n_valid]

    return FittingData(
        ASEDataset(train_structures, cutoff, pre_transform, property_map),
        ASEDataset(val_structures, cutoff, pre_transform, property_map),
    )
