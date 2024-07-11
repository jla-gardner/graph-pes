from __future__ import annotations

from pathlib import Path
from typing import Literal

from load_atoms import load_dataset, utils
from locache import persist

from graph_pes.data.dataset import ASEDataset, FittingData


@persist
def load_atoms_datasets(
    id: str | Path,
    cutoff: float,
    n_train: int,
    n_val: int,
    seed: int = 42,
    split: Literal["random", "sequential"] = "random",
    pre_transform: bool = True,
    root: str | Path | None = None,
) -> FittingData:
    """TODO"""
    dataset = list(load_dataset(id, root=root))

    if split == "random":
        train_structures, val_structures = utils.random_split(
            dataset, [n_train, n_val], seed
        )

    elif split == "sequential":
        train_structures = dataset[:n_train]
        val_structures = dataset[n_train : n_train + n_val]

    return FittingData(
        ASEDataset(train_structures, cutoff, pre_transform),
        ASEDataset(val_structures, cutoff, pre_transform),
    )
