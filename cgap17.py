import ase
from graph_pes.data.dataset import AseDataset
from graph_pes.data.module import FixedDatasets
from graph_pes.data.utils import random_split
from load_atoms import load_dataset


def load_data(batch_size: int = 32, n_train: int = 1_000) -> FixedDatasets:
    dataset: list[ase.Atoms] = load_dataset("C-GAP-17").filter_by(
        lambda x: len(x) > 2
    )  # type: ignore
    train, val, test = random_split(dataset, [n_train, 10, 10], seed=42)

    return FixedDatasets(
        AseDataset(train, cutoff=3.7, pre_transform=True),
        AseDataset(val, cutoff=3.7, pre_transform=True),
        {"test": AseDataset(test, cutoff=3.7, pre_transform=True)},
        batch_size=batch_size,
        num_workers=4,
    )
