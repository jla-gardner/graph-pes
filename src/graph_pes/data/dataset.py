from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol, Sequence, TypeVar

import ase
import torch.utils.data

from graph_pes.graphs import LabelledGraph
from graph_pes.graphs.keys import ALL_LABEL_KEYS, LabelKey
from graph_pes.logger import logger

from .io import to_atomic_graph

T = TypeVar("T", covariant=True)


class SizedDataset(Protocol[T]):
    def __getitem__(self, index: int) -> T: ...
    def __len__(self) -> int: ...


class LabelledGraphDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for datasets of :class:`~graph_pes
    .data.LabelledGraph` instances.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> LabelledGraph: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def shuffled(self, seed: int = 42) -> ShuffledDataset:
        """
        Return a shuffled version of this dataset.

        Parameters
        ----------
        seed
            The random seed to use for shuffling.

        Returns
        -------
        ShuffledDataset
            A shuffled version of this dataset.
        """
        return ShuffledDataset(self, seed)

    def up_to_n(self, n: int) -> LabelledGraphDataset:
        """
        Return a dataset with only the first `n` elements.

        Parameters
        ----------
        n
            The maximum number of elements to keep.

        Returns
        -------
        LabelledGraphDataset
            A dataset with only the first `n` elements.
        """
        if n > len(self):
            return self
        return ReMappedDataset(self, range(n))

    def sample(self, n: int, seed: int = 42) -> LabelledGraphDataset:
        """
        Randomly sample the dataset to get ``n`` elements without replacement.

        Parameters
        ----------
        n
            The number of elements to sample.
        seed
            The random seed to use for sampling.
        """
        return self.shuffled(seed).up_to_n(n)

    @property
    def labels(self) -> list[LabelKey]:
        """The labels that are available to train on with this dataset"""

        # assume all data points have the same labels
        # TODO: is there a nice way to enforce this?
        datapoint = self[0]
        return [key for key in datapoint if key in ALL_LABEL_KEYS]

    def __iter__(self) -> Iterator[LabelledGraph]:
        for i in range(len(self)):
            yield self[i]


class ReMappedDataset(LabelledGraphDataset):
    """
    A dataset where the indices have been remapped.

    Parameters
    ----------
    dataset
        The dataset to remap.
    indices
        The remapped indices to use.
    """

    def __init__(self, dataset: LabelledGraphDataset, indices: Sequence[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> LabelledGraph:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


class ShuffledDataset(ReMappedDataset):
    """
    A dataset that shuffles the order of the underlying dataset.

    Parameters
    ----------
    dataset
        The dataset to shuffle.
    seed
        The random seed to use for shuffling.
    """

    def __init__(self, dataset: LabelledGraphDataset, seed: int):
        indices: list[int] = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        super().__init__(dataset, indices)


class SequenceDataset(LabelledGraphDataset):
    """
    A dataset that wraps a sequence of :class:`~graph_pes.data.LabelledGraph`
    instances.

    Parameters
    ----------
    graphs
        The graphs to wrap.
    """

    def __init__(self, graphs: Sequence[LabelledGraph]):
        self.graphs = graphs

    def __getitem__(self, index: int) -> LabelledGraph:
        return self.graphs[index]

    def __len__(self) -> int:
        return len(self.graphs)


# TODO: differentiate between structure and graph
class AseDataset(LabelledGraphDataset):
    """
    A dataset that wraps an ASE dataset.

    Parameters
    ----------
    ase_dataset
        The ASE dataset to wrap.
    cutoff
        The cutoff to use when creating neighbour indexes for the graphs.
    pre_transform
        Whether to precompute the neighbour indexes for the graphs.
    """

    def __init__(
        self,
        ase_dataset: SizedDataset[ase.Atoms] | Sequence[ase.Atoms],
        cutoff: float = 5.0,
        pre_transform: bool = False,
    ):
        self.ase_dataset = ase_dataset
        self.cutoff = cutoff

        self.pre_transform = pre_transform
        if pre_transform:
            logger.info("Pre-transforming ASE dataset to graphs...")
            self.graphs = [
                to_atomic_graph(atoms, cutoff=cutoff) for atoms in ase_dataset
            ]

        else:
            self.grahsp = None

    def __getitem__(self, index: int) -> LabelledGraph:
        if self.pre_transform:
            return self.graphs[index]
        return to_atomic_graph(self.ase_dataset[index], cutoff=self.cutoff)

    def __len__(self) -> int:
        return len(self.ase_dataset)