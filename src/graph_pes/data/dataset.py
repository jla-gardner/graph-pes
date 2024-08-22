from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, TypeVar

import ase
import torch.utils.data
from locache import persist

from graph_pes.graphs import LabelledGraph, keys
from graph_pes.graphs.keys import ALL_LABEL_KEYS, LabelKey
from graph_pes.graphs.operations import available_labels
from graph_pes.logger import logger
from graph_pes.util import uniform_repr

from .io import to_atomic_graph, to_atomic_graphs

T = TypeVar("T", covariant=True)


class SizedDataset(Protocol[T]):
    def __getitem__(self, index: int) -> T: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...


class LabelledGraphDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for datasets of
    :class:`~graph_pes.graphs.LabelledGraph` instances.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> LabelledGraph: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def prepare_data(self):
        """
        Prepare the data for the dataset.

        Called on rank-0 only: don't set any state here.
        May be called multiple times.
        """

    def setup(self):
        """
        Set up the data for the dataset.

        Called on every process in the distributed setup: set state here.
        """

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
    A dataset that wraps a sequence of :class:`~graph_pes.graphs.LabelledGraph`
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


class ASEDataset(LabelledGraphDataset):
    """
    A dataset that wraps an ASE dataset.

    Parameters
    ----------
    structures
        The ASE dataset to wrap.
    cutoff
        The cutoff to use when creating neighbour indexes for the graphs.
    pre_transform
        Whether to precompute the neighbour indexes for the graphs.
    property_mapping
        A mapping from properties expected in ``graph-pes`` to their names
        in the dataset.
    """

    def __init__(
        self,
        structures: SizedDataset[ase.Atoms] | Sequence[ase.Atoms],
        cutoff: float,
        pre_transform: bool = False,
        property_mapping: dict[keys.LabelKey, str] | None = None,
    ):
        self.structures = structures
        self.cutoff = cutoff

        self.pre_transform = pre_transform
        self.property_mapping = property_mapping
        self.graphs = None

        # raise errors on instantiation if accessing a datapoint would fail
        _ = self[0]

    def prepare_data(self):
        # bit of a hack: pre_transformed_structures is cached to disk
        # for each unique combination of structures and cutoff: calling this
        # on rank-0 before any other rank ensures a cache hit for all ranks
        # in the distributed setup
        self.setup()

    def setup(self):
        if self.pre_transform:
            self.graphs = pre_transform_structures(
                self.structures,
                cutoff=self.cutoff,
                property_mapping=self.property_mapping,
            )

    def __getitem__(self, index: int) -> LabelledGraph:
        if self.graphs is not None:
            return self.graphs[index]
        return to_atomic_graph(
            self.structures[index],
            cutoff=self.cutoff,
            property_mapping=self.property_mapping,
        )

    def __len__(self) -> int:
        return len(self.structures)

    def __repr__(self) -> str:
        labels = available_labels(self[0])
        return f"ASEDataset({len(self):,}, labels={labels})"


@dataclass
class FittingData:
    """A convenience container for training and validation datasets."""

    train: LabelledGraphDataset
    """The training dataset."""
    valid: LabelledGraphDataset
    """The validation dataset."""

    def __repr__(self) -> str:
        return uniform_repr(
            self.__class__.__name__,
            train=self.train,
            valid=self.valid,
        )


@persist
def pre_transform_structures(
    structures: SizedDataset[ase.Atoms],
    cutoff: float = 5.0,
    property_mapping: dict[keys.LabelKey, str] | None = None,
) -> list[LabelledGraph]:
    logger.info(
        f"Caching neighbour lists for {len(structures)} structures "
        f"with cutoff {cutoff}"
    )
    return to_atomic_graphs(structures, cutoff, property_mapping)
