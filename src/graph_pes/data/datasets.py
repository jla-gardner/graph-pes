from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal, Mapping, Protocol, Sequence, TypeVar

import ase
import ase.io
import numpy as np
import torch.utils.data
from load_atoms import load_dataset
from locache import persist

from graph_pes.atomic_graph import (
    ALL_PROPERTY_KEYS,
    AtomicGraph,
    PropertyKey,
    available_properties,
)
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import uniform_repr

T = TypeVar("T", covariant=True)


class SizedDataset(Protocol[T]):
    """
    A protocol for datasets that can be indexed into and have a length.
    """

    def __getitem__(self, index: int) -> T:
        """Index into the dataset."""
        ...

    def __len__(self) -> int:
        """The number of elements in the dataset."""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over the dataset."""
        ...


class GraphDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for datasets of
    :class:`~graph_pes.AtomicGraph` instances.

    All subclasses are fully compatible with :class:`torch.utils.data.Dataset`,
    and with distributed training protocols providing that the
    :meth:`~graph_pes.data.GraphDataset.prepare_data` and
    :meth:`~graph_pes.data.GraphDataset.setup` methods are implemented
    correctly.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> AtomicGraph:
        """
        Indexing into the dataset should return an
        :class:`~graph_pes.AtomicGraph`.
        """

    @abstractmethod
    def __len__(self) -> int:
        """The number of graphs in the dataset."""

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

    def up_to_n(self, n: int) -> GraphDataset:
        """
        Return a dataset with only the first `n` elements.

        Parameters
        ----------
        n
            The maximum number of elements to keep.

        Returns
        -------
        GraphDataset
            A dataset with only the first `n` elements.
        """
        if n > len(self):
            return self
        return ReMappedDataset(self, range(n))

    def sample(self, n: int, seed: int = 42) -> GraphDataset:
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
    def properties(self) -> list[PropertyKey]:
        """The properties that are available to train on with this dataset"""

        # assume all data points have the same labels
        example_graph = self[0]
        return [
            key for key in ALL_PROPERTY_KEYS if key in example_graph.properties
        ]

    def __iter__(self) -> Iterator[AtomicGraph]:
        """Iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]


class ReMappedDataset(GraphDataset):
    """
    A dataset where the indices have been remapped.

    Parameters
    ----------
    dataset: GraphDataset
        The dataset to remap.
    indices: Sequence[int]
        The remapped indices to use.
    """

    def __init__(self, dataset: GraphDataset, indices: Sequence[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> AtomicGraph:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


class ShuffledDataset(ReMappedDataset):
    """
    A dataset that wraps an existing dataset, and mimics a shuffled
    version of it.

    Parameters
    ----------
    dataset: GraphDataset
        The dataset to shuffle.
    seed: int
        The random seed to use for shuffling.
    """

    def __init__(self, dataset: GraphDataset, seed: int):
        indices: list[int] = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        super().__init__(dataset, indices)


class SequenceDataset(GraphDataset):
    """
    A dataset that wraps a sequence of :class:`~graph_pes.AtomicGraph`
    instances.

    Parameters
    ----------
    graphs: Sequence[AtomicGraph]
        The graphs to wrap.
    """

    def __init__(self, graphs: Sequence[AtomicGraph]):
        self.graphs = graphs

    def __getitem__(self, index: int) -> AtomicGraph:
        return self.graphs[index]

    def __len__(self) -> int:
        return len(self.graphs)


class ASEDataset(GraphDataset):
    """
    A dataset that wraps a dataset of :class:`ase.Atoms` objects.

    We make no assumptions as to the format of the underlying dataset,
    so long as we can index into it, and know its length. This means
    that we can wrap:

    * a list of :class:`ase.Atoms` objects
    * an ``lmdb``-backed dataset (as from e.g. `load-atoms <https://jla-gardner.github.io/load-atoms/>`__)
    * any other collection of :class:`ase.Atoms` objects that we can index into

    Parameters
    ----------
    structures: SizedDataset[ase.Atoms] | typing.Sequence[ase.Atoms]
        The ASE dataset to wrap.
    cutoff: float
        The cutoff to use when creating neighbour indexes for the graphs.
    pre_transform: bool
        Whether to precompute the the :class:`~graph_pes.AtomicGraph`
        objects, or only do so on-the-fly when the dataset is accessed.
        This pre-computations stores the graphs in memory, and so will be
        prohibitively expensive for large datasets.
    property_mapping: Mapping[str, PropertyKey] | None
        A mapping from properties defined on the :class:`ase.Atoms` objects to
        their appropriate names in ``graph-pes``, see
        :meth:`~graph_pes.AtomicGraph.from_ase`.
    """

    def __init__(
        self,
        structures: SizedDataset[ase.Atoms] | Sequence[ase.Atoms],
        cutoff: float,
        pre_transform: bool = False,
        property_mapping: Mapping[str, PropertyKey] | None = None,
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

    def __getitem__(self, index: int) -> AtomicGraph:
        if self.graphs is not None:
            return self.graphs[index]
        return AtomicGraph.from_ase(
            self.structures[index],
            cutoff=self.cutoff,
            property_mapping=self.property_mapping,
        )

    def __len__(self) -> int:
        return len(self.structures)

    def __repr__(self) -> str:
        labels = available_properties(self[0])
        return f"ASEDataset({len(self):,}, properties={labels})"


@dataclass
class FittingData:
    """A convenience container for training and validation datasets."""

    train: GraphDataset
    """The training dataset."""
    valid: GraphDataset
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
    property_mapping: Mapping[str, PropertyKey] | None = None,
) -> list[AtomicGraph]:
    logger.info(
        f"Caching neighbour lists for {len(structures)} structures "
        f"with cutoff {cutoff}"
    )
    return [
        AtomicGraph.from_ase(s, cutoff, property_mapping) for s in structures
    ]


def load_atoms_dataset(
    id: str | pathlib.Path,
    cutoff: float,
    n_train: int,
    n_valid: int = -1,
    split: Literal["random", "sequential"] = "random",
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, PropertyKey] | None = None,
) -> FittingData:
    """
    Load an dataset of :class:`ase.Atoms` objects using
    `load-atoms <https://jla-gardner.github.io/load-atoms/>`__,
    convert them to :class:`~graph_pes.AtomicGraph` instances, and split into
    train and valid sets.

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
        The number of validation structures. If ``-1``, the number of validation
        structures is set to the number of remaining structures after
        training structures are chosen.
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
        A mapping from properties as named on the atoms objects to
        ``graph-pes`` property keys, e.g. ``{"U0": "energy"}``.

    Returns
    -------
    FittingData
        A tuple of training and validation datasets.

    Examples
    --------
    Load a subset of the QM9 dataset. Ensure that the ``U0`` property is
    mapped to ``energy``:

    >>> load_atoms_dataset(
    ...     "QM9",
    ...     cutoff=5.0,
    ...     n_train=1_000,
    ...     n_valid=100,
    ...     property_map={"U0": "energy"},
    ... )
    """
    structures = list(load_dataset(id))

    if split == "random":
        idxs = np.random.default_rng(seed).permutation(len(structures))
        structures = [structures[i] for i in idxs]

    if n_valid == -1:
        n_valid = len(structures) - n_train

    train_structures = structures[:n_train]
    val_structures = structures[n_train : n_train + n_valid]

    return FittingData(
        ASEDataset(train_structures, cutoff, pre_transform, property_map),
        ASEDataset(val_structures, cutoff, pre_transform, property_map),
    )


def file_dataset(
    path: str | pathlib.Path,
    cutoff: float,
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, PropertyKey] | None = None,
) -> ASEDataset:
    """
    Load an ASE dataset from a file.

    Parameters
    ----------
    path:
        The path to the file.
    cutoff:
        The cutoff radius for the neighbour list.
    n:
        The number of structures to load. If ``None``,
        all structures are loaded.
    shuffle:
        Whether to shuffle the structures.
    seed:
        The random seed used for shuffling.
    pre_transform:
        Whether to pre-calculate the neighbour lists for each structure.
    property_map:
        A mapping from properties as named on the atoms objects to
        ``graph-pes`` property keys, e.g. ``{"U0": "energy"}``.

    Returns
    -------
    ASEDataset
        The ASE dataset.

    Example
    -------
    Load a dataset from a file, ensuring that the ``energy`` property is
    mapped to ``U0``:

    >>> file_dataset(
    ...     "training_data.xyz",
    ...     cutoff=5.0,
    ...     property_map={"U0": "energy"},
    ... )
    """

    structures = ase.io.read(path, index=":")
    assert isinstance(structures, list)

    if shuffle:
        idxs = np.random.default_rng(seed).permutation(len(structures))
        structures = [structures[i] for i in idxs]

    if n is not None:
        structures = structures[:n]

    return ASEDataset(structures, cutoff, pre_transform, property_map)
