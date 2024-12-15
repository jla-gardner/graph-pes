from __future__ import annotations

import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, overload

import ase
import ase.db
import ase.io
import locache
import torch.utils.data
from load_atoms import load_dataset

from graph_pes.atomic_graph import (
    ALL_PROPERTY_KEYS,
    AtomicGraph,
    PropertyKey,
)
from graph_pes.data.ase_db import ASE_Database
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import slice_to_range, uniform_repr
from graph_pes.utils.sampling import SequenceSampler


class GraphDataset(torch.utils.data.Dataset, ABC):
    """
    A dataset of :class:`~graph_pes.AtomicGraph` instances.

    Parameters
    ----------
    graphs
        The collection of :class:`~graph_pes.AtomicGraph` instances.
    """

    def __init__(self, graphs: Sequence[AtomicGraph]):
        self.graphs = graphs
        # raise errors on instantiation if accessing a datapoint would fail
        _ = self[0]

    def __getitem__(self, index: int) -> AtomicGraph:
        return self.graphs[index]

    def __len__(self) -> int:
        return len(self.graphs)

    def prepare_data(self) -> None:
        """
        Make general preparations for loading the data for the dataset.

        Called on rank-0 only: don't set any state here.
        May be called multiple times.
        """

    def setup(self) -> GraphDataset | None:
        """
        Set-up the data for this specific instance of the dataset.

        Called on every process in the distributed setup:

        * if you want to set state directly, do it here and return ``None``
        * if you want to return a new dataset, return that instead
        """

    @property
    def properties(self) -> list[PropertyKey]:
        """The properties that are available to train on with this dataset"""

        # assume all data points have the same labels
        example_graph = self[0]
        return [
            key for key in ALL_PROPERTY_KEYS if key in example_graph.properties
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self):,}, "
            f"properties={self.properties})"
        )


class ASEToGraphsConverter(Sequence[AtomicGraph]):
    def __init__(
        self,
        structures: Sequence[ase.Atoms],
        cutoff: float,
        property_mapping: Mapping[str, PropertyKey] | None = None,
    ):
        self.structures = structures
        self.cutoff = cutoff
        self.property_mapping = property_mapping

    @overload
    def __getitem__(self, index: int) -> AtomicGraph: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[AtomicGraph]: ...
    def __getitem__(
        self, index: int | slice
    ) -> AtomicGraph | Sequence[AtomicGraph]:
        if isinstance(index, slice):
            indices = slice_to_range(index, len(self))
            return [self[i] for i in indices]

        return AtomicGraph.from_ase(
            self.structures[index],
            cutoff=self.cutoff,
            property_mapping=self.property_mapping,
        )

    def __len__(self) -> int:
        return len(self.structures)


# use the locache library to cache the graphs that result from this
# transform to disk: this means that multiple training runs on the
# same dataset will be able to reuse the same graphs, massively speeding
# up the start to training for the (n>1)th run
@locache.persist
def get_all_graphs_and_cache_to_disk(
    converter: ASEToGraphsConverter,
) -> list[AtomicGraph]:
    logger.info(
        f"Caching neighbour lists for {len(converter)} structures "
        f"with cutoff {converter.cutoff} and property mapping "
        f"{converter.property_mapping}"
    )
    return [converter[i] for i in range(len(converter))]


class ASEToGraphDataset(GraphDataset):
    """
    A dataset that wraps a :class:`Sequence` of :class:`ase.Atoms`, and converts
    them to :class:`~graph_pes.AtomicGraph` instances.

    Parameters
    ----------
    structures
        The collection of :class:`ase.Atoms` objects to convert to
        :class:`~graph_pes.AtomicGraph` instances.
    cutoff
        The cutoff to use when creating neighbour indexes for the graphs.
    pre_transform
        Whether to precompute the the :class:`~graph_pes.AtomicGraph`
        objects, or only do so on-the-fly when the dataset is accessed.
        This pre-computations stores the graphs in memory, and so will be
        prohibitively expensive for large datasets.
    property_mapping
        A mapping from properties defined on the :class:`ase.Atoms` objects to
        their appropriate names in ``graph-pes``, see
        :meth:`~graph_pes.AtomicGraph.from_ase`.
    """

    def __init__(
        self,
        structures: Sequence[ase.Atoms],
        cutoff: float,
        pre_transform: bool = False,
        property_mapping: Mapping[str, PropertyKey] | None = None,
    ):
        super().__init__(
            ASEToGraphsConverter(structures, cutoff, property_mapping),
        )
        self.pre_transform = pre_transform

    def prepare_data(self):
        if self.pre_transform:
            # cache the graphs to disk - this is done on rank-0 only
            # and means that expensive data pre-transforms don't need to be
            # recomputed on each rank in the distributed setup
            get_all_graphs_and_cache_to_disk(self.graphs)

    def setup(self) -> GraphDataset | None:
        if self.pre_transform:
            # load the graphs from disk
            actual_graphs = get_all_graphs_and_cache_to_disk(self.graphs)
            return GraphDataset(actual_graphs)


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
    structures = SequenceSampler(load_dataset(id))

    if split == "random":
        structures = structures.shuffled(seed)

    if n_valid == -1:
        n_valid = len(structures) - n_train

    train = structures[:n_train]
    val = structures[n_train : n_train + n_valid]

    return FittingData(
        ASEToGraphDataset(train, cutoff, pre_transform, property_map),
        ASEToGraphDataset(val, cutoff, pre_transform, property_map),
    )


def file_dataset(
    path: str | pathlib.Path,
    cutoff: float,
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, PropertyKey] | None = None,
) -> ASEToGraphDataset:
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
    ASEToGraphDataset
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

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix == ".db":
        structures = ASE_Database(path)
    else:
        structures = ase.io.read(path, index=":")
        assert isinstance(structures, list)

    structure_collection = SequenceSampler(structures)
    if shuffle:
        structure_collection = structure_collection.shuffled(seed)

    if n is not None:
        structure_collection = structure_collection[:n]

    return ASEToGraphDataset(
        structure_collection, cutoff, pre_transform, property_map
    )
