from __future__ import annotations

import functools
import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, Union, overload

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
from graph_pes.data.ase_db import ASEDatabase
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import MultiSequence, slice_to_range, uniform_repr
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

    def setup(self) -> None:
        """
        Set-up the data for this specific instance of the dataset.

        Called on every process in the distributed setup. May be called
        multiple times.
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


class ConcatDataset(GraphDataset):
    """
    A dataset that concatenates multiple :class:`GraphDataset` instances.
    Useful for e.g. training on datasets from multiple files simultaneously:

    .. code-block:: yaml

        data:
            train:
                +ConcatDataset:
                    dimers:
                        +file_dataset:
                            path: dimers.xyz
                            cutoff: 5.0
                    crystals:
                        +file_dataset:
                            path: crystals.xyz
                            cutoff: 5.0

            valid:
                ...


    Parameters
    ----------
    datasets
        The collection of :class:`GraphDataset` instances to concatenate.
        The keys are arbitrary names for the datasets, and the values are
        the :class:`GraphDataset` instances.
    """

    def __init__(self, **datasets: GraphDataset):
        self.datasets = datasets
        self._lengths = {k: len(v) for k, v in datasets.items()}

    def __len__(self):
        return sum(self._lengths.values())

    def __getitem__(self, index: int) -> AtomicGraph:
        for k, v in self._lengths.items():
            if index < v:
                return self.datasets[k][index]
            index -= v
        raise IndexError(f"Index {index} is out of bounds for the dataset")

    def prepare_data(self) -> None:
        for dataset in self.datasets.values():
            dataset.prepare_data()

    def setup(self) -> None:
        for dataset in self.datasets.values():
            dataset.setup()

    @property
    def properties(self) -> list[PropertyKey]:
        return list(
            set(
                p
                for dataset in self.datasets.values()
                for p in dataset.properties
            )
        )

    @property
    def graphs(self) -> MultiSequence[AtomicGraph]:
        return MultiSequence(
            [dataset.graphs for dataset in self.datasets.values()],
        )


class ASEToGraphsConverter(Sequence[AtomicGraph]):
    def __init__(
        self,
        structures: Sequence[ase.Atoms],
        cutoff: float,
        property_mapping: Mapping[str, PropertyKey] | None = None,
        others_to_include: list[str] | None = None,
    ):
        self.structures = structures
        self.cutoff = cutoff
        self.property_mapping = property_mapping
        self.others_to_include = others_to_include

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
            others_to_include=self.others_to_include,
        )

    def __len__(self) -> int:
        return len(self.structures)


# use the locache library to cache the graphs that result from this
# transform to disk: this means that multiple training runs on the
# same dataset will be able to reuse the same graphs, massively speeding
# up the start to training for the (n>1)th run
# to ensure that the graphs we get back are of the correct dtype,
# we need to pass the current torch dtype to this caching function
@locache.persist
def get_all_graphs_and_cache_to_disk(
    converter: ASEToGraphsConverter,
    torch_dtype: torch.dtype,
) -> list[AtomicGraph]:
    logger.info(
        f"Caching neighbour lists for {len(converter)} structures "
        f"with cutoff {converter.cutoff}, property mapping "
        f"{converter.property_mapping} and torch dtype {torch_dtype}"
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
    others_to_include
        A list of properties to include in the ``graph.other`` field
        that are present as per-atom or per-structure properties on the
        :class:`ase.Atoms` objects.
    """

    def __init__(
        self,
        structures: Sequence[ase.Atoms],
        cutoff: float,
        pre_transform: bool = False,
        property_mapping: Mapping[str, PropertyKey] | None = None,
        others_to_include: list[str] | None = None,
    ):
        super().__init__(
            ASEToGraphsConverter(
                structures, cutoff, property_mapping, others_to_include
            ),
        )
        self.pre_transform = pre_transform

    def prepare_data(self):
        if self.pre_transform:
            # cache the graphs to disk - this is done on rank-0 only
            # and means that expensive data pre-transforms don't need to be
            # recomputed on each rank in the distributed setup
            get_all_graphs_and_cache_to_disk(
                self.graphs, torch.get_default_dtype()
            )

    def setup(self) -> None:
        if self.pre_transform and isinstance(self.graphs, ASEToGraphsConverter):
            # load the graphs from disk
            actual_graphs = get_all_graphs_and_cache_to_disk(
                self.graphs, torch.get_default_dtype()
            )
            self.graphs = actual_graphs


@dataclass
class DatasetCollection:
    """
    A convenience container for training, validation, and optional test sets.
    """

    train: GraphDataset
    """The training dataset."""
    valid: GraphDataset
    """The validation dataset."""
    test: Union[GraphDataset, dict[str, GraphDataset], None] = None  # noqa: UP007
    """An optional test dataset, or collection of named test datasets."""

    def __repr__(self) -> str:
        kwargs = {"train": self.train, "valid": self.valid}
        if self.test is not None:
            kwargs["test"] = self.test  # type: ignore
        return uniform_repr(
            self.__class__.__name__,
            **kwargs,  # type: ignore
        )


def load_atoms_dataset(
    id: str | pathlib.Path,
    cutoff: float,
    n_train: int,
    n_valid: int,
    n_test: int | None = None,
    split: Literal["random", "sequential"] = "random",
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, PropertyKey] | None = None,
    others_to_include: list[str] | None = None,
) -> DatasetCollection:
    """
    Load an dataset of :class:`ase.Atoms` objects using
    `load-atoms <https://jla-gardner.github.io/load-atoms/>`__,
    convert them to :class:`~graph_pes.AtomicGraph` instances, and split into
    train and valid sets.

    Examples
    --------
    Load a subset of the QM9 dataset. Ensure that the ``U0`` property is
    mapped to ``energy``:

    >>> load_atoms_dataset(
    ...     "QM9",
    ...     cutoff=5.0,
    ...     n_train=1_000,
    ...     n_valid=100,
    ...     n_test=100,
    ...     property_map={"U0": "energy"},
    ... )

    Use this to specify a complete collection of datasets (train, val and test)
    in a YAML configuration file:

    .. code-block:: yaml

        data:
            +load_atoms_dataset:
                id: QM9
                cutoff: 5.0
                n_train: 1_000
                n_valid: 100
                n_test: 100
                property_map:
                    U0: energy

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
    n_test:
        The number of test structures. If ``None``, no test set is created.
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
    others_to_include:
        A list of properties to include in the ``graph.other`` field
        that are present as per-atom or per-structure properties on the
        :class:`ase.Atoms` objects.

    Returns
    -------
    DatasetCollection
        A collection of training, validation, and optional test datasets.
    """
    _dataset_factory = functools.partial(
        ASEToGraphDataset,
        cutoff=cutoff,
        pre_transform=pre_transform,
        property_mapping=property_map,
        others_to_include=others_to_include,
    )

    structures = SequenceSampler(load_dataset(id))

    if split == "random":
        structures = structures.shuffled(seed)

    train = structures[:n_train]
    val = structures[n_train : n_train + n_valid]

    return DatasetCollection(
        train=_dataset_factory(train),
        valid=_dataset_factory(val),
        test=_dataset_factory(
            structures[n_train + n_valid : n_train + n_valid + n_test],
        )
        if n_test is not None
        else None,
    )


def file_dataset(
    path: str | pathlib.Path,
    cutoff: float,
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    pre_transform: bool = True,
    property_map: dict[str, PropertyKey] | None = None,
    others_to_include: list[str] | None = None,
) -> ASEToGraphDataset:
    """
    Load an ASE dataset from a file that is either:

    * any plain-text file that can be read by :func:`ase.io.read`, e.g. an
      ``.xyz`` file
    * a ``.db`` file containing a SQLite database of :class:`ase.Atoms` objects
      that is readable as an `ASE database <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`__.
      Under the hood, this uses the :class:`~graph_pes.data.ase_db.ASEDatabase`
      class - see there for more details.

    Examples
    --------
    Load a dataset from a file, ensuring that the ``energy`` property is
    mapped to ``U0``:

    >>> file_dataset(
    ...     "training_data.xyz",
    ...     cutoff=5.0,
    ...     property_map={"U0": "energy"},
    ... )

    By default, this function gets called on any collection of arguments
    specified in a YAML configuration file:

    .. code-block:: yaml

        data:
            train:
                path: training_data.xyz
                n: 1000
                cutoff: 5.0
                property_map:
                    U0: energy

    is equivalent to:

    .. code-block:: yaml

        data:
            train:
                +file_dataset:
                    path: training_data.xyz
                    cutoff: 5.0
                    property_map:
                        U0: energy

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
    others_to_include:
        A list of properties to include in the ``graph.other`` field
        that are present as per-atom or per-structure properties on the
        :class:`ase.Atoms` objects.

    Returns
    -------
    ASEToGraphDataset
        The ASE dataset.
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix == ".db":
        structures = ASEDatabase(path)
    else:
        structures = ase.io.read(path, index=":")
        assert isinstance(structures, list)

    structure_collection = SequenceSampler(structures)
    if shuffle:
        structure_collection = structure_collection.shuffled(seed)

    if n is not None:
        structure_collection = structure_collection[:n]

    return ASEToGraphDataset(
        structure_collection,
        cutoff,
        pre_transform,
        property_map,
        others_to_include,
    )
