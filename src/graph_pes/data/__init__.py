from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Iterator, Mapping, Sequence, overload

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from typing_extensions import TypeAlias

from graph_pes.nn import left_aligned_mul
from graph_pes.util import _is_being_documented

from . import keys
from .graph_typing import AtomicGraph as AtomicGraphType
from .graph_typing import AtomicGraphBatch as AtomicGraphBatchType
from .graph_typing import LabelledBatch as LabelledBatchType
from .graph_typing import LabelledGraph as LabelledGraphType
from .utils import random_split

__all__ = [
    "AtomicGraph",
    "is_batch",
    "neighbour_vectors",
    "neighbour_distances",
    "random_split",
    "to_atomic_graph",
    "to_atomic_graphs",
]


if TYPE_CHECKING or _is_being_documented():
    # when people are writing code, we want correct types
    # and so use TypedDicts to raise warnings in IDEs
    AtomicGraph: TypeAlias = AtomicGraphType
    LabelledGraph: TypeAlias = LabelledGraphType
    AtomicGraphBatch: TypeAlias = AtomicGraphBatchType
    LabelledBatch: TypeAlias = LabelledBatchType
else:
    # at runtime, we want @torch.jit.script to work, and this requires
    # the key type to be a string, not Literal from TypedDict
    AtomicGraph: TypeAlias = Dict[str, torch.Tensor]
    LabelledGraph: TypeAlias = Dict[str, torch.Tensor]
    AtomicGraphBatch: TypeAlias = Dict[str, torch.Tensor]
    LabelledBatch: TypeAlias = Dict[str, torch.Tensor]


class AtomicGraph_Impl(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        info = {}

        if is_batch(self):  # type: ignore
            name = "AtomicGraphBatch"
            info["structures"] = number_of_structures(self)  # type: ignore
        else:
            name = "AtomicGraph"

        info["atoms"] = number_of_atoms(self)  # type: ignore
        info["edges"] = self[keys.NEIGHBOUR_INDEX].shape[1]  # type: ignore
        info["has_cell"] = has_cell(self)  # type: ignore

        labels = [label for label in keys.ALL_LABEL_KEYS if label in self]
        if labels:
            info["labels"] = labels

        info_str = ", ".join(f"{k}: {v}" for k, v in info.items())
        return f"{name}({info_str})"


def is_batch(graph: AtomicGraph) -> bool:
    """Is the data in `graph` batched?"""

    return "batch" in graph


def number_of_atoms(graph: AtomicGraph) -> int:
    """
    Get the number of atoms in the ``graph``.

    Parameters
    ----------
    graph
        The atomic graph
    """

    return graph[keys.ATOMIC_NUMBERS].shape[0]


def number_of_edges(graph: AtomicGraph) -> int:
    """
    Get the number of edges in the ``graph``.

    Parameters
    ----------
    graph
        The atomic graph
    """

    return graph[keys.NEIGHBOUR_INDEX].shape[1]


def is_local_property(x: Tensor, graph: AtomicGraph) -> bool:
    """
    Is the property ``x`` local to each atom in the ``graph``?

    Parameters
    ----------
    x
        The property to check.
    graph
        The graph to check the property for.
    """

    return len(x.shape) > 0 and x.shape[0] == number_of_atoms(graph)


def has_cell(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a structure with a defined unit cell?

    Parameters
    ----------
    graph
        The graph to check.
    """

    # TODO shouldn't this also check for at least one edge vector
    # across the periodic boundary?
    return bool(torch.any(graph[keys.CELL] != 0).item())


def neighbour_vectors(graph: AtomicGraph) -> Tensor:
    """
    Get the vector between each pair of atoms specified in the
    ``graph``'s ``"neighbour_index"`` property, respecting periodic
    boundary conditions where present.

    Parameters
    ----------
    graph
        The graph to extract the neighbour vectors from.
    """

    positions = graph[keys._POSITIONS]

    if not is_batch(graph):
        batch = torch.zeros_like(graph["atomic_numbers"])
        cell = graph["cell"].unsqueeze(0)
    else:
        batch = graph["batch"]  # type: ignore
        cell = graph["cell"]

    # avoid tuple de-structuring to keep torchscript happy
    i, j = graph["neighbour_index"][0], graph["neighbour_index"][1]
    cell_per_edge = cell[batch[i]]
    distance_offsets = torch.einsum(
        "kl,klm->km",
        graph["_neighbour_cell_offsets"].float(),
        cell_per_edge,
    )
    neighbour_positions = positions[j] + distance_offsets
    return neighbour_positions - positions[i]


def neighbour_distances(graph: AtomicGraph) -> Tensor:
    """
    Get the distance between each pair of atoms specified in the
    ``graph``'s ``neighbour_index`` property, respecting periodic
    boundary conditions where present.

    Parameters
    ----------
    graph
        The graph to extract the neighbour distances from.
    """
    return torch.linalg.norm(neighbour_vectors(graph), dim=-1)


# TODO: test if we need this caching behaviour for speed up?
# if so, add reset_graph argument to prediction functions
# (to handle ensemble predictions)?? might not be necessary since we aren't
# deleting tensors, just dropping them from a dict, and so gc won't remove
# if still part of computation graph

# def prepare_for_forward_pass(graph: AtomicGraph) -> AtomicGraph:
#     # remove everything with starting with "__cache"
#     return {k: v for k, v in graph.items() if not k.startswith("__cache")}


# def _property_name(name: str) -> str:
#     return f"__cache_{name}"


# def add_property(x: Tensor, name: str, graph: AtomicGraph):
#     graph[_property_name(name)] = x


# def get_property(name: str, graph: AtomicGraph) -> Tensor:
#     return graph[_property_name(name)]


def to_atomic_graph(
    structure: ase.Atoms,
    cutoff: float,
    property_mapping: Mapping[keys.LabelKey, str] | None = None,
) -> LabelledGraph:
    """
    Convert an ASE Atoms object to an AtomicGraph.

    Parameters
    ----------
    structure
        The ASE Atoms object.
    cutoff
        The cutoff distance for neighbour finding.
    property_mapping
        An optional mapping defining how relevant properties are labelled
        on the ASE Atoms object. If not provided, then the default mapping
        is used:

        .. code-block:: python

                {
                    keys.ENERGY: "energy",
                    keys.FORCES: "forces",
                    keys.STRESS: "stress",
                }
    """

    i, j, offsets = neighbor_list("ijS", structure, cutoff)

    graph = AtomicGraph_Impl(
        {
            keys.ATOMIC_NUMBERS: torch.LongTensor(structure.numbers),
            keys.CELL: torch.FloatTensor(structure.cell.array),
            keys._POSITIONS: torch.FloatTensor(structure.positions),
            keys.NEIGHBOUR_INDEX: torch.LongTensor(np.vstack([i, j])),
            keys._NEIGHBOUR_CELL_OFFSETS: torch.LongTensor(offsets),
        }
    )

    if property_mapping is None:
        all_keys: set[str] = set(structure.info) | set(structure.arrays)
        property_mapping = {k: k for k in keys.ALL_LABEL_KEYS if k in all_keys}

    for label, name_on_structure in property_mapping.items():
        if label in structure.info:
            data = structure.info[label]
            if isinstance(data, (int, float)):
                graph[label] = torch.scalar_tensor(data, dtype=torch.float)
            else:
                graph[label] = torch.FloatTensor(
                    structure.info[name_on_structure]
                )
        elif label in structure.arrays:
            graph[label] = torch.FloatTensor(
                structure.arrays[name_on_structure]
            )

    return graph  # type: ignore


def to_atomic_graphs(
    structures: Sequence[ase.Atoms],
    cutoff: float,
    property_mapping: dict[keys.LabelKey, str] | None = None,
) -> list[LabelledGraph]:
    """
    Equivalent to :code:`[to_atomic_graph(s, cutoff, property_mapping)
    for s in structures]`

    Parameters
    ----------
    structures
        The ASE Atoms objects.
    cutoff
        The cutoff distance for neighbour finding.
    property_mapping
        An optional mapping defining how relevant properties are labelled
        on the ASE Atoms object.
    """
    return [to_atomic_graph(s, cutoff, property_mapping) for s in structures]


def sum_over_neighbours(p: Tensor, graph: AtomicGraph) -> Tensor:
    r"""
    Shape-preserving sum over neighbours of a per-edge property, :math:`A_{ij}`,
    to get a per-atom property, :math:`B_i`:

    .. math::
        B_i = \sum_{j \in \mathcal{N}_i} A_{ij}

    where:

    * :math:`\mathcal{N}_i` is the set of neighbours of atom :math:`i`.
    * :math:`A_{ij}` is the property of the edge between atoms :math:`i` and
      :math:`j`.
    * :math:`A` is of shape :code:`(E, ...)` and :math:`B` is of shape
      :code:`(N, ...)` where :math:`E` is the number of edges and :math:`N` is
      the number of atoms. :code:`...` denotes any number of additional
      dimensions, including none.
    * :math:`B_i` = 0 if :math:`|\mathcal{N}_i| = 0`.

    Parameters
    ----------
    p
        The per-edge property to sum.
    graph
        The graph to sum the property for.
    """
    N = number_of_atoms(graph)
    central_atoms = graph[keys.NEIGHBOUR_INDEX][0]  # shape: (E,)

    # optimised implementations for common cases
    if p.dim() == 1:
        zeros = torch.zeros(N, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(0, central_atoms, p)

    elif p.dim() == 2:
        C = p.shape[1]
        zeros = torch.zeros(N, C, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(0, central_atoms.unsqueeze(1).expand(-1, C), p)

    shape = (N,) + p.shape[1:]
    zeros = torch.zeros(shape, dtype=p.dtype, device=p.device)

    if p.shape[0] == 0:
        # return all zeros if there are no atoms
        return zeros

    # create `index`, where index[e].shape = p.shape[1:]
    # and (index[e] == central_atoms[e]).all()
    ones = torch.ones_like(zeros)
    index = left_aligned_mul(ones, central_atoms).long()
    return zeros.scatter_add(0, index, p)


#### BATCHING ####


@overload
def to_batch(graphs: Sequence[LabelledGraph]) -> LabelledBatch: ...
@overload
def to_batch(graphs: Sequence[AtomicGraph]) -> AtomicGraphBatch: ...


@torch.no_grad()
def to_batch(graphs: Sequence[AtomicGraph]) -> AtomicGraphBatch:
    """
    Collate a sequence of atomic graphs into a single batch object.

    Parameters
    ----------
    graphs
        The graphs to collate.
    """
    # easy properties: just cat these together
    Z = torch.cat([g[keys.ATOMIC_NUMBERS] for g in graphs])
    positions = torch.cat([g[keys._POSITIONS] for g in graphs])
    neighbour_offsets = torch.cat(
        [g[keys._NEIGHBOUR_CELL_OFFSETS] for g in graphs]
    )

    # stack cells along a new batch dimension
    cells = torch.stack([g[keys.CELL] for g in graphs])

    # standard way to caculaute the batch and ptr properties
    batch = torch.cat(
        [
            torch.full_like(g[keys.ATOMIC_NUMBERS], fill_value=i)
            for i, g in enumerate(graphs)
        ]
    )
    ptr = torch.tensor(
        [0] + [g[keys.ATOMIC_NUMBERS].shape[0] for g in graphs]
    ).cumsum(dim=0)

    # use the ptr to increment the neighbour index appropriately
    neighbour_index = torch.cat(
        [g[keys.NEIGHBOUR_INDEX] + ptr[i] for i, g in enumerate(graphs)], dim=1
    )

    graph = AtomicGraph_Impl(
        {
            keys.ATOMIC_NUMBERS: Z,
            keys._POSITIONS: positions,
            keys.NEIGHBOUR_INDEX: neighbour_index,
            keys.CELL: cells,
            keys._NEIGHBOUR_CELL_OFFSETS: neighbour_offsets,
            keys.BATCH: batch,
            keys.PTR: ptr,
        }
    )

    # - per structure labels are concatenated along a new batch axis (0)
    for label in [keys.ENERGY, keys.STRESS]:
        if label in graphs[0]:
            graph[label] = torch.stack([g[label] for g in graphs])

    # - per atom labels are concatenated along the first axis
    if keys.FORCES in graphs[0]:
        graph[keys.FORCES] = torch.cat([g[keys.FORCES] for g in graphs])  # type: ignore

    return graph  # type: ignore


def sum_per_structure(x: Tensor, graph: AtomicGraph) -> Tensor:
    r"""
    Shape-preserving sum of a per-atom property, :math:`p`, to get a
    per-structure property, :math:`P`:

    If a single structure, containing ``N`` atoms, is used, then
    :math:`P = \sum_i p_i`, where:

    * :math:`p_i` is of shape ``(N, ...)``
    * :math:`P` is of shape ``(...)``
    * ``...`` denotes any
      number of additional dimensions, including ``None``.

    If a batch of ``S`` structures, containing a total of ``N`` atoms, is
    used, then :math:`P_k = \sum_{k \in K} p_k`, where:

    * :math:`K` is the collection of all atoms in structure :math:`k`
    * :math:`p_i` is of shape ``(N, ...)``
    * :math:`P` is of shape ``(S, ...)``
    * ``...`` denotes any
      number of additional dimensions, including ``None``.

    Parameters
    ----------
    x
        The per-atom property to sum.
    graph
        The graph to sum the property for.

    Examples
    --------
    Single graph case:

    >>> import torch
    >>> from ase.build import molecule
    >>> from graph_pes.data import sum_per_structure, to_atomic_graph
    >>> water = molecule("H2O")
    >>> graph = to_atomic_graph(water, cutoff=1.5)
    >>> # summing over a vector gives a scalar
    >>> sum_per_structure(torch.ones(3), graph)
    tensor(3.)
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(3, 2, 3), graph).shape
    torch.Size([2, 3])

    Batch case:

    >>> import torch
    >>> from ase.build import molecule
    >>> from graph_pes.data import sum_per_structure, to_atomic_graph, to_batch
    >>> water = molecule("H2O")
    >>> graph = to_atomic_graph(water, cutoff=1.5)
    >>> batch = to_batch([graph, graph])
    >>> batch
    AtomicGraphBatch(structures: 2, atoms: 6, edges: 8, has_cell: False)
    >>> # summing over a vector gives a tensor
    >>> sum_per_structure(torch.ones(6), graph)
    tensor([3., 3.])
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(6, 3, 4), graph).shape
    torch.Size([2, 3, 4])
    """

    if is_batch(graph):
        batch = graph[keys.BATCH]  # type: ignore
        shape = (number_of_structures(graph),) + x.shape[1:]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return zeros.scatter_add(0, batch, x)
    else:
        return x.sum(dim=0)


def number_of_structures(batch: AtomicGraph) -> int:
    """
    Get the number of structures in the ``batch``.

    Parameters
    ----------
    batch
        The batch to get the number of structures for.
    """

    if not is_batch(batch):
        return 1
    return batch[keys.PTR].shape[0] - 1  # type: ignore


def structure_sizes(batch: AtomicGraph) -> Tensor:
    """
    Get the number of atoms in each structure in the ``batch``, of shape
    ``(S,)`` where ``S`` is the number of structures.

    Parameters
    ----------
    batch
        The batch to get the structure sizes for.

    Examples
    --------
    >>> len(graphs)
    3
    >>> [number_of_atoms(g) for g in graphs]
    [3, 4, 5]
    >>> structure_sizes(to_batch(graphs))
    tensor([3, 4, 5])
    """

    if not is_batch(batch):
        return torch.scalar_tensor(number_of_atoms(batch))

    return batch[keys.PTR][1:] - batch[keys.PTR][:-1]  # type: ignore


class AtomicDataLoader(TorchDataLoader):
    r"""
    A data loader for merging :class:`AtomicGraph` objects into
    :class:`AtomicGraphBatch` objects.

    Parameters
    ----------
    dataset: Sequence[AtomicGraph]
        The dataset to load.
    batch_size: int
        The batch size.
    shuffle: bool
        Whether to shuffle the dataset.
    **kwargs:
        Additional keyword arguments are passed to the underlying
        :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Sequence[AtomicGraph],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            warnings.warn(
                "graph-pes uses a custom collate_fn (`collate_atomic_graphs`), "
                "are you sure you want to override this?",
                stacklevel=2,
            )

        collate_fn = kwargs.pop("collate_fn", to_batch)

        super().__init__(
            dataset,  # type: ignore
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    # add typing for mypy purposes
    def __iter__(self) -> Iterator[AtomicGraphBatch]:
        return super().__iter__()
