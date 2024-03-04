from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from torch import Tensor
from torch_geometric.utils import scatter

from . import keys
from .graph_typing import AtomicGraph as AtomicGraphType
from .graph_typing import AtomicGraphBatch as AtomicGraphBatchType

__all__ = [
    "AtomicGraph",
    "allow_position_access",
    "is_batch",
    "neighbour_vectors",
    "neighbour_distances",
]


if TYPE_CHECKING:
    # when people are writing code, we want correct types
    AtomicGraph = AtomicGraphType
    AtomicGraphBatch = AtomicGraphBatchType
else:
    # at runtime, we want @torch.jit.script to work, and this requires
    # the key-type to be a string
    AtomicGraph = dict[str, torch.Tensor]
    AtomicGraphBatch = dict[str, torch.Tensor]


# TODO work out how to do this while keeping torchscript happy
# currently not allowed global variable access
_WARN_ON_POS_ACCESS = False


class AtomicGraph_Impl(dict):
    def __getitem__(self, key):
        if _WARN_ON_POS_ACCESS and key == keys._POSITIONS:
            warnings.warn(
                (
                    "Accessing the raw atomic positions is discouraged. "
                    "Are you trying to access information derived from "
                    "relative atomic positions? If so, use "
                    "`neighbour_vectors` or `neighbour_distances` instead."
                ),
                stacklevel=2,
            )

        return super().__getitem__(key)

    def __repr__(self):
        return f"AtomicGraph({super().__repr__()})"


@contextmanager
def allow_position_access():
    global _WARN_ON_POS_ACCESS
    _WARN_ON_POS_ACCESS = False
    try:
        yield
    finally:
        _WARN_ON_POS_ACCESS = True


def is_batch(graph: AtomicGraph) -> bool:
    """Is the data in `graph` batched?"""

    return "batch" in graph


def is_periodic(graph: AtomicGraph) -> bool:
    """Is the data in `graph` periodic?"""

    return torch.any(graph[keys.CELL] != 0).item()  # type: ignore


def neighbour_vectors(graph: AtomicGraph) -> Tensor:
    """
    Get the vector between each pair of atoms specified in the
    `graph`'s `neighbour_index` property.

    Parameters
    ----------
    graph
        The graph to extract the neighbour vectors from.
    """

    # with allow_position_access():
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
    `graph`'s `neighbour_index` property.

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


def convert_to_atomic_graph(
    structure: ase.Atoms,
    cutoff: float,
    property_mapping: dict[keys.LABEL_KEY, str] | None = None,
) -> AtomicGraph:
    """
    Convert an ASE Atoms object to an AtomicGraph.

    Parameters
    ----------
    structures
        The ASE Atoms object.
    cutoff
        The cutoff distance for neighbour finding.
    atom_labels
        The names of any additional per-atom labels to include in the graph.
        If not provided, all possible labels will be included.
    structure_labels
        The names of any additional per-structure labels to include in the
        graph. If not provided, all possible labels will be included.
    property_mapping
        A mapping from custom property labels to the standard property
        names (e.g. "totalenergy" -> "energy")
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

    return graph  # type: ignore


#### BATCHING ####


def batch_graphs(graphs: list[AtomicGraph]) -> AtomicGraphBatch:
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

    # - per atom and per edge labels are concatenated along the first axis
    for label in [keys.FORCES, keys.LOCAL_ENERGIES]:
        if label in graphs[0]:
            graph[label] = torch.cat([g[label] for g in graphs])

    return graph  # type: ignore


def sum_per_structure(x: Tensor, graph: AtomicGraph) -> Tensor:
    """
    Sum a per-atom property to get a per-structure property.

    Parameters
    ----------
    x
        The per-atom property to sum.
    graph
        The graph to sum the property for.
    """

    if is_batch(graph):
        # we have more than one structure: sum over local energies to
        # get a total energy for each structure
        batch = graph[keys.BATCH]  # type: ignore
        return scatter(x, batch, dim=0, reduce="sum")
    else:
        # we only have one structure: sum over all the atoms
        return x.sum()
