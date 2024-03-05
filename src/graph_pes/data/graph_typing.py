from __future__ import annotations

from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired


class AtomicGraph(TypedDict):
    # required properties
    atomic_numbers: Tensor
    cell: Tensor
    neighbour_index: Tensor
    _positions: Tensor
    _neighbour_cell_offsets: Tensor

    # optional labels
    energy: NotRequired[Tensor]
    forces: NotRequired[Tensor]
    stress: NotRequired[Tensor]
    local_energies: NotRequired[Tensor]


class AtomicGraphBatch(AtomicGraph):
    batch: Tensor
    ptr: Tensor
