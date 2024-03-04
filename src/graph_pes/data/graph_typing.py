from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class AtomicGraph(TypedDict):
    # required properties
    atomic_numbers: Tensor
    cell: Tensor
    neighbour_index: Tensor
    _positions: Tensor
    _neighbour_cell_offsets: Tensor

    # optional labels
    energy: Tensor | None
    forces: Tensor | None
    stress: Tensor | None
    local_energies: Tensor | None


class AtomicGraphBatch(AtomicGraph):
    batch: Tensor
    ptr: Tensor
