from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# graph properties
ATOMIC_NUMBERS = "atomic_numbers"
CELL = "cell"
NEIGHBOUR_INDEX = "neighbour_index"
_POSITIONS = "_positions"
_NEIGHBOUR_CELL_OFFSETS = "_neighbour_cell_offsets"

# batch properties
BATCH = "batch"
PTR = "ptr"

# labels
ENERGY = "energy"
FORCES = "forces"
STRESS = "stress"

# appease torchscript by forcing LabelKey to be a simple string
# at runtime
if TYPE_CHECKING:
    LabelKey = Literal[
        "energy",
        "forces",
        "stress",
    ]
else:
    LabelKey = str

ALL_LABEL_KEYS: list[LabelKey] = [ENERGY, FORCES, STRESS]
