from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..util import _is_being_documented

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
LOCAL_ENERGIES = "local_energies"
FORCES = "forces"
STRESS = "stress"

# TODO: change label to property
# appease torchscript by forcing LabelKey to be a simple string
# at runtime
if TYPE_CHECKING or _is_being_documented():
    LabelKey = Literal[
        "energy",
        "forces",
        "stress",
        "local_energies",
    ]
else:
    LabelKey = str

ALL_LABEL_KEYS: list[LabelKey] = [ENERGY, FORCES, STRESS, LOCAL_ENERGIES]
