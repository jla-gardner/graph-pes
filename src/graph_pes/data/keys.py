from typing import Literal

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
LabelKey = Literal[
    "energy",
    "forces",
    "stress",
]
ALL_LABEL_KEYS = LabelKey.__args__  # type: ignore
