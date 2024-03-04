from typing import Literal

ATOMIC_NUMBERS = "atomic_numbers"
CELL = "cell"
NEIGHBOUR_INDEX = "neighbour_index"
_POSITIONS = "_positions"
_NEIGHBOUR_CELL_OFFSETS = "_neighbour_cell_offsets"

ENERGY = "energy"
FORCES = "forces"
STRESS = "stress"
LOCAL_ENERGIES = "local_energies"

BATCH = "batch"
PTR = "ptr"


LABEL_KEY = Literal[
    "energy",
    "forces",
    "stress",
    "local_energies",
]
ALL_LABEL_KEYS = LABEL_KEY.__args__  # type: ignore
