from __future__ import annotations

from typing import Iterable, Mapping

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list

from graph_pes.graphs import DEFAULT_CUTOFF, LabelledGraph, keys, with_nice_repr


def to_atomic_graph(
    structure: ase.Atoms,
    cutoff: float = DEFAULT_CUTOFF,
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

    graph = with_nice_repr(
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
        if name_on_structure in structure.info:
            data = structure.info[name_on_structure]
            if isinstance(data, (int, float)):
                graph[label] = torch.scalar_tensor(data, dtype=torch.float)
            else:
                graph[label] = torch.FloatTensor(
                    structure.info[name_on_structure]
                )
        elif name_on_structure in structure.arrays:
            graph[label] = torch.FloatTensor(
                structure.arrays[name_on_structure]
            )
        else:
            raise KeyError(
                f"Property {name_on_structure} not found for {structure}"
            )

    return graph  # type: ignore


def to_atomic_graphs(
    structures: Iterable[ase.Atoms],
    cutoff: float = DEFAULT_CUTOFF,
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
