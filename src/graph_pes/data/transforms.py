from __future__ import annotations

from typing import Callable, Literal

from graph_pes.atomic_graph import AtomicGraph, is_batch
from graph_pes.utils.misc import random_rotation_matrix

Transform = Callable[[AtomicGraph], AtomicGraph]
TransformName = Literal["identity", "random_rotation"]


def identity_transform(graph: AtomicGraph) -> AtomicGraph:
    """A transform that returns the graph unchanged."""
    return graph


def random_rotation(graph: AtomicGraph) -> AtomicGraph:
    """Randomly rotate the atom positions, cell and forces (if present)."""

    if is_batch(graph):
        raise ValueError("Batched graphs are not supported")

    R = random_rotation_matrix()

    props = {**graph.properties}
    if "forces" in props:
        props["forces"] = props["forces"] @ R

    new_graph = graph._replace(
        R=graph.R @ R,
        cell=graph.cell @ R,
        properties=props,
    )
    return new_graph


def parse_transform(t: TransformName | Transform | None) -> Transform:
    if t is None or t == "identity":
        return identity_transform
    elif t == "random_rotation":
        return random_rotation
    elif callable(t):
        return t
    else:
        raise ValueError(f"Unknown transform: {t}")
