from __future__ import annotations

from dataclasses import dataclass

from graph_pes.core import GraphPESModel
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.graphs.operations import neighbour_distances, number_of_edges
from graph_pes.models import AdditionModel, SchNet
from graph_pes.models.offsets import FixedOffset
from helpers import graph_from_molecule
from torch import Tensor


@dataclass
class Stats:
    n_neighbours: int
    max_edge_length: float


class DummyModel(GraphPESModel):
    def __init__(self, name: str, cutoff: float, info: dict[str, Stats]):
        super().__init__()
        self.name = name
        self.cutoff = cutoff
        self.info = info

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        # insert statistics here
        self.info[self.name] = Stats(
            n_neighbours=number_of_edges(graph),
            max_edge_length=neighbour_distances(graph).max().item(),
        )
        # dummy return value
        return graph["atomic_numbers"].float()


def test_cutoff_trimming():
    info = {}
    graph = graph_from_molecule("CH3CH2OCH3", cutoff=5.0)

    large_model = DummyModel("large", cutoff=5.0, info=info)
    small_model = DummyModel("small", cutoff=3.0, info=info)

    # forward passes to gather info
    large_model(graph)
    small_model(graph)

    # check that graph remains unchanged
    assert "_rmax" not in graph

    # check that cutoff filtering is working
    assert info["large"].n_neighbours == number_of_edges(graph)
    assert info["small"].n_neighbours < number_of_edges(graph)

    assert (
        info["large"].max_edge_length == neighbour_distances(graph).max().item()
    )
    assert (
        info["small"].max_edge_length < neighbour_distances(graph).max().item()
    )


def test_model_cutoffs():
    model = AdditionModel(
        small=SchNet(cutoff=3.0),
        large=SchNet(cutoff=5.0),
    )

    assert model.cutoff == 5.0

    model = FixedOffset()
    assert model.cutoff is None
