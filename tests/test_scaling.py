import torch
from ase.build import molecule
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.models.scaling import UnScaledPESModel


class StupidModel(UnScaledPESModel):
    def predict_unscaled_energies(self, graph: AtomicGraph) -> torch.Tensor:
        return torch.ones_like(graph["atomic_numbers"]).float()


def test_scaling():
    model = StupidModel()

    # set the scaling terms for H and C
    with torch.no_grad():
        model._per_element_scaling[1] = 0.5
        model._per_element_scaling[6] = 2.0

    graph = to_atomic_graph(molecule("CH4"), cutoff=3)
    assert torch.equal(
        graph["atomic_numbers"],
        torch.tensor([6, 1, 1, 1, 1]),
    )

    unscaled = model.predict_unscaled_energies(graph)
    assert torch.equal(unscaled, torch.ones(5))

    local = model.predict_local_energies(graph)
    assert torch.equal(local, torch.tensor([2.0, 0.5, 0.5, 0.5, 0.5]))
