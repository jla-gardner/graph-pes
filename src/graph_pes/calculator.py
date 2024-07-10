from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from graph_pes.core import GraphPESModel, get_predictions
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs import AtomicGraph


class GraphPESCalculator(Calculator):
    """
    ASE calculator wrapping any GraphPESModel.

    Parameters
    ----------
    model
        The model to use for the calculation.
    cutoff
        The cutoff radius for the atomic environment.
    device
        The device to use for the calculation, e.g. "cpu" or "cuda".
    **kwargs
        Properties passed to the :code:`ase` base class.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: GraphPESModel,
        cutoff: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.cutoff = cutoff
        self.device = device

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        if properties is None:
            properties = ["energy"]

        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        graph = to_atomic_graph(atoms, self.cutoff)
        graph: AtomicGraph = {k: v.to(self.device) for k, v in graph.items()}  # type: ignore

        self.results = {
            k: v.detach().cpu().numpy()
            for k, v in get_predictions(self.model, graph).items()
            if k in properties
        }
