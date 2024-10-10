from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from graph_pes.core import ConservativePESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs import AtomicGraph


class GraphPESCalculator(Calculator):
    """
    ASE calculator wrapping any ConservativePESModel.

    Parameters
    ----------
    model
        The model to use for the calculation.
    device
        The device to use for the calculation, e.g. "cpu" or "cuda".
    **kwargs
        Properties passed to the :class:`ase.calculators.calculator.Calculator`
        base class.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: ConservativePESModel,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.device = device

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        if properties is None:
            properties = ["energy"]

        # call to base-class to set atoms attribute
        super().calculate(atoms)
        assert self.atoms is not None and isinstance(self.atoms, Atoms)

        # account for numerical inprecision by nudging the cutoff up slightly
        # for all well-implemented models this has no effect
        graph = to_atomic_graph(self.atoms, self.model.cutoff.item() + 0.001)
        graph: AtomicGraph = {k: v.to(self.device) for k, v in graph.items()}  # type: ignore

        results = {
            k: v.detach().cpu().numpy()
            for k, v in self.model.get_predictions(
                graph,
                properties=properties,  # type: ignore
            ).items()
            if k in properties
        }
        if "energy" in properties:
            results["energy"] = results["energy"].item()
        if "stress" in properties:
            results["stress"] = full_3x3_to_voigt_6_stress(results["stress"])

        self.results = results
