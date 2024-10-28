from __future__ import annotations

import ase
import numpy
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    number_of_atoms,
    to_batch,
)
from graph_pes.graph_pes_model import GraphPESModel


class GraphPESCalculator(Calculator):
    """
    ASE calculator wrapping any GraphPESModel.

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
        model: GraphPESModel,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        if properties is None:
            properties = ["energy"]

        # call to base-class to set atoms attribute
        super().calculate(atoms)
        assert self.atoms is not None and isinstance(self.atoms, ase.Atoms)

        # account for numerical inprecision by nudging the cutoff up slightly
        graph = AtomicGraph.from_ase(
            self.atoms, self.model.cutoff.item() + 0.001
        )
        graph = graph.to(self.device)

        results = {
            k: v.detach().cpu().numpy()
            for k, v in self.model.predict(
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

    def batched_prediction(
        self,
        structures: list[AtomicGraph | ase.Atoms],
        properties: list[PropertyKey] | None = None,
        batch_size: int = 5,
    ) -> list[dict[str, numpy.ndarray]]:
        """
        Make a batched prediction on the given list of structures.

        Parameters
        ----------
        structures
            A list of :class:`~graph_pes.graphs.AtomicGraph` or
            :class:`ase.Atoms` objects.
        properties
            The properties to predict.
        batch_size
            The number of structures to predict at once.

        Examples
        --------
        >>> calculator = GraphPESCalculator(model, device="cuda")
        >>> structures = [Atoms(...), Atoms(...), Atoms(...)]
        >>> predictions = calculator.batched_prediction(
        ...     structures,
        ...     properties=["energy", "forces"],
        ...     batch_size=2,
        ... )
        >>> print(predictions)
        [{'energy': array(...), 'forces': array(...)},
         {'energy': array(...), 'forces': array(...)},
         {'energy': array(...), 'forces': array(...)}]
        """
        # recursive batching
        if len(structures) > batch_size:
            first_batch, rest = structures[:batch_size], structures[batch_size:]
            return self.batched_prediction(
                first_batch, properties, batch_size
            ) + self.batched_prediction(rest, properties, batch_size)

        if properties is None:
            properties = ["energy", "forces", "stress"]
        graphs = [
            AtomicGraph.from_ase(s, self.model.cutoff.item() + 0.001)
            if isinstance(s, ase.Atoms)
            else s
            for s in structures
        ]
        batch_preds = self.model.predict(
            to_batch(graphs), properties=properties
        )

        # de-convolve the batch predictions into the original structures
        preds_list = []
        atom_index = 0
        for structure_index, graph in enumerate(graphs):
            N = number_of_atoms(graph)
            preds = {}
            if "energy" in properties:
                preds["energy"] = batch_preds["energy"][structure_index]
            if "forces" in properties:
                preds["forces"] = batch_preds["forces"][
                    atom_index : atom_index + N
                ]
            if "stress" in properties:
                preds["stress"] = batch_preds["stress"][structure_index]
            if "local_energies" in properties:
                preds["local_energies"] = batch_preds["local_energies"][
                    atom_index : atom_index + N
                ]
            preds_list.append(preds)
            atom_index += N
        return preds_list
