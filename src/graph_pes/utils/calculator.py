from __future__ import annotations

from typing import TypeVar

import ase
import numpy
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from graph_pes.atomic_graph import AtomicGraph, PropertyKey, has_cell, to_batch
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import groups_of, pairs


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
        properties: list[str] | list[PropertyKey] | None = None,
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

    def calculate_all(
        self,
        structures: list[AtomicGraph | ase.Atoms],
        properties: list[PropertyKey] | None = None,
        batch_size: int = 5,
    ) -> list[dict[PropertyKey, numpy.ndarray]]:
        """
        Semantically identical to:

        .. code-block::

            [calc.calculate(structure, properties) for structure in structures]

        but with significant acceleration due to internal batching.

        Parameters
        ----------
        structures
            A list of :class:`~graph_pes.AtomicGraph` or
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

        # defaults
        graphs = [
            AtomicGraph.from_ase(s, self.model.cutoff.item() + 0.001)
            if isinstance(s, ase.Atoms)
            else s
            for s in structures
        ]
        if properties is None:
            properties = ["energy", "forces"]
            if all(map(has_cell, graphs)):
                properties.append("stress")

        results: list[dict[PropertyKey, numpy.ndarray]] = []
        for batch in map(to_batch, groups_of(batch_size, graphs)):
            predictions = self.model.predict(batch, properties)
            seperated = _seperate(predictions, batch)
            results.extend(map(_to_numpy, seperated))

        if "stress" in properties:
            for r in results:
                r["stress"] = full_3x3_to_voigt_6_stress(r["stress"])

        return results


## utils ##

T = TypeVar("T")
TensorLike = TypeVar("TensorLike", torch.Tensor, numpy.ndarray)


def _to_numpy(results: dict[T, torch.Tensor]) -> dict[T, numpy.ndarray]:
    return {key: tensor.detach().numpy() for key, tensor in results.items()}


def _seperate(
    batched_prediction: dict[PropertyKey, TensorLike],
    batch: AtomicGraph,
) -> list[dict[PropertyKey, TensorLike]]:
    preds_list = []

    for idx, (start, stop) in enumerate(pairs(batch.other["ptr"])):
        preds = {}

        # per-structure properties
        if "energy" in batched_prediction:
            preds["energy"] = batched_prediction["energy"][idx]
        if "stress" in batched_prediction:
            preds["stress"] = batched_prediction["stress"][idx]

        # per-atom properties
        if "forces" in batched_prediction:
            preds["forces"] = batched_prediction["forces"][start:stop]
        if "local_energies" in batched_prediction:
            preds["local_energies"] = batched_prediction["local_energies"][
                start:stop
            ]

        preds_list.append(preds)

    return preds_list


def merge_predictions(
    predictions: list[dict[PropertyKey, numpy.ndarray]],
) -> dict[PropertyKey, numpy.ndarray]:
    """
    Take a list of property predictions and merge them
    in a sensible way.

    TODO write

    TODO examples
    """
    merged: dict[PropertyKey, numpy.ndarray] = {}

    # stack per-structure properties along new axis
    for key in ["energy", "stress"]:
        if key in predictions[0]:
            merged[key] = numpy.stack([p[key] for p in predictions])

    # concatenat per-atom properties along the first axis
    for key in ["forces", "local_energies"]:
        if key in predictions[0]:
            merged[key] = numpy.concatenate([p[key] for p in predictions])

    return merged
