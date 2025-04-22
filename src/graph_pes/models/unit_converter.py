from __future__ import annotations

import torch

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.graph_pes_model import GraphPESModel


class UnitConverter(GraphPESModel):
    r"""
    A wrapper that converts the units of the energy, forces and stress
    predictions of an underlying model.

    Parameters
    ----------
    model
        The underlying model.
    energy_to_eV
        The conversion factor for energy, such that the
        ``model.predict_energy(graph) * energy_to_eV`` gives the
        energy prediction in eV.
    length_to_A
        The conversion factor for length, such that the
        ``model.predict_forces(graph) * (energy_to_eV / length_to_A)``
        gives the force prediction in eV/Å.
    """

    def __init__(
        self, model: GraphPESModel, energy_to_eV: float, length_to_A: float
    ):
        super().__init__(
            cutoff=model.cutoff.item(),
            implemented_properties=model.implemented_properties,
        )
        self._model = model
        self._energy_to_eV = energy_to_eV
        self._length_to_A = length_to_A

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        predictions = self._model(graph)
        for key in predictions:
            if key in ["energy", "virial"]:
                predictions[key] *= self._energy_to_eV
            elif key == "forces":
                predictions[key] *= self._energy_to_eV / self._length_to_A
            elif key == "stress":
                predictions[key] *= self._energy_to_eV / self._length_to_A**3

        return predictions
