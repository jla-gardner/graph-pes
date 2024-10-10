from __future__ import annotations

from pathlib import Path

import e3nn.util.jit
import torch

from graph_pes.core import ConservativePESModel
from graph_pes.graphs import keys
from graph_pes.graphs.graph_typing import AtomicGraph


class LAMMPSModel(torch.nn.Module):
    def __init__(self, model: ConservativePESModel):
        super().__init__()
        self.model = model

    @torch.jit.export
    def get_cutoff(self) -> torch.Tensor:
        return self.model.cutoff

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        debug = graph.get("debug", torch.tensor(False)).item()

        if debug:
            print("Received graph:")
            for key, value in graph.items():
                print(f"{key}: {value}")

        compute_virial = graph["compute_virial"].item()  # type: ignore
        properties = [keys.ENERGY, keys.FORCES]
        if compute_virial:
            properties.append(keys.STRESS)

        preds: dict[str, torch.Tensor] = self.model._get_predictions(
            graph,
            properties=properties,  # type: ignore
            training=False,
            get_local_energies=True,
        )

        # cast to float64
        for key in preds:
            preds[key] = preds[key].double()

        # add virial output if required
        if compute_virial:
            # LAMMPS expects the **virial** in Voigt notation
            # we provide the **stress** in full 3x3 matrix notation
            # therefore, convert:
            volume = graph[keys.CELL].det().abs().item()

            virial_tensor = -preds[keys.STRESS] * volume
            virial_voigt = torch.zeros(
                6,
                device=preds[keys.STRESS].device,
                dtype=preds[keys.STRESS].dtype,
            )
            virial_voigt[0] = virial_tensor[0, 0]
            virial_voigt[1] = virial_tensor[1, 1]
            virial_voigt[2] = virial_tensor[2, 2]
            virial_voigt[3] = virial_tensor[0, 1]
            virial_voigt[4] = virial_tensor[0, 2]
            preds["virial"] = virial_voigt

        return preds

    def __call__(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        return super().__call__(graph)


def deploy_model(model: ConservativePESModel, path: str | Path):
    """
    Deploy a :class:`~graph_pes.core.ConservativePESModel` for use with LAMMPS.

    Use the resulting model with LAMMPS according to:

    .. code-block:: bash

        pair_style graph_pes <cpu/cuda>
        pair_coeff * * path/to/model.pt

    Parameters
    ----------
    model
        The model to deploy.
    path
        The path to save the deployed model to.
    """
    lammps_model = LAMMPSModel(model)
    scripted_model = e3nn.util.jit.script(lammps_model)
    torch.jit.save(scripted_model, path)
