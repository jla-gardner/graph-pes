from __future__ import annotations

from pathlib import Path

import e3nn
import torch

from graph_pes.core import GraphPESModel
from graph_pes.graphs import keys
from graph_pes.graphs.graph_typing import AtomicGraph


def differentiate(y: torch.Tensor, x: torch.Tensor):
    grad = torch.autograd.grad(
        outputs=[y.sum()],
        inputs=[x],
        create_graph=True,
        allow_unused=True,
    )[0]

    default = torch.zeros_like(x)
    default.requires_grad_(True)
    return grad if grad is not None else default


class LAMMPSModel(torch.nn.Module):
    def __init__(self, model: GraphPESModel, cutoff: float = 3.7):
        super().__init__()
        self.model = model
        self.cutoff = cutoff

    @torch.jit.export
    def get_cutoff(self) -> torch.Tensor:
        return torch.tensor(self.cutoff)

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        debug = graph.get("debug", torch.tensor(False)).item()

        if debug:
            print("Received graph:")
            for key, value in graph.items():
                print(f"{key}: {value}")

        compute_virial = graph["compute_virial"].item()  # type: ignore
        if compute_virial:
            # The virial stress tensor is the gradient of the total energy wrt
            # an infinitesimal change in the cell parameters.
            # We therefore add this change to the cell, such that
            # we can calculate the gradient wrt later if required.
            #
            # See <> TODO: find reference
            actual_cell = graph[keys.CELL]
            change_to_cell = torch.zeros_like(actual_cell)
            symmetric_change = 0.5 * (
                change_to_cell + change_to_cell.transpose(-1, -2)
            )
            graph[keys.CELL] = actual_cell + symmetric_change
        else:
            change_to_cell = torch.zeros_like(graph[keys.CELL])

        props: dict[str, torch.Tensor] = {}

        graph[keys._POSITIONS].requires_grad_(True)
        change_to_cell.requires_grad_(True)

        local_energies = self.model.predict_local_energies(graph).squeeze()
        props["local_energies"] = local_energies
        total_energy = torch.sum(local_energies)
        props["total_energy"] = total_energy
        dE_dR = differentiate(total_energy, graph[keys._POSITIONS])
        props["forces"] = -dE_dR
        if compute_virial:
            virial = differentiate(total_energy, change_to_cell)
            props["virial"] = virial

        graph[keys._POSITIONS].requires_grad_(False)

        # cast to float64
        for key in props:
            props[key] = props[key].double()
        return props


def deploy_model(model: GraphPESModel, cutoff: float, path: str | Path):
    lammps_model = LAMMPSModel(model, cutoff)
    scripted_model = e3nn.util.jit.script(lammps_model)
    torch.jit.save(scripted_model, path)
