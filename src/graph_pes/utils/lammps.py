from __future__ import annotations

import pathlib

import e3nn.util.jit
import torch
from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.graph_pes_model import GraphPESModel


def as_lammps_data(
    graph: AtomicGraph,
    compute_virial: bool = False,
    debug: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "atomic_numbers": graph.Z,
        "positions": graph.R,
        "cell": graph.cell,
        "neighbour_list": graph.neighbour_list,
        "neighbour_cell_offsets": graph.neighbour_cell_offsets,
        "compute_virial": torch.tensor(compute_virial),
        "debug": torch.tensor(debug),
    }


class LAMMPSModel(torch.nn.Module):
    def __init__(self, model: GraphPESModel):
        super().__init__()
        self.model = model

    @torch.jit.export
    def get_cutoff(self) -> torch.Tensor:
        return self.model.cutoff

    def forward(
        self, graph_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        debug = graph_data.get("debug", torch.tensor(False)).item()

        if debug:
            print("Received graph:")
            for key, value in graph_data.items():
                print(f"{key}: {value}")

        compute_virial = graph_data["compute_virial"].item()
        properties: list[PropertyKey] = ["energy", "forces", "local_energies"]
        if compute_virial:
            properties.append("stress")

        # graph_data is a dict, so we need to convert it to an AtomicGraph
        graph = AtomicGraph(
            Z=graph_data["atomic_numbers"],
            R=graph_data["positions"],
            cell=graph_data["cell"],
            neighbour_list=graph_data["neighbour_list"],
            neighbour_cell_offsets=graph_data["neighbour_cell_offsets"],
            properties={},
            other={},
        )
        preds = self.model.predict(graph, properties=properties)

        # cast to float64
        for key in preds:
            preds[key] = preds[key].double()

        # add virial output if required
        if compute_virial:
            # LAMMPS expects the **virial** in Voigt notation
            # we provide the **stress** in full 3x3 matrix notation
            # therefore, convert:
            volume = graph.cell.det().abs().item()

            assert "stress" in preds
            virial_tensor = -preds["stress"] * volume
            virial_voigt = torch.zeros(
                6,
                device=preds["stress"].device,
                dtype=preds["stress"].dtype,
            )
            virial_voigt[0] = virial_tensor[0, 0]
            virial_voigt[1] = virial_tensor[1, 1]
            virial_voigt[2] = virial_tensor[2, 2]
            virial_voigt[3] = virial_tensor[0, 1]
            virial_voigt[4] = virial_tensor[0, 2]
            preds["virial"] = virial_voigt  # type: ignore

        return preds  # type: ignore

    def __call__(
        self, graph_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return super().__call__(graph_data)


def deploy_model(model: GraphPESModel, path: str | pathlib.Path):
    """
    Deploy a :class:`~graph_pes.GraphPESModel` for use with LAMMPS.

    Use the resulting model with LAMMPS according to:

    .. code-block:: bash

        pair_style graph_pes <cpu>
        pair_coeff * * path/to/model.pt <element-of-type-1> <element-of-type-2> ...

    Parameters
    ----------
    model
        The model to deploy.
    path
        The path to save the deployed model to.
    """  # noqa: E501
    lammps_model = LAMMPSModel(model)
    scripted_model = e3nn.util.jit.script(lammps_model)
    torch.jit.save(scripted_model, path)
