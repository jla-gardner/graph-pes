from __future__ import annotations

from itertools import chain
from typing import Callable, Literal

import torch

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import PropertyKey, is_batch, to_batch
from graph_pes.utils.misc import MAX_Z

MACE_KEY_MAPPING: dict[str, PropertyKey] = {
    "node_energy": "local_energies",
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "virials": "virial",
}


class ZToOneHot(torch.nn.Module):
    def __init__(self, elements: list[int]):
        super().__init__()
        self.register_buffer("z_to_index", torch.full((MAX_Z + 1,), -1))
        for i, z in enumerate(elements):
            self.z_to_index[z] = i
        self.num_classes = len(elements)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        indices = self.z_to_index[Z]
        return torch.nn.functional.one_hot(indices, self.num_classes)


def _atomic_graph_to_mace_input(
    graph: AtomicGraph,
    z_to_one_hot: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not is_batch(graph):
        graph = to_batch([graph])

    assert graph.batch is not None
    assert graph.ptr is not None

    _cell_per_edge = graph.cell[
        graph.batch[graph.neighbour_list[0]]
    ]  # (E, 3, 3)
    _shifts = torch.einsum(
        "kl,klm->km", graph.neighbour_cell_offsets, _cell_per_edge
    )  # (E, 3)
    data = {
        "node_attrs": z_to_one_hot(graph.Z).to(torch.get_default_dtype()),
        "positions": graph.R,
        "cell": graph.cell,
        "edge_index": graph.neighbour_list,
        "unit_shifts": graph.neighbour_cell_offsets,
        "shifts": _shifts,
        "batch": graph.batch,
        "ptr": graph.ptr,
    }
    return {k: v.to(graph.Z.device) for k, v in data.items()}


class MACEWrapper(GraphPESModel):
    """
    Converts any MACE model from the `mace-torch <https://github.com/ACEsuit/mace-torch>`__
    package into a :class:`~graph_pes.GraphPESModel`.

    You can use this to drive MD using LAMMPS, fine-tune MACE models,
    or any functionality that ``graph-pes`` provides.

    Parameters
    ----------
    model
        The MACE model to wrap.

    Examples
    --------
    >>> mace_torch_model = ...  # create your MACE model any-which way
    >>> from graph_pes.interfaces._mace import MACEWrapper
    >>> graph_pes_model = MACEWrapper(mace_torch_model)  # convert to graph-pes
    >>> graph_pes_model.predict_energy(graph)
    torch.Tensor([123.456])
    >>> from graph_pes.utils.calculator import GraphPESCalculator
    >>> calculator = GraphPESCalculator(graph_pes_model)
    >>> calculator.calculate(ase_atoms)

    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(
            model.r_max.item(),
            implemented_properties=[
                "local_energies",
                "energy",
                "forces",
                "stress",
                "virial",
            ],
        )
        self.model = model
        self.z_to_one_hot = ZToOneHot(self.model.atomic_numbers.tolist())

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        return self.predict(
            graph, ["local_energies", "energy", "forces", "stress", "virial"]
        )

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[PropertyKey],
    ) -> dict[PropertyKey, torch.Tensor]:
        raw_predictions = self.model.forward(
            _atomic_graph_to_mace_input(graph, self.z_to_one_hot),
            training=self.training,
            compute_force="forces" in properties,
            compute_stress="stress" in properties,
            compute_virials="virial" in properties,
        )

        predictions: dict[PropertyKey, torch.Tensor] = {
            MACE_KEY_MAPPING[key]: value
            for key, value in raw_predictions.items()
            if key in MACE_KEY_MAPPING
        }
        if not is_batch(graph):
            for p in ["energy", "stress", "virial"]:
                if p in properties:
                    predictions[p] = predictions[p].squeeze()
        return {k: v for k, v in predictions.items() if k in properties}


def _fix_precision(model: torch.nn.Module, precision: str) -> None:
    dtype = {"float32": torch.float32, "float64": torch.float64}[precision]
    for tensor in chain(
        model.parameters(),
        model.buffers(),
    ):
        if tensor.dtype.is_floating_point:
            tensor.data = tensor.data.to(dtype)


def mace_mp(
    model: Literal["small", "medium", "large"],
    precision: Literal["float32", "float64"] = "float64",
) -> MACEWrapper:
    """
    Donwload a MACE-MP model and convert it for use with ``graph-pes``.

    Internally, we use the `foundation_models <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>`__
    functionality from the `mace-torch <https://github.com/ACEsuit/mace-torch>`__ package.

    Please cite the following if you use this model:

    - MACE-MP by Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena,
      Dávid P. Kovács, Janosh Riebesell, et al., 2023, arXiv:2401.00096
    - MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b,
      DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal
    - Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Philipp Benner,
      Yuan Chiang, Alpha A Lee, Anubhav Jain, Kristin A Persson, 2023,
      arXiv:2308.14920

    Parameters
    ----------
    model
        The size of the MACE-MP model to download.
    precision
        The precision of the model.
    """  # noqa: E501
    from mace.calculators.foundations_models import mace_mp

    mace_torch_model = mace_mp(
        model,
        device="cpu",
        default_dtype=precision,
        return_raw_model=True,
    )
    assert isinstance(mace_torch_model, torch.nn.Module)
    _fix_precision(mace_torch_model, precision)
    return MACEWrapper(mace_torch_model)


def mace_off(
    model: Literal["small", "medium", "large"],
    precision: Literal["float32", "float64"] = "float64",
) -> MACEWrapper:
    """
    Download a MACE-OFF model and convert it for use with ``graph-pes``.

    If you use this model, please cite the relevant paper by Kovacs et.al., arXiv:2312.15211

    Parameters
    ----------
    model
        The size of the MACE-OFF model to download.
    precision
        The precision of the model.
    """  # noqa: E501
    from mace.calculators.foundations_models import mace_off

    mace_torch_model = mace_off(
        model,
        device="cpu",
        default_dtype=precision,
        return_raw_model=True,
    )
    assert isinstance(mace_torch_model, torch.nn.Module)
    _fix_precision(mace_torch_model, precision)
    return MACEWrapper(mace_torch_model)
