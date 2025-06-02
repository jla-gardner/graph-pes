from __future__ import annotations

import warnings
from contextlib import redirect_stdout
from itertools import chain
from pathlib import Path
from typing import Literal

import requests
import torch

from graph_pes import AtomicGraph
from graph_pes.atomic_graph import PropertyKey, is_batch
from graph_pes.interfaces.base import InterfaceModel
from graph_pes.utils.misc import MAX_Z


class ZToOneHot(torch.nn.Module):
    def __init__(self, elements: list[int]):
        super().__init__()
        self.z_to_index: torch.Tensor
        self.register_buffer("z_to_index", torch.full((MAX_Z + 1,), -1))
        for i, z in enumerate(elements):
            self.z_to_index[z] = i
        self.num_classes = len(elements)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        indices = self.z_to_index[Z]
        if (indices < 0).any():
            raise ValueError(
                "ZToOneHot received an atomic number that is not in the model's"
                f" element list: {Z[indices < 0]}. Please ensure the model was "
                "trained with all elements present in the input graph."
            )
        return torch.nn.functional.one_hot(indices, self.num_classes)


def _atomic_graph_to_mace_input(
    graph: AtomicGraph,
    z_to_one_hot: ZToOneHot,
) -> dict[str, torch.Tensor]:
    batch = graph.batch
    if batch is None:
        batch = torch.zeros_like(graph.Z)

    ptr = graph.ptr
    if ptr is None:
        ptr = torch.tensor([0, graph.Z.shape[0]])

    cell = graph.cell.unsqueeze(0) if not is_batch(graph) else graph.cell

    _cell_per_edge = cell[batch[graph.neighbour_list[0]]]  # (E, 3, 3)
    _shifts = torch.einsum(
        "kl,klm->km", graph.neighbour_cell_offsets, _cell_per_edge
    )  # (E, 3)
    data = {
        "node_attrs": z_to_one_hot.forward(graph.Z).to(graph.R.dtype),
        "positions": graph.R.clone().detach(),
        "cell": cell.clone().detach(),
        "edge_index": graph.neighbour_list,
        "unit_shifts": graph.neighbour_cell_offsets,
        "shifts": _shifts,
        "batch": batch,
        "ptr": ptr,
    }
    return {k: v.to(graph.Z.device) for k, v in data.items()}


class MACEWrapper(InterfaceModel):
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
            model.r_max.item(),  # type: ignore
            implemented_properties=[
                "local_energies",
                "energy",
                "forces",
                "stress",
                "virial",
            ],
        )
        self.model = model
        self.z_to_one_hot = ZToOneHot(self.model.atomic_numbers.tolist())  # type: ignore

    def convert_to_underlying_input(self, graph: AtomicGraph):
        return _atomic_graph_to_mace_input(graph, self.z_to_one_hot)

    def raw_forward_pass(
        self,
        input,
        is_batched: bool,
        properties: list[PropertyKey],
    ) -> dict[PropertyKey, torch.Tensor]:
        MACE_KEY_MAPPING: dict[str, PropertyKey] = {
            "node_energy": "local_energies",
            "energy": "energy",
            "forces": "forces",
            "stress": "stress",
            "virials": "virial",
        }

        raw_predictions = self.model.forward(
            input,
            training=self.training,
            compute_force="forces" in properties,
            compute_stress="stress" in properties,
            compute_virials="virial" in properties,
        )

        predictions: dict[PropertyKey, torch.Tensor] = {}
        for key, value in raw_predictions.items():
            if key in MACE_KEY_MAPPING:
                property_key = MACE_KEY_MAPPING[key]
                if property_key in properties and value is not None:
                    predictions[property_key] = value

        if not is_batched:
            for p in ["energy", "stress", "virial"]:
                if p in properties:
                    predictions[p] = predictions[p].squeeze()

        return predictions

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[PropertyKey],
    ) -> dict[PropertyKey, torch.Tensor]:
        # override this predict property to use the underlying
        # mace-torch mechanisms for calculating forces etc.
        input = self.convert_to_underlying_input(graph)
        is_batched = is_batch(graph)
        return self.raw_forward_pass(input, is_batched, properties)


def _fix_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for tensor in chain(
        model.parameters(),
        model.buffers(),
    ):
        if tensor.dtype.is_floating_point:
            tensor.data = tensor.data.to(dtype)


def _get_dtype(
    precision: Literal["float32", "float64"] | None,
) -> torch.dtype:
    if precision is None:
        return torch.get_default_dtype()
    return {"float32": torch.float32, "float64": torch.float64}[precision]


def mace_mp(
    model: str = "small",
    precision: Literal["float32", "float64"] | None = None,
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

    As of 10th April 2025, the following models are available: ``["small", "medium", "large", "medium-mpa-0", "small-0b", "medium-0b", "small-0b2", "medium-0b2", "medium-0b3", "large-0b2", "medium-omat-0"]``

    Parameters
    ----------
    model
        The size of the MACE-MP model to download.
    precision
        The precision of the model. If ``None``, the default precision
        of torch will be used (you can set this when using ``graph-pes-train``
        via ``general/torch/dtype``)
    """  # noqa: E501
    from mace.calculators.foundations_models import mace_mp as mace_mp_impl

    dtype = _get_dtype(precision)
    precision_str = {torch.float32: "float32", torch.float64: "float64"}[dtype]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        with redirect_stdout(None):
            mace_torch_model = mace_mp_impl(
                model,
                device="cpu",
                default_dtype=precision_str,
                return_raw_model=True,
            )

    assert isinstance(mace_torch_model, torch.nn.Module)
    _fix_dtype(mace_torch_model, dtype)
    return MACEWrapper(mace_torch_model)


def mace_off(
    model: Literal["small", "medium", "large"],
    precision: Literal["float32", "float64"] | None = None,
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

    dtype = _get_dtype(precision)
    precision_str = {torch.float32: "float32", torch.float64: "float64"}[dtype]

    mace_torch_model = mace_off(
        model,
        device="cpu",
        default_dtype=precision_str,
        return_raw_model=True,
    )
    assert isinstance(mace_torch_model, torch.nn.Module)
    _fix_dtype(mace_torch_model, dtype)
    # un freeze all parameters
    for p in mace_torch_model.parameters():
        p.requires_grad = True
    return MACEWrapper(mace_torch_model)


def go_mace_23(
    precision: Literal["float32", "float64"] | None = None,
) -> MACEWrapper:
    """
    Download the `GO-MACE-23 model <https://doi.org/10.1002/anie.202410088>`__
    and convert it for use with ``graph-pes``.

    .. note::

        This model is only for use on structures containing Carbon, Hydrogen and
        Oxygen. Attempting to use on structures with other elements will raise
        an error.

    If you use this model, please cite the following:

    .. code-block:: bibtex

        @article{El-Machachi-24,
            title = {Accelerated {{First-Principles Exploration}} of {{Structure}} and {{Reactivity}} in {{Graphene Oxide}}},
            author = {{El-Machachi}, Zakariya and Frantzov, Damyan and Nijamudheen, A. and Zarrouk, Tigany and Caro, Miguel A. and Deringer, Volker L.},
            year = {2024},
            journal = {Angewandte Chemie International Edition},
            volume = {63},
            number = {52},
            pages = {e202410088},
            doi = {10.1002/anie.202410088},
        }

    """  # noqa: E501

    return _mace_from_url(
        "https://github.com/zakmachachi/GO-MACE-23/raw/refs/heads/main/models/fitting/potential/iter-12-final-model/go-mace-23.pt",
        "GO-MACE-23",
        precision,
    )


def egret(
    model: Literal["egret-1", "egret-1t", "egret-1e"] = "egret-1",
) -> MACEWrapper:
    """
    Download an `Egret <https://arxiv.org/abs/2504.20955>`__ model
    and convert it for use with ``graph-pes``.

    Use the ``egret-1`` model via the Python API:

    .. code-block:: python

        from graph_pes.interfaces._mace import egret
        model = egret("egret-1")

    or :doc:`fine-tune <../quickstart/fine-tuning>` it on your own data using
    the :doc:`graph-pes-train <../cli/graph-pes-train/root>` command:

    .. code-block:: yaml

        model:
            +egret: {model: "egret-1e"}

        data:
            ...
        # etc.

    If you use this model, please cite the following:

    .. code-block:: bibtex

        @misc{Mann-25-04,
            title = {
                Egret-1: {{Pretrained Neural Network
                Potentials For Efficient}} and
                {{Accurate Bioorganic Simulation}}
            },
            author = {
                Mann, Elias L. and Wagen, Corin C.
                and Vandezande, Jonathon E. and Wagen, Arien M.
                and Schneider, Spencer C.
            },
            year = {2025},
            number = {arXiv:2504.20955},
            doi = {10.48550/arXiv.2504.20955},
        }

    As of 1st May 2025, the following models are available: ``["egret-1", "egret-1t", "egret-1e"]``
    Parameters
    ----------
    model
        The model to download.
    """  # noqa: E501

    urls = {
        "egret-1": "https://github.com/rowansci/egret-public/raw/b1b4c1261315b38f0dd3f6ec0fce891a9119ffe0/compiled_models/EGRET_1.model",
        "egret-1t": "https://github.com/rowansci/egret-public/raw/b1b4c1261315b38f0dd3f6ec0fce891a9119ffe0/compiled_models/EGRET_1T.model",
        "egret-1e": "https://github.com/rowansci/egret-public/raw/b1b4c1261315b38f0dd3f6ec0fce891a9119ffe0/compiled_models/EGRET_1E.model",
    }

    return _mace_from_url(urls[model], model)


def _mace_from_url(
    url: str,
    model_name: str,
    precision: Literal["float32", "float64"] | None = None,
) -> MACEWrapper:
    dtype = _get_dtype(precision)
    file_name = f"{model_name.replace(' ', '-')}.pt"

    save_path = Path.home() / ".graph-pes" / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not save_path.exists():
        print(f"Downloading {model_name} model to {save_path}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(save_path, "wb") as file:
            file.write(response.content)

    print(f"Loading {model_name} model from {save_path}")
    mace_torch_model = torch.load(
        save_path, weights_only=False, map_location=torch.device("cpu")
    )
    for p in mace_torch_model.parameters():
        p.data = p.data.to(dtype)
    model = MACEWrapper(mace_torch_model)

    return model
