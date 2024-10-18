from __future__ import annotations

from pathlib import Path
from typing import Callable

import ase.build
import pytest
import pytorch_lightning
import torch
from ase import Atoms
from ase.io import read
from graph_pes.core import GraphPESModel
from graph_pes.data.io import to_atomic_graph
from graph_pes.graphs import keys
from graph_pes.graphs.graph_typing import AtomicGraph
from graph_pes.models import (
    ALL_MODELS,
    MACE,
    AdditionModel,
    FixedOffset,
    LennardJones,
    NequIP,
    ZEmbeddingMACE,
    ZEmbeddingNequIP,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler


def all_model_factories(
    expected_elements: list[str],
) -> tuple[list[str], list[Callable[[], GraphPESModel]]]:
    pytorch_lightning.seed_everything(42)
    # make these models as small as possible to speed up tests
    required_kwargs = {
        NequIP: {"elements": expected_elements, "n_layers": 1, "l_max": 1},
        ZEmbeddingNequIP: {"n_layers": 1, "l_max": 1},
        MACE: {
            "elements": expected_elements,
            "layers": 1,
            "max_ell": 1,
            "correlation": 1,
            "hidden_irreps": "4x0e + 4x1o",
        },
        ZEmbeddingMACE: {
            "layers": 1,
            "max_ell": 1,
            "correlation": 1,
            "hidden_irreps": "4x0e + 4x1o",
        },
    }

    def _model_factory(
        model_klass: type[GraphPESModel],
    ) -> Callable[[], GraphPESModel]:
        return lambda: model_klass(**required_kwargs.get(model_klass, {}))

    names = [model.__name__ for model in ALL_MODELS]
    factories = [_model_factory(model) for model in ALL_MODELS]
    names.append("AdditionModel")
    factories.append(
        lambda: AdditionModel(lj=LennardJones(), offset=FixedOffset())
    )
    return names, factories


def all_models(
    expected_elements: list[str],
) -> tuple[list[str], list[GraphPESModel]]:
    names, factories = all_model_factories(expected_elements)
    return names, [factory() for factory in factories]


def parameterise_all_models(
    expected_elements: list[str],
):
    def decorator(func):
        names, models = all_models(expected_elements)
        return pytest.mark.parametrize("model", models, ids=names)(func)

    return decorator


def parameterise_model_classes(
    expected_elements: list[str],
):
    def decorator(func):
        names, factories = all_model_factories(expected_elements)
        return pytest.mark.parametrize("model_class", factories, ids=names)(
            func
        )

    return decorator


def graph_from_molecule(molecule: str, cutoff: float = 3.7) -> AtomicGraph:
    return to_atomic_graph(ase.build.molecule(molecule), cutoff)


CU_STRUCTURES_FILE = Path(__file__).parent / "test.xyz"
CU_TEST_STRUCTURES: list[Atoms] = read(CU_STRUCTURES_FILE, ":")  # type: ignore

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


class DoesNothingModel(GraphPESModel):
    def __init__(self):
        super().__init__(
            cutoff=3.7,
            implemented_properties=["local_energies"],
        )
        self.scaler = LocalEnergiesScaler()

    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        local_energies = torch.zeros(len(graph["atomic_numbers"]))
        local_energies = self.scaler(local_energies, graph)
        return {"local_energies": local_energies}
