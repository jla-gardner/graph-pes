from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Callable

import ase.build
import pytest
import pytorch_lightning
import torch
from ase import Atoms
from ase.io import read
from locache import reset

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.data.datasets import get_all_graphs_and_cache_to_disk
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.models import (
    ALL_MODELS,
    EDDP,
    MACE,
    AdditionModel,
    FixedOffset,
    LennardJones,
    NequIP,
    PaiNN,
    TensorNet,
    ZEmbeddingMACE,
    ZEmbeddingNequIP,
)
from graph_pes.models.components.scaling import LocalEnergiesScaler

# remove cache so that any changes are actually tested
reset(get_all_graphs_and_cache_to_disk)

# non-verbose load-atoms to avoid poluting the test output
os.environ["LOAD_ATOMS_VERBOSE"] = "0"


def all_model_factories(
    expected_elements: list[str],
    cutoff: float,
) -> tuple[list[str], list[Callable[[], GraphPESModel]]]:
    pytorch_lightning.seed_everything(42)
    # make these models as small as possible to speed up tests
    _small_nequip = {
        "layers": 2,
        "features": dict(
            channels=16,
            l_max=1,
            use_odd_parity=False,
        ),
    }
    required_kwargs = {
        NequIP: {"elements": expected_elements, **_small_nequip},
        ZEmbeddingNequIP: {**_small_nequip},
        MACE: {
            "elements": expected_elements,
            "layers": 3,
            "l_max": 2,
            "correlation": 3,
            "channels": 4,
        },
        ZEmbeddingMACE: {
            "layers": 3,
            "l_max": 2,
            "correlation": 3,
            "channels": 4,
            "z_embed_dim": 4,
        },
        PaiNN: {
            "layers": 2,
            "channels": 16,
        },
        TensorNet: {
            "layers": 2,
            "radial_features": 24,
            "channels": 8,
        },
        EDDP: {"elements": expected_elements},
    }

    def _model_factory(
        model_klass: type[GraphPESModel],
    ) -> Callable[[], GraphPESModel]:
        # inspect for if cutoff is a required argument
        requires_cutoff = False
        for arg in inspect.signature(model_klass.__init__).parameters.values():
            if arg.name == "cutoff":
                requires_cutoff = True
                break
        kwargs = required_kwargs.get(model_klass, {})
        if requires_cutoff:
            kwargs["cutoff"] = cutoff
        return lambda: model_klass(**kwargs)

    names = [model.__name__ for model in ALL_MODELS]
    factories = [_model_factory(model) for model in ALL_MODELS]
    names.append("AdditionModel")
    factories.append(
        lambda: AdditionModel(
            lj=LennardJones(cutoff=cutoff), offset=FixedOffset()
        )
    )
    return names, factories


def all_models(
    expected_elements: list[str],
    cutoff: float,
) -> tuple[list[str], list[GraphPESModel]]:
    torch.manual_seed(42)
    names, factories = all_model_factories(expected_elements, cutoff)
    return names, [factory() for factory in factories]


def parameterise_all_models(expected_elements: list[str], cutoff: float = 5.0):
    def decorator(func):
        names, models = all_models(expected_elements, cutoff)
        return pytest.mark.parametrize("model", models, ids=names)(func)

    return decorator


def parameterise_model_classes(
    expected_elements: list[str],
    cutoff: float = 5.0,
):
    def decorator(func):
        names, factories = all_model_factories(expected_elements, cutoff)
        return pytest.mark.parametrize("model_class", factories, ids=names)(
            func
        )

    return decorator


def graph_from_molecule(molecule: str, cutoff: float = 3.7) -> AtomicGraph:
    return AtomicGraph.from_ase(ase.build.molecule(molecule), cutoff)


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

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        local_energies = torch.zeros(len(graph.Z))
        local_energies = self.scaler(local_energies, graph)
        return {"local_energies": local_energies}
