from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    module="e3nn",
    message=".*you are using `torch.load`.*",
)

import pathlib

import torch

from graph_pes.core import GraphPESModel

from .addition import AdditionModel
from .e3nn.mace import MACE, ZEmbeddingMACE
from .e3nn.nequip import NequIP, ZEmbeddingNequIP
from .offsets import FixedOffset, LearnableOffset
from .painn import PaiNN
from .pairwise import (
    LennardJones,
    LennardJonesMixture,
    Morse,
    PairPotential,
    SmoothedPairPotential,
    ZBLCoreRepulsion,
)
from .schnet import SchNet
from .tensornet import TensorNet

# TODO get rid of this all business
__all__ = [
    "AdditionModel",
    "FixedOffset",
    "LearnableOffset",
    "LennardJones",
    "LennardJonesMixture",
    "MACE",
    "Morse",
    "NequIP",
    "PaiNN",
    "PairPotential",
    "SchNet",
    "SmoothedPairPotential",
    "TensorNet",
    "ZBLCoreRepulsion",
    "ZEmbeddingMACE",
    "ZEmbeddingNequIP",
]

# TODO: nicer way to do this?
ALL_MODELS: list[type[GraphPESModel]] = [
    globals()[model]
    for model in __all__
    if model
    not in [
        "FixedOffset",
        "LearnableOffset",
        "AdditionModel",
        "PairPotential",
        "SmoothedPairPotential",
    ]  # TODO: remove offsets?
]


def load_model(
    path: str | pathlib.Path,
) -> GraphPESModel:
    """
    Load a model from a file.

    Parameters
    ----------
    path
        The path to the file.

    Returns
    -------
    GraphPESModel
        The model.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")

    return torch.load(path)


def load_model_component(path: str | pathlib.Path, key: str) -> GraphPESModel:
    """
    Load a component from an :class:`~graph_pes.core.AdditionModel`.

    Parameters
    ----------
    path
        The path to the file.
    key
        The key to load.

    Returns
    -------
    GraphPESModel
        The component.
    """

    base_model = load_model(path)
    if not isinstance(base_model, AdditionModel):
        raise ValueError(
            f"Expected to load an AdditionModel, got {type(base_model)}"
        )

    return base_model[key]
