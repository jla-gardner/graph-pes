from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    module="e3nn",
    message=".*you are using `torch.load`.*",
)

import pathlib

import torch

from graph_pes.core import ConservativePESModel

from .addition import AdditionModel
from .e3nn.mace import MACE, ZEmbeddingMACE
from .e3nn.nequip import NequIP, ZEmbeddingNequIP
from .offsets import FixedOffset, LearnableOffset
from .painn import PaiNN
from .pairwise import LennardJones, LennardJonesMixture, Morse
from .schnet import SchNet
from .tensornet import TensorNet

__all__ = [
    "AdditionModel",
    "PaiNN",
    "LennardJones",
    "SchNet",
    "TensorNet",
    "Morse",
    "LennardJonesMixture",
    "NequIP",
    "ZEmbeddingNequIP",
    "MACE",
    "ZEmbeddingMACE",
    "FixedOffset",
    "LearnableOffset",
    "load_model",
]

# TODO: nicer way to do this?
ALL_MODELS: list[type[ConservativePESModel]] = [
    globals()[model]
    for model in __all__
    if model
    not in ["FixedOffset", "LearnableOffset", "load_model", "AdditionModel"]
]


def load_model(
    path: str | pathlib.Path,
) -> ConservativePESModel:
    """
    Load a model from a file.

    Parameters
    ----------
    path
        The path to the file.

    Returns
    -------
    ConservativePESModel
        The model.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")

    return torch.load(path)


def load_model_component(
    path: str | pathlib.Path, key: str
) -> ConservativePESModel:
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
    ConservativePESModel
        The component.
    """

    base_model = load_model(path)
    if not isinstance(base_model, AdditionModel):
        raise ValueError(
            f"Expected to load an AdditionModel, got {type(base_model)}"
        )

    return base_model[key]
