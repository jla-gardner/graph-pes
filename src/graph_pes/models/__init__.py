from __future__ import annotations

from pathlib import Path

import torch

from graph_pes.core import GraphPESModel

from .e3nn.nequip import NequIP, OneHotNequIP, ZEmbeddingNequIP
from .offsets import FixedOffset, LearnableOffset
from .painn import PaiNN
from .pairwise import LennardJones, LennardJonesMixture, Morse
from .schnet import SchNet
from .tensornet import TensorNet

__all__ = [
    "PaiNN",
    "LennardJones",
    "SchNet",
    "TensorNet",
    "Morse",
    "LennardJonesMixture",
    "NequIP",
    "OneHotNequIP",
    "ZEmbeddingNequIP",
    "FixedOffset",
    "LearnableOffset",
    "load_model",
]

# TODO: nicer way to do this?
ALL_MODELS: list[type[GraphPESModel]] = [
    globals()[model]
    for model in __all__
    if model not in ["FixedOffset", "LearnableOffset", "load_model"]
]

STR_ALL_MODELS: list[str] = [
    model
    for model in __all__
    if model not in ["FixedOffset", "LearnableOffset", "load_model"]
]


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")

    return torch.load(path)
