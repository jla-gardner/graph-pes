from __future__ import annotations

import re
import warnings
from typing import TypeVar

warnings.filterwarnings(
    "ignore",
    module="e3nn",
    message=".*you are using `torch.load`.*",
)

import pathlib

import torch

from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.logger import logger

from .addition import AdditionModel
from .e3nn.mace import MACE, ZEmbeddingMACE
from .e3nn.nequip import NequIP, ZEmbeddingNequIP
from .eddp import EDDP
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
from .scripted import ScriptedModel
from .stillinger_weber import StillingerWeber
from .tensornet import TensorNet
from .unit_converter import UnitConverter

__all__ = [
    "AdditionModel",
    "EDDP",
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
    "StillingerWeber",
    "UnitConverter",
]

MODEL_EXCLUSIONS = {
    "FixedOffset",
    "LearnableOffset",
    "AdditionModel",
    "PairPotential",
    "SmoothedPairPotential",
    "UnitConverter",
}

ALL_MODELS: list[type[GraphPESModel]] = [
    globals()[model] for model in __all__ if model not in MODEL_EXCLUSIONS
]


def load_model(path: str | pathlib.Path) -> GraphPESModel:
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

    Examples
    --------

    Use this function to load an existing model for further training using
    ``graph-pes-train``:

    .. code-block:: yaml

        model:
            +load_model:
                path: path/to/model.pt

    See :doc:`fine-tuning <../quickstart/fine-tuning>` for more details.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")

    model = torch.load(path, weights_only=False)

    if isinstance(model, torch.jit.ScriptModule):
        model = ScriptedModel(model)

    if not isinstance(model, GraphPESModel):
        raise ValueError(
            "Expected the loaded object to be a GraphPESModel "
            f"but got {type(model)}"
        )

    import graph_pes

    if model._GRAPH_PES_VERSION != graph_pes.__version__:
        warnings.warn(
            "You are attempting to load a model that was trained with "
            f"a different version of graph-pes ({model._GRAPH_PES_VERSION}) "
            f"than what you are currently using ({graph_pes.__version__}). "
            "We won't stop you from doing this, but it may cause issues.",
            stacklevel=2,
        )

    return model


def load_model_component(
    path: str | pathlib.Path,
    key: str,
) -> GraphPESModel:
    """
    Load a component from an :class:`~graph_pes.models.AdditionModel`.

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


T = TypeVar("T", bound=torch.nn.Module)


def freeze(model: T) -> T:
    """
    Freeze all parameters in a module.

    Parameters
    ----------
    model
        The model to freeze.

    Returns
    -------
    T
        The model.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def freeze_matching(model: T, pattern: str) -> T:
    r"""
    Freeze all parameters that match the given pattern.

    Parameters
    ----------
    model
        The model to freeze.
    pattern
        The regular expression to match the names of the parameters to freeze.

    Returns
    -------
    T
        The model.

    Examples
    --------

    Freeze all the parameters in the first layer of a MACE-MP0 model from
    :func:`~graph_pes.interfaces.mace_mp` (which have names of the form
    ``"model.interactions.0.<name>"``):

    .. code-block:: yaml

        model:
            +freeze_any_matching:
                model: +mace_mp()
                pattern: model\.interactions\.0\..*
    """
    for name, param in model.named_parameters():
        if re.match(pattern, name):
            logger.info(f"Freezing {name}")
            param.requires_grad = False
    return model


def freeze_any_matching(model: T, patterns: list[str]) -> T:
    r"""
    Freeze all parameters that match any of the given patterns.

    Parameters
    ----------
    model
        The model to freeze.
    patterns
        The patterns to match.

    Returns
    -------
    T
        The model.
    """
    for pattern in patterns:
        freeze_matching(model, pattern)
    return model


def freeze_all_except(model: T, pattern: str | list[str]) -> T:
    r"""
    Freeze all parameters in a model except those matching a given pattern.

    Parameters
    ----------
    model
        The model to freeze.
    pattern
        The pattern/s to match.

    Returns
    -------
    T
        The model.

    Examples
    --------

    Freeze all parameters in a MACE-MP0 model from
    :func:`~graph_pes.interfaces.mace_mp` except those in the
    read-out heads:

    .. code-block:: yaml

        model:
            +freeze_all_except:
                model: +mace_mp()
                pattern: model\.readouts.*
    """
    freeze(model)

    if isinstance(pattern, str):
        pattern = [pattern]
    for name, param in model.named_parameters():
        if any(re.match(p, name) for p in pattern):
            param.requires_grad = True

    return model
