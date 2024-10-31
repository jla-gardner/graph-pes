from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    module="e3nn",
    message=".*you are using `torch.load`.*",
)

import pathlib

import torch

from graph_pes.graph_pes_model import GraphPESModel

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
    ]
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
            graph_pes.models.load_model:
                path: path/to/model.pt

    To account for some new energy offset in your training data, you could do
    something like this:
    (see also :func:`~graph_pes.models.load_model_component`)

    .. code-block:: yaml

        model:
            # add an offset to an existing model before fine-tuning
            offset:
                graph_pes.models.LearnableOffset: {}
            many-body:
                graph_pes.models.load_model:
                    path: path/to/model.pt
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")

    model = torch.load(path, weights_only=False)

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


def load_model_component(path: str | pathlib.Path, key: str) -> GraphPESModel:
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

    Examples
    --------

    Train on data with a new energy offset:

    .. code-block:: yaml

        model:
            offset:
                graph_pes.models.LearnableOffset: {}
            many-body:
                graph_pes.models.load_model_component:
                    path: path/to/model.pt
                    key: many-body
    """

    base_model = load_model(path)
    if not isinstance(base_model, AdditionModel):
        raise ValueError(
            f"Expected to load an AdditionModel, got {type(base_model)}"
        )

    return base_model[key]
