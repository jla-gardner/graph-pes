from __future__ import annotations

from typing import Callable, Sequence, overload

import torch

from graph_pes.core import ConservativePESModel
from graph_pes.graphs import AtomicGraph, AtomicGraphBatch, keys


class FunctionalModel(ConservativePESModel):
    """
    Wrap a function that returns an energy prediction into a model
    that can be used in the same way as other
    :class:`~graph_pes.core.ConservativePESModel` subclasses.

    .. warning::

        This model does not support local energy predictions, and therefore
        cannot be used for LAMMPS simulations.

        Force and stress predictions are still supported however.

    Parameters
    ----------
    func
        The function to wrap.

    """

    def __init__(
        self,
        func: Callable[[AtomicGraph], torch.Tensor],
    ):
        super().__init__(auto_scale=False, cutoff=0)
        self.func = func

    def forward(self, graph: AtomicGraph) -> torch.Tensor:
        return self.func(graph)

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        raise Exception("local energies not implemented for functional models")


@overload
def get_predictions(
    model: Callable[[AtomicGraph], torch.Tensor],
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    properties: list[keys.LabelKey] | None = None,
) -> dict[keys.LabelKey, torch.Tensor]: ...


@overload
def get_predictions(
    model: Callable[[AtomicGraph], torch.Tensor],
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    property: keys.LabelKey,
) -> torch.Tensor: ...


def get_predictions(
    model: Callable[[AtomicGraph], torch.Tensor],
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    properties: list[keys.LabelKey] | None = None,
    property: keys.LabelKey | None = None,
) -> dict[keys.LabelKey, torch.Tensor] | torch.Tensor:
    """
    Evaluate the model on the given structure to get
    the properties requested.

    Parameters
    ----------
    model
        The model to evaluate. Can be any callable that takes an
        :class:`~graph_pes.graphs.AtomicGraph` and returns a scalar
        energy prediction.
    graph
        The atomic structure to evaluate.
    properties
        The properties to predict. If not provided, defaults to
        :code:`[Property.ENERGY, Property.FORCES]` if the structure
        has no cell, and :code:`[Property.ENERGY, Property.FORCES,
        Property.STRESS]` if it does.
    property
        The property to predict. Can't be used when :code:`properties`
        is also provided.
    training
        Whether the model is currently being trained. If :code:`False`,
        the gradients of the predictions will be detached.

    Examples
    --------
    >>> get_predictions(model, graph_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...), 'stress': tensor(...)}
    >>> get_predictions(model, graph_no_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...)}
    >>> get_predictions(model, graph, property="energy")
    tensor(-12.3)
    """
    if not isinstance(model, ConservativePESModel):
        model = FunctionalModel(model)
    return model.get_predictions(
        graph, properties=properties, property=property
    )  # type: ignore
