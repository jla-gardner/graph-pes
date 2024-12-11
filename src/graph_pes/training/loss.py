from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple, Sequence

import torch
from torch import Tensor, nn

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, divide_per_atom
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import force_to_single_line, uniform_repr
from graph_pes.utils.nn import UniformModuleList

Metric = Callable[[Tensor, Tensor], Tensor]
MetricName = Literal["MAE", "RMSE", "MSE"]


class Loss(nn.Module, ABC):
    """
    A general base class for all loss functions in ``graph-pes``.

    Implementations **must** override:

    * :meth:`forward` to compute the loss value.
    * :meth:`name` to return the name of the loss function.
    * :meth:`required_properties` to return the properties that this loss
      function needs to have available in order to compute its value.

    Additionally, implementations can optionally override:

    * :meth:`pre_fit` to perform any necessary operations before training
      commences.
    """

    @abstractmethod
    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor:
        r"""
        :class:`Loss`\ s can act on any of:

        Parameters
        ----------
        model
            The model being trained.
        graph
            The graph (usually a batch) the ``model`` was applied to.
        predictions
            The predictions from the ``model`` for the given ``graph``.
        """

    @property
    @abstractmethod
    def required_properties(self) -> list[PropertyKey]:
        """The properties that are required by this loss function."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this loss function, for logging purposes."""

    def pre_fit(self, training_data: AtomicGraph):
        """
        Perform any necessary operations before training commences.

        For example, this could be used to pre-compute a standard deviation
        of some property in the training data, which could then be used in
        :meth:`forward`.

        Parameters
        ----------
        training_data
            The training data to pre-fit this loss function to.
        """

    # add type hints to play nicely with mypy
    def __call__(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor:
        return super().__call__(model, graph, predictions)


class PropertyLoss(Loss):
    r"""
    A :class:`PropertyLoss` instance applies its :class:`Metric` to compare a
    model's predictions to the true values for a given property of a
    :class:`~graph_pes.AtomicGraph`.

    Parameters
    ----------
    property
        The property to apply the loss metric to.
    metric
        The loss metric to use. Defaults to :class:`RMSE`.

    Examples
    --------

    .. code-block:: python

        energy_rmse_loss = Loss("energy", RMSE())
        energy_rmse_value = energy_rmse_loss(
            predictions,  # a dict of key (energy/force/etc.) to value
            graph.properties,
        )
    """

    def __init__(
        self,
        property: PropertyKey,
        metric: Metric | MetricName = "RMSE",
    ):
        super().__init__()
        self.property: PropertyKey = property
        self.metric = parse_metric(metric)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        predictions
            The predictions from the ``model`` for the given ``graph``.
        """

        return self.metric(
            predictions[self.property],
            graph.properties[self.property],
        )

    @property
    def name(self) -> str:
        """Get the name of this loss for logging purposes."""
        return f"{self.property}_{_get_metric_name(self.metric)}"

    @property
    def required_properties(self) -> list[PropertyKey]:
        return [self.property]

    def __repr__(self) -> str:
        return uniform_repr(
            self.__class__.__name__,
            self.property,
            metric=self.metric,
        )


@dataclass
class WeightedLoss:
    """
    Specification for a component of a
    :class:`~graph_pes.training.loss.TotalLoss`.
    """

    component: Loss
    """Point to a :class:`~graph_pes.training.loss.Loss` instance."""

    weight: int | float = 1.0
    """The weight of this loss component."""


class SubLossPair(NamedTuple):
    loss_value: torch.Tensor
    weighted_loss_value: torch.Tensor


class TotalLossResult(NamedTuple):
    loss_value: torch.Tensor
    components: dict[str, SubLossPair]


class TotalLoss(torch.nn.Module):
    r"""
    A lightweight wrapper around a collection of (optionally weighted) losses.

    .. math::

        \mathcal{L}_{\text{total}} = \sum_i w_i \mathcal{L}_i

    where :math:`\mathcal{L}_i` is the :math:`i`-th loss and :math:`w_i` is the
    corresponding weight.

    ``graph-pes`` models are trained by minimising a :class:`TotalLoss` value.

    Parameters
    ----------
    losses
        The collection of losses to aggregate.
    """

    def __init__(self, losses: Sequence[Loss | WeightedLoss]):
        super().__init__()
        _losses = [
            loss if isinstance(loss, Loss) else loss.component
            for loss in losses
        ]
        _weights = [
            loss.weight if isinstance(loss, WeightedLoss) else 1.0
            for loss in losses
        ]
        self.losses = UniformModuleList(_losses)
        self.weights = _weights

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> TotalLossResult:
        """
        Computes the total loss value.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graph
            The graph (usually a batch) the ``model`` was applied to.
        """

        total_loss = torch.scalar_tensor(0.0, device=graph.Z.device)
        components: dict[str, SubLossPair] = {}

        for loss, weight in zip(self.losses, self.weights):
            loss_value = loss(model, graph, predictions)
            weighted_loss_value = loss_value * weight

            total_loss += weighted_loss_value
            components[loss.name] = SubLossPair(loss_value, weighted_loss_value)

        return TotalLossResult(total_loss, components)

    # add type hints to appease mypy
    def __call__(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> TotalLossResult:
        return super().__call__(model, graph, predictions)

    def __repr__(self) -> str:
        losses = ["(loss)"] + [
            force_to_single_line(str(loss)) for loss in self.losses
        ]
        weights = ["(weight)"] + [str(weight) for weight in self.weights]

        max_width = max(len(w) for w in weights)
        return "\n".join(
            ["TotalLoss:"]
            + [f"    {w:>{max_width}} : {l}" for w, l in zip(weights, losses)]
        )


############################# CUSTOM LOSSES #############################


class PerAtomEnergyLoss(PropertyLoss):
    r"""
    A loss function that evaluates some metric on the total energy normalised
    by the number of atoms in the structure.

    .. math::
        \mathcal{L} = \text{metric}\left(
        \bigoplus_i \frac{\hat{E}_i}{N_i}, \bigoplus_i\frac{E_i}{N_i} \right)

    where :math:`\hat{E}_i` is the predicted energy for structure :math:`i`,
    :math:`E_i` is the true energy for structure :math:`i`, :math:`N_i`
    is the number of atoms in structure :math:`i` and :math:`\bigoplus_i`
    denotes the cocatenation over all structures in the batch.

    Parameters
    ----------
    metric
        The loss metric to use. Defaults to :class:`RMSE`.
    """

    def __init__(
        self,
        metric: Metric | MetricName = "RMSE",
    ):
        super().__init__("energy", metric)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor:
        return self.metric(
            divide_per_atom(predictions["energy"], graph),
            divide_per_atom(graph.properties["energy"], graph),
        )

    @property
    def name(self) -> str:
        return f"per_atom_energy_{_get_metric_name(self.metric)}"


## METRICS ##


def parse_metric(metric: Metric | MetricName | None) -> Metric:
    if isinstance(metric, str):
        return {
            "MAE": MAE(),
            "RMSE": RMSE(),
            "MSE": MSE(),
        }[metric]

    if metric is None:
        return RMSE()

    return metric


class MSE(torch.nn.MSELoss):
    r"""
    Mean squared error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2
    """


class RMSE(torch.nn.MSELoss):
    r"""
    Root mean squared error metric:

    .. math::
        \sqrt{ \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2 }
    """

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (super().forward(input, target)).sqrt()


class MAE(torch.nn.L1Loss):
    r"""
    Mean absolute error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left| \hat{P}_i - P_i \right|
    """


def _get_metric_name(metric: Metric) -> str:
    # if metric is a function, we want the function's name, otherwise
    # we want the metric's class name, all lowercased
    # and without the word "loss" in it

    return (
        getattr(
            metric,
            "__name__",
            metric.__class__.__name__,
        )
        .lower()
        .replace("loss", "")
    )
