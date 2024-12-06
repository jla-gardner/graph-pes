from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple, Sequence

import torch
from graph_pes.atomic_graph import AtomicGraph, PropertyKey, divide_per_atom
from graph_pes.utils.misc import force_to_single_line, uniform_repr
from graph_pes.utils.nn import UniformModuleList
from torch import Tensor, nn

Metric = Callable[[Tensor, Tensor], Tensor]
MetricName = Literal["MAE", "RMSE", "MSE"]


class Loss(nn.Module):
    r"""
    A :class:`Loss` instance applies its :class:`Metric` to the predictions and
    labels for a given property in a :class:`~graph_pes.AtomicGraph`.

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
        predictions: dict[PropertyKey, torch.Tensor],
        graphs: AtomicGraph,
    ) -> torch.Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graphs
            The graphs containing the labels.
        """

        return self.metric(
            predictions[self.property],
            graphs.properties[self.property],
        )

    @property
    def name(self) -> str:
        """Get the name of this loss for logging purposes."""
        return f"{self.property}_{_get_metric_name(self.metric)}"

    # add type hints to play nicely with mypy
    def __call__(
        self,
        predictions: dict[PropertyKey, torch.Tensor],
        graphs: AtomicGraph,
    ) -> torch.Tensor:
        return super().__call__(predictions, graphs)

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
        predictions: dict[PropertyKey, torch.Tensor],
        graphs: AtomicGraph,
    ) -> TotalLossResult:
        """
        Computes the total loss value.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graphs
            The graphs containing the labels.
        """

        total_loss = torch.scalar_tensor(0.0, device=graphs.Z.device)
        components: dict[str, SubLossPair] = {}

        for loss, weight in zip(self.losses, self.weights):
            loss_value = loss(predictions, graphs)
            weighted_loss_value = loss_value * weight

            total_loss += weighted_loss_value
            components[loss.name] = SubLossPair(loss_value, weighted_loss_value)

        return TotalLossResult(total_loss, components)

    # add type hints to appease mypy
    def __call__(
        self,
        predictions: dict[PropertyKey, torch.Tensor],
        graphs: AtomicGraph,
    ) -> TotalLossResult:
        return super().__call__(predictions, graphs)

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


class PerAtomEnergyLoss(Loss):
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
        predictions: dict[PropertyKey, torch.Tensor],
        graphs: AtomicGraph,
    ) -> torch.Tensor:
        return self.metric(
            divide_per_atom(predictions["energy"], graphs),
            divide_per_atom(graphs.properties["energy"], graphs),
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
