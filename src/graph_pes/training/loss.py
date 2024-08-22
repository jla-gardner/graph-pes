from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import torch
from graph_pes.graphs import LabelledBatch, keys
from graph_pes.nn import UniformModuleList
from graph_pes.transform import divide_per_atom
from graph_pes.util import force_to_single_line, uniform_repr
from torch import Tensor, nn


class Loss(nn.Module):
    r"""
    Measure the discrepancy between predictions and labels for a given property.

    You can create a weighted, multi-component loss by using the ``+`` and ``*``
    operators on :class:`Loss` instances. For example:

    .. code-block:: python

            Loss("energy") + Loss("forces") * 10

    will create a loss function that is the sum of an energy loss
    and a 10x-weighted force loss.

    Parameters
    ----------
    property_key
        The property to apply the loss metric to.
    metric
        The loss metric to use. Defaults to :class:`MAE`.
    """

    def __init__(
        self,
        property_key: keys.LabelKey,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ):
        super().__init__()
        self.property_key: keys.LabelKey = property_key
        self.metric = MAE() if metric is None else metric

    def forward(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graphs: LabelledBatch,
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
            predictions[self.property_key],
            graphs[self.property_key],
        )

    @property
    def name(self) -> str:
        """Get the name of this loss for logging purposes."""
        return f"{self.property_key}_{_get_metric_name(self.metric)}"

    # add type hints to play nicely with mypy
    def __call__(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graphs: LabelledBatch,
    ) -> torch.Tensor:
        return super().__call__(predictions, graphs)

    def __repr__(self) -> str:
        return uniform_repr(
            self.__class__.__name__,
            self.property_key,
            metric=self.metric,
        )


class SubLossPair(NamedTuple):
    loss_value: torch.Tensor
    weighted_loss_value: torch.Tensor


class TotalLossResult(NamedTuple):
    loss_value: torch.Tensor
    components: dict[str, SubLossPair]


class TotalLoss(torch.nn.Module):
    r"""
    A lightweight wrapper around a collection of weighted losses.

    Creation can be done in two ways:

    1. Directly, by passing a list of losses and weights.
    2. Via ``+`` and ``*`` operators

    Hence:

    .. code-block:: python

        WeightedLoss([Loss("energy"), Loss("forces")], weights=[10, 1])
        # is equivalent to
        10 * Loss("energy") + 1 * Loss("forces")

    Parameters
    ----------
    losses
        The collection of losses to aggregate.
    weights
        The weights to apply to each loss.
    """

    def __init__(
        self,
        losses: Sequence[Loss],
        weights: Sequence[float | int] | None = None,
    ):
        super().__init__()
        self.losses = UniformModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graphs: LabelledBatch,
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

        total_loss = torch.scalar_tensor(
            0.0, device=graphs["atomic_numbers"].device
        )
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
        predictions: dict[keys.LabelKey, torch.Tensor],
        graphs: LabelledBatch,
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
    A loss function that computes the per-atom energy loss:

    .. math::
        \mathcal{L} = \text{metric}\left(
        \sum_i \frac{\hat{E}_i}{N_i}, \sum_i \frac{E_i}{N_i} \right)

    where :math:`\hat{E}_i` is the predicted energy for structure :math:`i`,
    :math:`E_i` is the true energy for structure :math:`i`, and :math:`N_i`
    is the number of atoms in structure :math:`i`.

    Parameters
    ----------
    metric
        The loss metric to use. Defaults to :class:`MAE`.
    """

    def __init__(
        self,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ):
        super().__init__(keys.ENERGY, metric)

    def forward(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graphs: LabelledBatch,
    ) -> torch.Tensor:
        return self.metric(
            divide_per_atom(predictions[keys.ENERGY], graphs),
            divide_per_atom(graphs[keys.ENERGY], graphs),
        )

    @property
    def name(self) -> str:
        return f"per_atom_energy_{_get_metric_name(self.metric)}"


## METRICS ##


class RMSE(torch.nn.MSELoss):
    r"""
    Root mean squared error metric:

    .. math::
        \sqrt{ \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2 }
    """

    def __init__(self):
        super().__init__()

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

    def __init__(self):
        super().__init__()


def _get_metric_name(metric: Callable[[Tensor, Tensor], Tensor]) -> str:
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
