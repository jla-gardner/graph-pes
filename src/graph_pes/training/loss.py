from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, NamedTuple, Sequence

import torch
import torchmetrics
from torch import Tensor, nn
from torchmetrics import Metric as TorchMetric

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, divide_per_atom
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import uniform_repr
from graph_pes.utils.nn import UniformModuleList

Metric = Callable[[Tensor, Tensor], Tensor]
MetricName = Literal["MAE", "RMSE", "MSE"]


class WeightedLoss:
    def __init__(self, *args, **kwargs):
        # this is now depracated: pass weight directly to the loss object
        raise ImportError(
            "The WeightedLoss class has been removed from graph-pes "
            "as of version 0.0.22. Please now pass loss weights directly "
            "to the loss instances! See the docs for more information: "
            "https://jla-gardner.github.io/graph-pes/fitting/losses.html"
        )


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

    Parameters
    ----------
    weight
        a scalar multiplier for weighting the value returned by
        :meth:`forward` as part of a :class:`TotalLoss`.
    is_per_atom
        whether this loss returns a value that is normalised per atom, or not.
        For instance, some metric that acts on ``"forces"`` is naturally
        per-atom, while a metric that acts on ``"energy"``, or the model etc.,
        is not. Specifying this correctly ensures that the effective batch size
        is chosen correctly when averaging over batches.
    """

    def __init__(self, weight, is_per_atom: bool = False):
        super().__init__()
        self.weight = weight
        self.is_per_atom = is_per_atom

    @abstractmethod
    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        r"""
        Compute the unweighted loss value.

        Note that only :class:`Loss`\ s that return a tensor can be used
        for training: we reserve the use of :class:`TorchMetric`\ s for
        evaluation metrics only.

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
    ) -> torch.Tensor | TorchMetric:
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

        energy_rmse_loss = PropertyLoss("energy", RMSE())
        energy_rmse_value = energy_rmse_loss(
            predictions,  # a dict of key (energy/force/etc.) to value
            graph.properties,
        )
    """

    def __init__(
        self,
        property: PropertyKey,
        metric: Metric | MetricName | TorchMetric = "RMSE",
        weight: float = 1.0,
    ):
        super().__init__(
            weight,
            is_per_atom=property in ("forces", "local_energies"),
        )
        self.property: PropertyKey = property
        self.metric = parse_metric(metric)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        """
        Computes the loss value.

        Parameters
        ----------
        predictions
            The predictions from the ``model`` for the given ``graph``.
        """

        if isinstance(self.metric, torchmetrics.Metric):
            self.metric.update(
                predictions[self.property],
                graph.properties[self.property],
            )
            return self.metric

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


class SubLossPair(NamedTuple):
    loss_value: torch.Tensor
    weighted_loss_value: torch.Tensor
    is_per_atom: bool


class TotalLossResult(NamedTuple):
    loss_value: torch.Tensor
    components: dict[str, SubLossPair]


class TotalLoss(torch.nn.Module):
    r"""
    A lightweight wrapper around a collection of losses.

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

    def __init__(self, losses: Sequence[Loss]):
        super().__init__()
        self.losses = UniformModuleList(losses)

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

        for loss in self.losses:
            loss_value = loss(model, graph, predictions)
            weighted_loss_value = loss_value * loss.weight

            if not isinstance(loss_value, torch.Tensor):
                raise ValueError(
                    f"Total losses can only be used alongside metrics that "
                    f"return a tensor. Instead, loss {loss.name} returned "
                    f"{type(loss_value)}."
                )

            total_loss += weighted_loss_value
            components[loss.name] = SubLossPair(
                loss_value,
                weighted_loss_value,
                loss.is_per_atom,
            )

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
        return "\n".join(
            ["TotalLoss:"]
            + ["    ".join(str(loss).split("\n")) for loss in self.losses]
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
        metric: Metric | MetricName | TorchMetric = "RMSE",
        weight: float = 1.0,
    ):
        super().__init__("energy", metric, weight)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        if isinstance(self.metric, torchmetrics.Metric):
            self.metric.update(
                divide_per_atom(predictions["energy"], graph),
                divide_per_atom(graph.properties["energy"], graph),
            )
            return self.metric

        return self.metric(
            divide_per_atom(predictions["energy"], graph),
            divide_per_atom(graph.properties["energy"], graph),
        )

    @property
    def name(self) -> str:
        return f"per_atom_energy_{_get_metric_name(self.metric)}"


class ForceRMSE(PropertyLoss):
    """
    Alias for :class:`PropertyLoss` with ``property="forces"`` and
    ``metric=RMSE``.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("forces", RMSE(), weight)


## METRICS ##


def parse_metric(metric: Metric | MetricName | TorchMetric | None) -> Metric:
    if isinstance(metric, str):
        return {"MAE": MAE(), "RMSE": RMSE(), "MSE": MSE()}[metric]

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

    .. note::

        Metrics are computed per-batch in the Lightning Trainer.
        When aggregating this metric across multiple batches, we therefore
        get the mean of the per-batch RMSEs, which is not the same as the
        RMSE between all predictions and targets from all batches.

        ``graph-pes`` avoids this issue for all RMSE-based metrics logged as
        ``"{valid|test}/metrics/..._rmse"`` by using a different,
        ``torchmetrics`` based implementation.
    """

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (super().forward(input, target)).sqrt()

    @property
    def name(self) -> str:
        return "rmse_batchwise"


class MAE(torch.nn.L1Loss):
    r"""
    Mean absolute error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left| \hat{P}_i - P_i \right|
    """


def _get_metric_name(metric: Metric) -> str:
    if hasattr(metric, "name"):
        return metric.name  # type: ignore

    if isinstance(metric, torchmetrics.MeanSquaredError):
        return "mse" if metric.squared else "rmse"

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
