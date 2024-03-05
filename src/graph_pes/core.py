from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence, overload

import torch
from torch import Tensor, nn

from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    batch_graphs,
    is_periodic,
    keys,
    sum_per_structure,
)
from graph_pes.transform import Identity, PerAtomStandardScaler, Transform
from graph_pes.util import differentiate, require_grad


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all graph-based, energy-conserving models of the
    PES that make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E(\mathcal{G}) = \sum_i \varepsilon_i

    To create such a model, implement :meth:`predict_local_energies`,
    which takes an :class:`AtomicGraph`, or an :class:`AtomicGraphBatch`,
    and returns a per-atom prediction of the local energy. For a simple example,
    see the :class:`PairPotential <graph_pes.models.pairwise.PairPotential>`
    `implementation <_modules/graph_pes/models/pairwise.html#PairPotential>`_.

    Under the hood, :class:`GraphPESModel`s pass the local energy predictions
    through a :class:`graph_pes.transform.Transform` before summing them to
    get the total energy. By default, this learns a per-species local-energy
    offset and scale. This can be overridden by setting the
    :attr:`energy_transform` attribute to any custom
    :class:`graph_pes.transform.Transform`, trainable or otherwise.
    """

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        """
        Predict the (non-transformed) local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        Tensor
            The per-atom local energy predictions with shape :code:`(N,)`.
        """

    def __init__(self):
        super().__init__()
        # assigned here to appease torchscript and mypy:
        # this gets overridden in pre_fit below
        self.energy_transform: Transform = Identity()

    def pre_fit(self, graphs: AtomicGraphBatch):
        """
        Perform optional pre-processing of the training data.

        By default, this fits a :class:`graph_pes.transform.PerAtomScale`
        and :class:`graph_pes.transform.PerAtomShift` to the energies
        of the training data, such that, before training, a unit-Normal
        output by the underlying model will result in energy predictions
        that are distributed according to the training data.

        For an example customisation of this method, see the
        :class:`LennardJones <graph_pes.models.pairwise.LennardJones>`
        `implementation
        <_modules/graph_pes/models/pairwise.html#LennardJones>`_.

        Parameters
        ----------
        graphs
            The training data.
        """
        if "energy" not in graphs:
            raise ValueError("No energy data in the training graphs.")

        transform = PerAtomStandardScaler()
        transform.fit(graphs["energy"], graphs)
        # The standard scaler maps final energies to a unit normal
        # distribution. We want to go the other way for our predictions,
        # and so use the inverse transform.
        self.energy_transform = transform.inverse()

    def forward(self, graph: AtomicGraph) -> Tensor:
        """
        Calculate the total energy of the structure.

        Parameters
        ----------
        graph
            The atomic structure to evaluate.

        Returns
        -------
        Tensor
            The total energy of the structure/s. If the input is a batch
            of graphs, the result will be a tensor of shape :code:`(B,)`,
            where :code:`B` is the batch size, else a scalar.
        """
        local_energies = self.predict_local_energies(graph).squeeze()
        transformed = self.energy_transform(local_energies, graph)
        return sum_per_structure(transformed, graph)

    # add type hints to play nicely with mypy
    def __call__(self, graph: AtomicGraph) -> Tensor:
        return super().__call__(graph)

    def __add__(self, other: GraphPESModel) -> Ensemble:
        return Ensemble([self, other], aggregation="sum")


class Ensemble(GraphPESModel):
    """
    An ensemble of :class:`GraphPESModel` models.

    Parameters
    ----------
    models
        the models to ensemble.
    aggregation
        the method of aggregating the predictions of the models.
    weights
        scalar weights for combining each model's prediction.
    trainable_weights
        whether the weights are trainable.

    Examples
    --------
    Create a model with explicit two-body and multi-body terms:

    .. code-block:: python

        from graph_pes.models.pairwise import LennardJones
        from graph_pes.models.schnet import SchNet
        from graph_pes.core import Ensemble

        # create an ensemble of two models
        # equivalent to Ensemble([LennardJones(), SchNet()], aggregation="sum")
        ensemble = LennardJones() + SchNet()

    Use several models to get an average prediction:

    .. code-block:: python

        models = ... # load/train your models
        ensemble = Ensemble(models, aggregation="mean")
        predictions = ensemble.predict(test_graphs)
        ...
    """

    def __init__(
        self,
        models: list[GraphPESModel],
        aggregation: Literal["mean", "sum"] = "mean",
        weights: list[float] | None = None,
        trainable_weights: bool = False,
    ):
        super().__init__()
        self.models: list[GraphPESModel] = nn.ModuleList(models)  # type: ignore
        self.aggregation = aggregation
        self.weights = nn.Parameter(
            torch.tensor(
                weights or [1.0] * len(models), requires_grad=trainable_weights
            )
        )

        # use the energy summation of each model separately
        self.energy_summation = None

    def predict_local_energies(self, graph: AtomicGraph):
        raise NotImplementedError(
            "Ensemble models don't have a single local energy prediction."
        )

    def forward(self, graph: AtomicGraph):
        predictions: Tensor = torch.stack(
            [
                w * model.total_energy(graph)
                for w, model in zip(self.weights, self.models)
            ]
        ).sum(dim=0)
        if self.aggregation == "mean":
            return predictions / self.weights.sum()
        else:
            return predictions

    def __repr__(self):
        info = [str(self.models), f"aggregation={self.aggregation}"]
        if self.weights.requires_grad:
            info.append(f"weights={self.weights.tolist()}")
        info = "\n  ".join(info)
        return f"Ensemble(\n  {info}\n)"


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | list[AtomicGraph],
    *,
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]: ...


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | list[AtomicGraph],
    *,
    properties: Sequence[keys.LabelKey],
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]: ...


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | list[AtomicGraph],
    *,
    property: keys.LabelKey,
    training: bool = False,
) -> Tensor: ...


# TODO: implement max batch size
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | list[AtomicGraph],
    *,
    properties: Sequence[keys.LabelKey] | None = None,
    property: keys.LabelKey | None = None,
    training: bool = False,
) -> dict[keys.LabelKey, Tensor] | Tensor:
    """
    Evaluate the model on the given structure to get
    the properties requested.

    Parameters
    ----------
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
    >>> model.predict(graph_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...), 'stress': tensor(...)}
    >>> model.predict(graph_no_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...)}
    >>> model.predict(graph_pbc, property="energy")
    tensor(-12.3)
    """

    # check correctly called
    if property is not None and properties is not None:
        raise ValueError("Can't specify both `property` and `properties`")

    if isinstance(graph, list):
        graph = batch_graphs(graph)

    if properties is None:
        if is_periodic(graph):
            properties = [keys.ENERGY, keys.FORCES, keys.STRESS]
        else:
            properties = [keys.ENERGY, keys.FORCES]

    if keys.STRESS in properties and not is_periodic(graph):
        raise ValueError("Can't predict stress without cell information.")

    predictions: dict[keys.LabelKey, Tensor] = {}

    # setup for calculating stress:
    if keys.STRESS in properties:
        # The virial stress tensor is the gradient of the total energy wrt
        # an infinitesimal change in the cell parameters.
        # We therefore add this change to the cell, such that
        # we can calculate the gradient wrt later if required.
        #
        # See <> TODO: find reference
        actual_cell = graph[keys.CELL]
        change_to_cell = torch.zeros_like(actual_cell, requires_grad=True)
        symmetric_change = 0.5 * (
            change_to_cell + change_to_cell.transpose(-1, -2)
        )
        graph[keys.CELL] = actual_cell + symmetric_change
    else:
        change_to_cell = torch.zeros_like(graph[keys.CELL])

    # use the autograd machinery to auto-magically
    # calculate forces and stress from the energy
    with require_grad(graph[keys._POSITIONS]), require_grad(change_to_cell):
        energy = model(graph)

        if keys.ENERGY in properties:
            predictions[keys.ENERGY] = energy

        if keys.FORCES in properties:
            dE_dR = differentiate(energy, graph[keys._POSITIONS])
            predictions[keys.FORCES] = -dE_dR

        if keys.STRESS in properties:
            stress = differentiate(energy, change_to_cell)
            predictions[keys.STRESS] = stress

    if not training:
        for key, value in predictions.items():
            predictions[key] = value.detach()

    if property is not None:
        return predictions[property]

    return predictions
