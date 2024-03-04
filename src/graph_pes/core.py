from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence, overload

import torch
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
from jaxtyping import Float
from torch import Tensor, nn


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

    Under the hood, :class:`GraphPESModel` contains an
    :class:`EnergySummation` module, which is responsible for
    summing over local energies to obtain the total energy/ies,
    with optional transformations of the local and total energies.
    By default, this learns a per-species, local energy offset and scale.
    """

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> Float[Tensor, "N"]:
        """
        Predict the (standardized) local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        """

    @overload
    def forward(self, graph: AtomicGraphBatch) -> Float[Tensor, "N"]: ...

    @overload
    def forward(self, graph: AtomicGraph) -> Float[Tensor, "1"]: ...

    def forward(self, graph: AtomicGraph):
        """
        Predict the total energy of the structure/s.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        """

        # local predictions
        local_energies = self.predict_local_energies(graph).squeeze()

        # sum over atoms to get total energy
        return self.energy_summation(local_energies, graph)

    def __init__(self):
        super().__init__()
        self.energy_summation = EnergySummation()

    def __add__(self, other: GraphPESModel) -> Ensemble:
        return Ensemble([self, other], aggregation="sum")

    def pre_fit(self, graphs: AtomicGraphBatch):
        """
        Perform optional pre-processing of the training data.

        By default, this fits a :class:`graph_pes.transform.PerAtomShift`
        and :class:`graph_pes.transform.PerAtomScale` to the energies
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
        self.energy_summation.fit_to_graphs(graphs)

    @overload
    def predict(
        self,
        graph: AtomicGraph | list[AtomicGraph],
        *,
        training: bool = False,
    ) -> dict[keys.LabelKey, Tensor]: ...

    @overload
    def predict(
        self,
        graph: AtomicGraph | list[AtomicGraph],
        *,
        properties: Sequence[keys.LabelKey],
        training: bool = False,
    ) -> dict[keys.LabelKey, Tensor]: ...

    @overload
    def predict(
        self,
        graph: AtomicGraph | list[AtomicGraph],
        *,
        property: keys.LabelKey,
        training: bool = False,
    ) -> Tensor: ...

    # TODO: implement max batch size
    def predict(
        self,
        graph: AtomicGraph | list[AtomicGraph],
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
            :code:`[keys.ENERGY, keys.FORCES]` if the structure
            has no cell, and :code:`[keys.ENERGY, keys.FORCES,
            keys.STRESS]` if it does.
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
            energy = self(graph)

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

    # add type hints to play nicely with mypy
    @overload
    def __call__(self, graph: AtomicGraph) -> Float[Tensor, "1"]: ...

    @overload
    def __call__(self, graph: AtomicGraphBatch) -> Float[Tensor, "N"]: ...

    def __call__(self, graph: AtomicGraph):
        return super().__call__(graph)


class EnergySummation(nn.Module):
    """
    A module for summing local energies to obtain the total energy.

    Before summation, :code:`local_transform` is applied to the local energies.
    After summation, :code:`total_transform` is applied to the total energy.

    By default, :code:`EnergySummation()` learns a per-species, local energy
    offset and scale.

    Parameters
    ----------
    local_transform
        A transformation of the local energies.
    total_transform
        A transformation of the total energy.
    """

    def __init__(
        self,
        local_transform: Transform | None = None,
        total_transform: Transform | None = None,
    ):
        super().__init__()

        # if both None, default to a per-species, local energy offset
        if local_transform is None and total_transform is None:
            local_transform = PerAtomStandardScaler()
        self.local_transform: Transform = local_transform or Identity()
        self.total_transform: Transform = total_transform or Identity()

    @overload
    def forward(
        self, local_energies: Float[Tensor, "N"], graph: AtomicGraphBatch
    ) -> Float[Tensor, "S"]: ...

    @overload
    def forward(
        self, local_energies: Float[Tensor, "N"], graph: AtomicGraph
    ) -> Float[Tensor, "1"]: ...

    def forward(self, local_energies: Tensor, graph: AtomicGraph):
        """
        Sum the local energies to obtain the total energy.

        Parameters
        ----------
        local_energies
            The local energies.
        graph
            The graph representation of the structure/s.
        """
        local_energies = self.local_transform.inverse(local_energies, graph)
        total_E = sum_per_structure(local_energies, graph)
        total_E = self.total_transform.inverse(total_E, graph)
        return total_E

    def fit_to_graphs(self, graphs: AtomicGraphBatch | list[AtomicGraph]):
        """
        Fit the transforms to the training data.

        Parameters
        ----------
        graphs
            The training data.
        """

        if isinstance(graphs, list):
            graphs = batch_graphs(graphs)

        assert keys.ENERGY in graphs, "No energy data in training graphs."

        for transform in [self.local_transform, self.total_transform]:
            transform.fit(graphs[keys.ENERGY], graphs)  # type: ignore

    def __repr__(self):
        # only show non-default transforms
        info = [
            f"{t}_transform={transform}"
            for t, transform in [
                ("local", self.local_transform),
                ("total", self.total_transform),
            ]
            if not isinstance(transform, Identity)
        ]
        info = "\n  ".join(info)
        return f"EnergySummation(\n  {info}\n)"

    # add type hints to play nicely with mypy
    def __call__(
        self,
        local_energies: Float[Tensor, "graph.n_atoms"],
        graph: AtomicGraph,
    ) -> Tensor:
        return super().__call__(local_energies, graph)


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
        predictions: Tensor = sum(
            w * model(graph) for w, model in zip(self.weights, self.models)
        )  # type: ignore
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

    # add type hints to play nicely with mypy
    def __call__(self, graph: AtomicGraph) -> Tensor:
        return super().__call__(graph)
