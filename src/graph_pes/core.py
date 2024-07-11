from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Sequence, overload

import torch
from torch import Tensor, nn

from graph_pes.data.dataset import LabelledGraphDataset

from .graphs import (
    AtomicGraph,
    AtomicGraphBatch,
    LabelledBatch,
    LabelledGraph,
    keys,
)
from .graphs.operations import (
    has_cell,
    sum_per_structure,
    to_batch,
)
from .nn import PerElementParameter
from .util import differentiate, require_grad, uniform_repr


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all graph-based, energy-conserving models of the
    PES that make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E(\mathcal{G}) = \sum_i \varepsilon_i

    To create such a model, implement :meth:`predict_local_energies`,
    which takes an :class:`~graph_pes.data.AtomicGraph`, or an
    :class:`~graph_pes.data.AtomicGraphBatch`,
    and returns a per-atom prediction of the local energy. For a simple example,
    see the :class:`PairPotential <graph_pes.models.pairwise.PairPotential>`
    `implementation <_modules/graph_pes/models/pairwise.html#PairPotential>`_.
    """

    def __init__(self):
        super().__init__()

        # save as a buffer so that this is de/serialized
        # with the model
        self._has_been_pre_fit: Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(False))

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
            where :code:`B` is the batch size. Otherwise, a scalar tensor
            will be returned.
        """
        local_energies = self.predict_local_energies(graph).squeeze()
        return sum_per_structure(local_energies, graph)

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        """
        Predict the local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        Tensor
            The per-atom local energy predictions, with shape :code:`(N,)`.
        """

    def pre_fit(self, graphs: LabelledGraphDataset | Sequence[LabelledGraph]):
        """
        Pre-fit the model to the training data.

        This method detects the unique atomic numbers in the training data
        and registers these with all of the model's per-element parameters
        to ensure correct parameter counting.

        Additionally, this method performs any model-specific pre-fitting
        steps, as implemented in :meth:`model_specific_pre_fit`.

        As an example of a model-specific pre-fitting process, see the
        :class:`~graph_pes.models.pairwise.LennardJones`
        `implementation
        <_modules/graph_pes/models/pairwise.html#LennardJones>`__.

        If the model has already been pre-fitted, subsequent calls to
        :meth:`pre_fit` will be ignored (and a warning will be raised).

        Parameters
        ----------
        graphs
            The training data.
        """

        if self._has_been_pre_fit.item():
            model_name = self.__class__.__name__
            warnings.warn(
                f"This model ({model_name}) has already been pre-fitted. "
                "This, and any subsequent, call to pre_fit will be ignored.",
                stacklevel=2,
            )
            return

        if len(graphs) > 10_000:
            warnings.warn(
                f"Pre-fitting on a large dataset ({len(graphs):,} structures). "
                "This may take some time. Consider using a smaller, "
                "representative collection of structures for pre-fitting. "
                "See LabelledGraphDataset.sample() for more information.",
                stacklevel=2,
            )

        if isinstance(graphs, LabelledGraphDataset):
            graphs = list(graphs)

        graph_batch = to_batch(graphs)

        self._has_been_pre_fit.fill_(True)
        self.model_specific_pre_fit(graph_batch)

        # register all per-element parameters
        for param in self.parameters():
            if isinstance(param, PerElementParameter):
                param.register_elements(
                    torch.unique(graph_batch[keys.ATOMIC_NUMBERS]).tolist()
                )

    def model_specific_pre_fit(self, graphs: LabelledBatch) -> None:
        """
        Override this method to perform additional pre-fitting steps.

        As an example, see the
        :class:`~graph_pes.models.pairwise.LennardJones`
        `implementation
        <_modules/graph_pes/models/pairwise.html#LennardJones>`__.

        Parameters
        ----------
        graphs
            The training data.
        """

    # add type hints to play nicely with mypy
    def __call__(self, graph: AtomicGraph) -> Tensor:
        return super().__call__(graph)

    def __add__(self, other: GraphPESModel | AdditionModel) -> AdditionModel:
        if not isinstance(other, GraphPESModel):
            raise TypeError(f"Can't add {type(self)} and {type(other)}")

        if isinstance(other, AdditionModel):
            if isinstance(self, AdditionModel):
                return AdditionModel([*self.models, *other.models])
            return AdditionModel([self, *other.models])

        if isinstance(self, AdditionModel):
            return AdditionModel([*self.models, other])
        return AdditionModel([self, other])


class AdditionModel(GraphPESModel):
    """
    A wrapper that makes predictions as the sum of the predictions
    of its constituent models.

    Parameters
    ----------
    models
        the models to sum.

    Examples
    --------
    Create a model with explicit two-body and multi-body terms:

    .. code-block:: python

        from graph_pes.models.zoo import LennardJones, SchNet
        from graph_pes.core import AdditionModel

        # create a model that sums two models
        # equivalent to LennardJones() + SchNet()
        model = AdditionModel([LennardJones(), SchNet()])
    """

    def __init__(self, models: Sequence[GraphPESModel]):
        super().__init__()
        self.models: list[GraphPESModel] = nn.ModuleList(models)  # type: ignore

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        predictions = torch.stack(
            [
                model.predict_local_energies(graph).squeeze()
                for model in self.models
            ]
        )  # (atoms, models)
        return torch.sum(predictions, dim=0)  # (atoms,) sum over models

    def model_specific_pre_fit(self, graphs: LabelledBatch) -> None:
        for model in self.models:
            model.model_specific_pre_fit(graphs)

    def __repr__(self):
        # model_info = "\n  ".join(map(str, self.models))
        # return f"{self.__class__.__name__}(\n  {model_info}\n)"
        return uniform_repr(self.__class__.__name__, *self.models)


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]: ...
@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    properties: Sequence[keys.LabelKey],
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]: ...
@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    property: keys.LabelKey,
    training: bool = False,
) -> Tensor: ...
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
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
    >>> get_predictions(model, graph_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...), 'stress': tensor(...)}
    >>> get_predictions(model, graph_no_pbc)
    {'energy': tensor(-12.3), 'forces': tensor(...)}
    >>> get_predictions(model, graph, property="energy")
    tensor(-12.3)
    """

    # check correctly called
    if property is not None and properties is not None:
        raise ValueError("Can't specify both `property` and `properties`")

    if isinstance(graph, Sequence):
        graph = to_batch(graph)

    if properties is None:
        if has_cell(graph):
            properties = [keys.ENERGY, keys.FORCES, keys.STRESS]
        else:
            properties = [keys.ENERGY, keys.FORCES]

    want_stress = keys.STRESS in properties or property == keys.STRESS
    if want_stress and not has_cell(graph):
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
    with require_grad(graph[keys._POSITIONS], change_to_cell):
        energy = model(graph)

        if keys.ENERGY in properties:
            predictions[keys.ENERGY] = energy

        if keys.FORCES in properties:
            dE_dR = differentiate(energy, graph[keys._POSITIONS])
            predictions[keys.FORCES] = -dE_dR

        # TODO: check stress vs virial common definition
        if keys.STRESS in properties:
            stress = differentiate(energy, change_to_cell)
            predictions[keys.STRESS] = stress

    if not training:
        for key, value in predictions.items():
            predictions[key] = value.detach()

    if property is not None:
        return predictions[property]

    return predictions
