from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal, Sequence, overload

import torch
from torch import Tensor, nn

from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    has_cell,
    keys,
    sum_per_structure,
    to_batch,
)
from graph_pes.transform import PerAtomStandardScaler, Transform
from graph_pes.util import differentiate, require_grad


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

    Under the hood, :class:`GraphPESModel`\ s pass the local energy predictions
    through a :class:`graph_pes.transform.Transform` before summing them to
    get the total energy. By default, this learns a per-species local-energy
    scale and shift. This can be changed by directly altering passing a
    different :class:`~graph_pes.transform.Transform` to this base class's
    constructor.

    Parameters
    ----------
    energy_transform
        The transform to apply to the local energy predictions before summing
        them to get the total energy. By default, this is a learnable
        per-species scale and shift.
    """

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

    def __init__(self, energy_transform: Transform | None = None):
        super().__init__()
        self.energy_transform: Transform = (
            PerAtomStandardScaler()
            if energy_transform is None
            else energy_transform
        )

        #
        self._has_been_pre_fit: Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(False))

    def pre_fit(self, graphs: AtomicGraphBatch, relative: bool = True):
        """
        Pre-fit the model to the training data.

        By default, this fits the :code:`energy_transform` to the energies
        of the training data. To add additional pre-fitting steps, override
        :meth:`_extra_pre_fit`. As an example of this, see the
        :class:`~graph_pes.models.pairwise.LennardJones`
        `implementation
        <_modules/graph_pes/models/pairwise.html#LennardJones>`__.

        If the model has already been pre-fitted, subsequent calls to
        :meth:`pre_fit` will be ignored.

        Parameters
        ----------
        graphs
            The training data.
        relative
            Whether to account for the current energy predictions when fitting
            the energy transform.

        Example
        -------
        Without any pre-fitting, models *tend* to predict energies that are
        close to 0:

        >>> from graph_pes.models.zoo import LennardJones
        >>> model = LennardJones()
        >>> model
        LennardJones(
          (epsilon): 0.1
          (sigma): 1.0
          (energy_transform): PerAtomShift(
              Cu : [0.],
          )
        )
        >>> from graph_pes.analysis import parity_plot
        >>> parity_plot(model, val, units="eV")

        .. image:: /_static/lj-parity-raw.svg
            :align: center

        Pre-fitting a model's :code:`energy_transform` to the training data
        (together with any other steps defined in :meth:`_extra_pre_fit`)
        dramatically improves the predictions for free:

        >>> from graph_pes.data import to_batch
        >>> model = LennardJones()
        >>> model.pre_fit(to_batch(train_set), relative=False)
        >>> model
        LennardJones(
          (epsilon): 0.1
          (sigma): 2.27
          (energy_transform): PerAtomShift(
              Cu : [3.5229],
          )
        )
        >>> parity_plot(model, val, units="eV")

        .. image:: /_static/lj-parity-prefit.svg
            :align: center

        Accounting for the model's current predictions when fitting the
        energy transforms (the default behaviour) leads to even better
        pre-conditioned models:

        >>> model = LennardJones()
        >>> model.pre_fit(to_batch(train_set), relative=True)
        >>> model
        LennardJones(
          (epsilon): 0.1
          (sigma): 2.27
          (energy_transform): PerAtomShift(
              Cu : [2.9238],
          )
        )
        >>> parity_plot(model, val, units="eV")

        .. image:: /_static/lj-parity-relative.svg
            :align: center
        """
        if self._has_been_pre_fit:
            warnings.warn(
                "This model has already been pre-fitted. "
                "Subsequent calls to pre_fit will be ignored.",
                stacklevel=2,
            )
            return

        self._has_been_pre_fit.fill_(True)

        if self._extra_pre_fit(graphs):
            return

        if "energy" not in graphs:
            warnings.warn(
                "The training data doesn't contain energies. "
                "The energy transform will not be fitted.",
                stacklevel=2,
            )
            return

        target = graphs["energy"]
        if relative:
            with torch.no_grad():
                target = graphs["energy"] - self(graphs)

        self.energy_transform.fit_to_target(target, graphs)

    def _extra_pre_fit(self, graphs: AtomicGraphBatch) -> bool | None:
        """
        Override this method to perform additional pre-fitting steps.
        Return ``True`` to surpress the default pre-fitting of the energy
        transform implemented on this base class.
        """

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
            [w * model(graph) for w, model in zip(self.weights, self.models)]
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
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]:
    """test"""


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    properties: Sequence[keys.LabelKey],
    training: bool = False,
) -> dict[keys.LabelKey, Tensor]:
    """
    test
    """
    ...


@overload
def get_predictions(
    model: GraphPESModel,
    graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
    *,
    property: keys.LabelKey,
    training: bool = False,
) -> Tensor:
    """test"""


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
