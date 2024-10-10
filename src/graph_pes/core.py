from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Sequence, overload

import torch
from ase.data import chemical_symbols
from torch import nn

from graph_pes.data.dataset import LabelledGraphDataset
from graph_pes.logger import logger

from .graphs import (
    AtomicGraph,
    AtomicGraphBatch,
    LabelledBatch,
    LabelledGraph,
    keys,
)
from .graphs.operations import (
    has_cell,
    is_batch,
    sum_per_structure,
    to_batch,
    trim_edges,
)
from .nn import PerElementParameter
from .util import differentiate


class ConservativePESModel(nn.Module, ABC):
    r"""
    An abstract base class for all energy-conserving models of the
    PES that make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E(\mathcal{G}) = \sum_i \varepsilon_i

    where :math:`\varepsilon_i` is the local energy of atom :math:`i`, and
    :math:`\mathcal{G}` is the atomic graph representation of the structure.

    To create such a model, implement :meth:`predict_local_energies`,
    which takes an :class:`~graph_pes.graphs.AtomicGraph`, or an
    :class:`~graph_pes.graphs.AtomicGraphBatch`,
    and returns a per-atom prediction of the local energy. For a simple example,
    see the :class:`PairPotential <graph_pes.models.pairwise.PairPotential>`
    `implementation <_modules/graph_pes/models/pairwise.html#PairPotential>`_.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model (if applicable). During the forward
        pass, only edges between atoms that are closer than this distance will
        be considered.
    auto_scale
        Whether to automatically scale raw predictions by (learnable)
        per-element scaling factors as calculated from the data passed to
        :meth:`pre_fit` (typically the training data). If ``True``,
        :math:`\varepsilon_i = \sigma_{Z_i} \cdot \varepsilon_i`, where
        :math:`\sigma_{Z_i}` is the scaling factor for element :math:`Z_i`.
    """

    def __init__(self, cutoff: float, auto_scale: bool):
        super().__init__()

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))

        self.per_element_scaling: PerElementParameter | None
        if auto_scale:
            self.per_element_scaling = PerElementParameter.of_length(
                1,
                default_value=1.0,
                requires_grad=True,
            )
        else:
            self.per_element_scaling = None

        # save as a buffer so that this is de/serialized with the model
        # need to use int(False) to ensure nice torchscript behaviour
        self._has_been_pre_fit: torch.Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(int(False)))

    def predict_scaled_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        local_energies = self.predict_local_energies(graph).squeeze()
        if self.per_element_scaling is not None:
            scales = self.per_element_scaling[
                graph[keys.ATOMIC_NUMBERS]
            ].squeeze()
            local_energies = local_energies * scales
        return local_energies

    def forward(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Calculate the total energy of the structure.

        Parameters
        ----------
        graph
            The atomic structure to evaluate.

        Returns
        -------
        torch.Tensor
            The total energy of the structure/s. If the input is a batch
            of graphs, the result will be a tensor of shape :code:`(B,)`,
            where :code:`B` is the batch size. Otherwise, a scalar tensor
            will be returned.
        """
        graph = trim_edges(graph, self.cutoff.item())
        local_energies = self.predict_scaled_local_energies(graph).squeeze()
        return sum_per_structure(local_energies, graph)

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Predict the local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        torch.Tensor
            The per-atom local energy predictions, with shape :code:`(N,)`.
        """

    def pre_fit(
        self,
        graphs: LabelledGraphDataset | Sequence[LabelledGraph],
    ):
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

        model_name = self.__class__.__name__
        logger.debug(f"Attempting to pre-fit {model_name}")

        # 1. get the graphs as a single batch
        if isinstance(graphs, LabelledGraphDataset):
            graphs = list(graphs)
        graph_batch = to_batch(graphs)

        # 2a. if the graph has already been pre-fitted: warn
        if self._has_been_pre_fit.item():
            model_name = self.__class__.__name__
            warnings.warn(
                f"This model ({model_name}) has already been pre-fitted. "
                "This, and any subsequent, call to pre_fit will be ignored.",
                stacklevel=2,
            )

        # 2b. if the graph has not been pre-fitted: pre-fit
        else:
            if len(graphs) > 10_000:
                warnings.warn(
                    f"Pre-fitting on a large dataset ({len(graphs):,} graphs). "
                    "This may take some time. Consider using a smaller, "
                    "representative collection of structures for pre-fitting. "
                    "Set ``max_n_pre_fit`` in your config, or "
                    "see LabelledGraphDataset.sample() for more information.",
                    stacklevel=2,
                )

            self._has_been_pre_fit.fill_(int(True))
            self.model_specific_pre_fit(graph_batch)

        # 3. finally, register all per-element parameters
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
    def __call__(self, graph: AtomicGraph) -> torch.Tensor:
        return super().__call__(graph)

    @torch.jit.unused
    @property
    def elements_seen(self) -> list[str]:
        """The elements that the model has seen during training."""

        Zs = set()
        for param in self.parameters():
            if isinstance(param, PerElementParameter):
                Zs.update(param._accessed_Zs)
        return [chemical_symbols[Z] for Z in sorted(Zs)]

    @overload
    def get_predictions(
        self,
        graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
        *,
        properties: list[keys.LabelKey] | None = None,
    ) -> dict[keys.LabelKey, torch.Tensor]: ...
    @overload
    def get_predictions(
        self,
        graph: AtomicGraph | AtomicGraphBatch | Sequence[AtomicGraph],
        *,
        property: keys.LabelKey,
    ) -> torch.Tensor: ...
    @torch.jit.unused
    def get_predictions(
        self,
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
        if isinstance(graph, Sequence):
            graph = to_batch(graph)

        if property is not None:
            if properties is not None:
                raise ValueError(
                    "Cannot provide both `property` and `properties` arguments."
                )
            properties = [property]

        if properties is None:
            if has_cell(graph):
                properties = [keys.ENERGY, keys.FORCES, keys.STRESS]
            else:
                properties = [keys.ENERGY, keys.FORCES]

        preds = self._get_predictions(graph, properties, training=False)
        return preds[property] if property is not None else preds

    def _get_predictions(
        self,
        graph: AtomicGraph,
        properties: list[keys.LabelKey],
        training: bool,
        get_local_energies: bool = False,
    ) -> dict[keys.LabelKey, torch.Tensor]:
        # meant for internal use:
        # unfortunately verbose to ensure torchscript compatibility

        # [boring] setup, check correct usage and type checking
        # if isinstance(graph, Sequence):
        # graph = to_batch(graph)
        # if properties is None:
        #     if has_cell(graph):
        #         properties = [keys.ENERGY, keys.FORCES, keys.STRESS]
        #     else:
        #         properties = [keys.ENERGY, keys.FORCES]

        want_stress = keys.STRESS in properties
        if want_stress and not has_cell(graph):
            raise ValueError("Can't predict stress without cell information.")

        # [interesting] calculate the predictions
        # we do this in a conservative manner:
        # 1. always predict the energy
        # 2. if forces are requested, make sure that the graph's positions
        #    have gradients enabled **before** calling the model. Then use
        #    autograd to calculate the forces by differentiating the energy
        #    wrt the positions.
        # 3. if stress is requested, make sure that the graph's cell has
        #    gradients enabled **before** calling the model. Then use
        #    autograd to calculate the stress by differentiating the energy
        #    wrt a distortion of the cell.

        predictions: dict[keys.LabelKey, torch.Tensor] = {}

        existing_positions = graph[keys._POSITIONS]
        existing_cell = graph[keys.CELL]

        if want_stress:
            # See About>Theory in the graph-pes for an explanation of the
            # maths behind this.
            #
            # The stress tensor is the gradient of the total energy wrt
            # a symmetric expansion of the structure (i.e. that acts on
            # both the cell and the atomic positions).
            #
            # F. Knuth et al. All-electron formalism for total energy strain
            # derivatives and stress tensor components for numeric atom-centered
            # orbitals. Computer Physics Communications 190, 33–50 (2015).

            change_to_cell = torch.zeros_like(existing_cell)
            change_to_cell.requires_grad_(True)
            symmetric_change = 0.5 * (
                change_to_cell + change_to_cell.transpose(-1, -2)
            )  # (n_structures, 3, 3) if batched, else (3, 3)
            scaling = torch.eye(3) + symmetric_change

            if is_batch(graph):
                scaling_per_atom = torch.index_select(
                    scaling,
                    dim=0,
                    index=graph[keys.BATCH],  # type: ignore
                )  # (n_atoms, 3, 3)

                # to go from (N, 3) @ (N, 3, 3) -> (N, 3), we need un/squeeze:
                # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3) -> (N, 3)
                new_positions = (
                    graph[keys._POSITIONS].unsqueeze(-2) @ scaling_per_atom
                ).squeeze()
                # (M, 3, 3) @ (M, 3, 3) -> (M, 3, 3)
                new_cell = existing_cell @ scaling

            else:
                # (N, 3) @ (3, 3) -> (N, 3)
                new_positions = graph[keys._POSITIONS] @ scaling
                new_cell = existing_cell @ scaling

            # change to positions will be a tensor of all 0's, but will allow
            # gradients to flow backwards through the energy calculation
            # and allow us to calculate the stress tensor as the gradient
            # of the energy wrt the change in cell.
            graph[keys._POSITIONS] = new_positions
            graph[keys.CELL] = new_cell

        else:
            change_to_cell = torch.zeros_like(graph[keys.CELL])

        if keys.FORCES in properties:
            graph[keys._POSITIONS].requires_grad_(True)

        if not get_local_energies:
            energy = self(graph)
        else:
            scaled_local_energies = self.predict_scaled_local_energies(graph)
            predictions["local_energies"] = scaled_local_energies  # type: ignore
            energy = sum_per_structure(scaled_local_energies, graph)

        # use the autograd machinery to auto-magically
        # calculate forces and stress from the energy
        if keys.ENERGY in properties:
            predictions[keys.ENERGY] = energy

        if keys.FORCES in properties:
            dE_dR = differentiate(energy, graph[keys._POSITIONS])
            predictions[keys.FORCES] = -dE_dR

        # TODO: check stress vs virial common definition
        if keys.STRESS in properties:
            stress = differentiate(energy, change_to_cell)
            cell_volume = torch.det(graph[keys.CELL])
            if is_batch(graph):
                cell_volume = cell_volume.view(-1, 1, 1)
            predictions[keys.STRESS] = stress / cell_volume

        graph[keys._POSITIONS] = existing_positions
        graph[keys.CELL] = existing_cell

        if not training:
            for key, value in predictions.items():
                predictions[key] = value.detach()

        return predictions


class FunctionalModel(ConservativePESModel):
    """
    Wrap a function that returns an energy prediction into a model
    that can be used in the same way as other
    :class:`~graph_pes.core.ConservativePESModel` subclasses.

    .. warning::

        This model does not support local energy predictions, and therefore
        cannot be used for LAMMPS simulations. Force and stress predictions
        are still supported.

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
