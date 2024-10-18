from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from ase.data import chemical_symbols
from torch import nn

from graph_pes.data.dataset import LabelledGraphDataset
from graph_pes.logger import logger

from .graphs import (
    AtomicGraph,
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


class GraphPESModel(nn.Module, ABC):
    r"""
    All models implemented in ``graph-pes`` are subclasses of
    :class:`~graph_pes.core.GraphPESModel`.

    These models make predictions (via the
    :meth:`~graph_pes.core.GraphPESModel.predict` method) of the
    following properties:

    .. list-table::
            :header-rows: 1

            * - Key
              - Single graph
              - Batch of graphs
            * - :code:`"local_energies"`
              - :code:`(N,)`
              - :code:`(N,)`
            * - :code:`"energy"`
              - :code:`()`
              - :code:`(M,)`
            * - :code:`"forces"`
              - :code:`(N, 3)`
              - :code:`(N, 3)`
            * - :code:`"stress"`
              - :code:`(3, 3)`
              - :code:`(M, 3, 3)`

    assuming an input of an :class:`~graph_pes.graphs.AtomicGraph` with
    :code:`N` atoms, or an :class:`~graph_pes.graphs.AtomicGraphBatch` composed
    of :code:`M` graphs, each with :code:`N` atoms:

    Implementations must override the
    :meth:`~graph_pes.core.GraphPESModel.forward` method to generate a
    dictionary of predictions for the given graph. As a minimum, this must
    include a per-atom energy contribution (``"local_energies"``).

    For any other properties not returned by the forward pass,
    :class:`graph_pes.core.GraphPESModel.predict` will automatically infer these
    properties from the local energies as required:

    * ``"energy"``: as the sum of the local energies per structure.
    * ``"forces"``: as the negative gradient of the energy with respect to the
      atomic positions.
    * ``"stress"``: as the negative gradient of the energy with respect to a
      symmetric expansion of the unit cell.

    For more details on how these are calculated, see :doc:`../theory`.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model.
    implemented_properties
        The property predictions that the model implements in the forward pass.
        Must include at least ``"local_energies"``.
    """

    def __init__(
        self,
        cutoff: float,
        implemented_properties: list[keys.LabelKey],
    ):
        super().__init__()

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self._has_been_pre_fit: torch.Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(0))

        # setup up the output enhancers
        self.implemented_properties = implemented_properties
        if "local_energies" not in implemented_properties:
            raise ValueError(
                'All GraphPESModel\'s must implement a "local_energies" '
                "prediction."
            )

    @abstractmethod
    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        """
        The model's forward pass. Generate all properties for the given graph
        that are in this model's ``implemented_properties`` list.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        dict[keys.LabelKey, torch.Tensor]
            A dictionary mapping each implemented property to a tensor of
            predictions (see above for the expected shapes). Use
            :func:`~graph_pes.graphs.operations.is_batch` to check if the
            graph is batched in the forward pass.
        """
        ...

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[keys.LabelKey],
    ) -> dict[keys.LabelKey, torch.Tensor]:
        """
        Generate (optionally batched) predictions for the given
        ``properties`` and  ``graph``.

        This method returns a dictionary mapping each requested
        ``property`` to a tensor of predictions, relying on the model's
        :meth:`~graph_pes.core.GraphPESModel.forward` implementation
        together with :func:`torch.autograd.grad` to automatically infer any
        missing properties.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        properties
            The properties to predict. Can be any combination of
            ``"energy"``, ``"forces"``, ``"stress"``, and ``"local_energies"``.
        """

        # before anything, remove unnecessary edges:
        graph = trim_edges(graph, self.cutoff.item())

        # check to see if we need to infer any properties
        infer_stress = (
            keys.STRESS in properties
            and keys.STRESS not in self.implemented_properties
        )
        infer_forces = (
            keys.FORCES in properties
            and keys.FORCES not in self.implemented_properties
        )
        infer_energy = (
            any([keys.ENERGY in properties, infer_stress, infer_forces])
        ) and keys.ENERGY not in self.implemented_properties
        if infer_stress and not has_cell(graph):
            raise ValueError("Can't predict stress without cell information.")

        existing_positions = graph[keys._POSITIONS]
        existing_cell = graph[keys.CELL]

        # inference specific set up
        if infer_stress:
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
            scaling = (
                torch.eye(3, device=existing_cell.device) + symmetric_change
            )

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

        if infer_forces:
            graph[keys._POSITIONS].requires_grad_(True)

        # get the implemented properties
        predictions = self.forward(graph)

        if infer_energy:
            predictions[keys.ENERGY] = sum_per_structure(
                predictions[keys.LOCAL_ENERGIES],
                graph,
            )

        # use the autograd machinery to auto-magically
        # calculate forces and stress from the energy
        if infer_forces:
            dE_dR = differentiate(
                predictions[keys.ENERGY], graph[keys._POSITIONS]
            )
            predictions[keys.FORCES] = -dE_dR

        if infer_stress:
            stress = differentiate(predictions[keys.ENERGY], change_to_cell)
            cell_volume = torch.det(graph[keys.CELL])
            if is_batch(graph):
                cell_volume = cell_volume.view(-1, 1, 1)
            predictions[keys.STRESS] = stress / cell_volume

        # put things back to how they were before
        graph[keys._POSITIONS] = existing_positions
        graph[keys.CELL] = existing_cell

        # tidy up if in eval mode
        if not self.training:
            predictions = {k: v.detach() for k, v in predictions.items()}

        # return the output
        return predictions  # type: ignore

    @torch.no_grad()
    def pre_fit_all_components(
        self,
        graphs: LabelledGraphDataset | Sequence[LabelledGraph],
    ):
        """
        Pre-fit the model, and all its components, to the training data.

        Some models require pre-fitting to the training data to set certain
        parameters. For example, the :class:`~graph_pes.models.pairwise.LennardJones`
        model uses the distribution of interatomic distances in the training data
        to set the length-scale parameter.

        In the ``graph-pes-train`` routine, this method is called before
        "normal" training begins (you can turn this off with a config option).

        This method does two things:

        1. iterates over all the model's :class:`~torch.nn.Module` components
            (inlcuding itself) and calls their :meth:`pre_fit` method (if it exists
            - see for instance :class:`~graph_pes.models.pairwise.LennardJones` for
            an example of a model-specific pre-fit method, and
            :class:`~graph_pes.models.components.scaling.LocalEnergiesScaler` for an example of a
            component-specific pre-fit method).
        2. registers all the unique atomic numbers in the training data with
           all of the model's :class:`~graph_pes.nn.PerElementParameter`
           instances to ensure correct parameter counting.

        If the model has already been pre-fitted, subsequent calls to
        :meth:`pre_fit_all_components` will be ignored (and a warning will be raised).

        Parameters
        ----------
        graphs
            The training data.
        """  # noqa: E501

        model_name = self.__class__.__name__
        logger.debug(f"Attempting to pre-fit {model_name}")

        # 1. get the graphs as a single batch
        if isinstance(graphs, LabelledGraphDataset):
            graphs = list(graphs)
        graph_batch = to_batch(graphs)

        # 2a. if the graph has already been pre-fitted: warn
        if self._has_been_pre_fit:
            model_name = self.__class__.__name__
            warnings.warn(
                f"This model ({model_name}) has already been pre-fitted. "
                "This, and any subsequent, call to pre_fit will be ignored.",
                stacklevel=2,
            )

        # 2b. if the model has not been pre-fitted: pre-fit
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

            self.pre_fit(graph_batch)
            # pre-fit any sub-module with a pre_fit method
            for module in self.modules():
                if hasattr(module, "pre_fit"):
                    module.pre_fit(graph_batch)

            self._has_been_pre_fit = torch.tensor(1)

        # 3. finally, register all per-element parameters (no harm in doing this
        #    multiple times)
        for param in self.parameters():
            if isinstance(param, PerElementParameter):
                param.register_elements(
                    torch.unique(graph_batch[keys.ATOMIC_NUMBERS]).tolist()
                )

    def pre_fit(self, graphs: LabelledBatch) -> None:
        """
        Override this method to perform additional pre-fitting steps.

        See :class:`~graph_pes.models.components.scaling.LocalEnergiesScaler` or
        :class:`~graph_pes.models.offsets.EnergyOffset` for examples of this.

        Parameters
        ----------
        graphs
            The training data.
        """

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        """
        Return a list of parameters that should not be decayed during training.

        By default, this method recurses over all available sub-modules
        and calls their :meth:`non_decayable_parameters` (if it is defined).

        See :class:`~graph_pes.models.components.scaling.LocalEnergiesScaler`
        for an example of this.
        """
        found = []
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "non_decayable_parameters"):
                found.extend(module.non_decayable_parameters())
        return found

    # add type hints for mypy etc.
    def __call__(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        return super().__call__(graph)

    def get_all_PES_predictions(
        self, graph: AtomicGraph
    ) -> dict[keys.LabelKey, torch.Tensor]:
        """
        Get all the properties that the model can predict
        for the given ``graph``.
        """
        properties: list[keys.LabelKey] = [
            keys.ENERGY,
            keys.FORCES,
            keys.LOCAL_ENERGIES,
        ]
        if has_cell(graph):
            properties.append(keys.STRESS)
        return self.predict(graph, properties)

    def predict_energy(self, graph: AtomicGraph) -> torch.Tensor:
        """Convenience method to predict just the energy."""

        return self.predict(graph, ["energy"])["energy"]

    def predict_forces(self, graph: AtomicGraph) -> torch.Tensor:
        """Convenience method to predict just the forces."""
        return self.predict(graph, ["forces"])["forces"]

    def predict_stress(self, graph: AtomicGraph) -> torch.Tensor:
        """Convenience method to predict just the stress."""
        return self.predict(graph, ["stress"])["stress"]

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """Convenience method to predict just the local energies."""
        return self.predict(graph, ["local_energies"])["local_energies"]

    @torch.jit.unused
    @property
    def elements_seen(self) -> list[str]:
        """The elements that the model has seen during training."""

        Zs = set()
        for param in self.parameters():
            if isinstance(param, PerElementParameter):
                Zs.update(param._accessed_Zs)
        return [chemical_symbols[Z] for Z in sorted(Zs)]
