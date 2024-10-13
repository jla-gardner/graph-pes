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


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all models of the PES that act on
    graph-representations (:class:`~graph_pes.graphs.AtomicGraph`)
    of atomic structures.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model.
    """

    def __init__(self, cutoff: float):
        super().__init__()

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self._has_been_pre_fit = False

    @abstractmethod
    def predict(
        self,
        graph: AtomicGraph,
        properties: list[keys.LabelKey],
    ) -> dict[keys.LabelKey, torch.Tensor]:
        """
        Generate (optionally batched) predictions for the given
        ``properties`` and  ``graph``.

        This method should return a dictionary mapping each requested
        ``property`` to a tensor of predictions.

        For a single structure with :code:`N` atoms, or a batch of
        :code:`M` structures with :code:`N` total atoms, the predictions should
        be of shape:

        .. list-table::
            :header-rows: 1

            * - Key
              - Single graph
              - Batch of graphs
            * - :code:`"energy"`
              - :code:`()`
              - :code:`(M,)`
            * - :code:`"forces"`
              - :code:`(N, 3)`
              - :code:`(N, 3)`
            * - :code:`"stress"`
              - :code:`(3, 3)`
              - :code:`(M, 3, 3)`
            * - :code:`"local_energies"`
              - :code:`(N,)`
              - :code:`(N,)`

        See :doc:`this page <../theory>` for more details, and in particular
        the convention that ``graph-pes`` uses for stresses. Use the
        :meth:`~graph_pes.graphs.operations.is_batch` function when implementing
        this method to check if the graph is batched.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        properties
            The properties to predict. Can be any combination of
            ``"energy"``, ``"forces"``, ``"stress"``, and ``"local_energies"``.
        """

    @torch.no_grad()
    def pre_fit(
        self,
        graphs: LabelledGraphDataset | Sequence[LabelledGraph],
    ):
        """
        Pre-fit the model to the training data.

        Some models require pre-fitting to the training data to set certain
        parameters. For example, the :class:`~graph_pes.models.pairwise.LennardJones`
        model uses the distribution of interatomic distances in the training data
        to set the length-scale parameter.

        In the ``graph-pes-train`` routine, this method is called before
        "normal" training begins (you can turn this off with a config option).

        This method detects the unique atomic numbers in the training data
        and registers these with all of the model's
        :class:`~graph_pes.nn.PerElementParameter`
        instances to ensure correct parameter counting.
        To implement model-specific pre-fitting, override the
        :meth:`model_specific_pre_fit` method.

        If the model has already been pre-fitted, subsequent calls to
        :meth:`pre_fit` will be ignored (and a warning will be raised).

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

            # TODO make this pre-fit process nicer
            if (
                hasattr(self, "per_element_scaling")
                and self.per_element_scaling is not None
                and "energy" in graph_batch
            ):
                from graph_pes.models.pre_fit import (
                    guess_per_element_mean_and_var,
                )

                means, variances = guess_per_element_mean_and_var(
                    graph_batch["energy"], graph_batch
                )
                for Z, var in variances.items():
                    self.per_element_scaling[Z] = torch.sqrt(torch.tensor(var))

            self._has_been_pre_fit = True
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

        Parameters
        ----------
        graphs
            The training data.
        """

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        """
        Return a list of parameters that should not be decayed during training.
        """
        return []

    def forward(self, graph: AtomicGraph) -> torch.Tensor:
        """
        The main access point for :class:`~graph_pes.core.GraphPESModel`
        instances is :meth:`~graph_pes.core.GraphPESModel.predict` (see above).

        For convenience, we alias the forward pass of the model to be the
        :meth:`~graph_pes.core.GraphPESModel.predict_energy` method.
        """
        graph = trim_edges(graph, self.cutoff.item())
        return self.predict_energy(graph)

    def __call__(self, graph: AtomicGraph) -> torch.Tensor:
        return super().__call__(graph)

    def get_all_PES_predictions(
        self, graph: AtomicGraph | AtomicGraphBatch
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

    def get_extra_state(self) -> bool:
        return self._has_been_pre_fit

    def set_extra_state(self, state: bool) -> None:
        self._has_been_pre_fit = state


class LocalEnergyModel(GraphPESModel, ABC):
    r"""
    An abstract base class for all models of the PES that:

    1. make predictions of the total energy of a structure as a sum
       of local contributions:

       .. math::
           E(\mathcal{G}) = \sum_i \varepsilon_i

       where :math:`\varepsilon_i` is the local energy of atom :math:`i`, and
       :math:`\mathcal{G}` is the atomic graph representation of the structure.
    2. make force and stress predictions as the (numerical) gradient of the
       total energy prediction.


    To create such a model, implement :meth:`predict_raw_energies`,
    which takes an :class:`~graph_pes.graphs.AtomicGraph`, or an
    :class:`~graph_pes.graphs.AtomicGraphBatch`,
    and returns a per-atom prediction of the local energy. For a simple example,
    see the :class:`~graph_pes.models.LennardJones` implementation.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model. During the forward pass, only edges
        between atoms that are closer than this distance will be considered.
    auto_scale
        Whether to automatically scale raw predictions by (learnable)
        per-element scaling factors, :math:`\sigma_{Z_i}`.
        The starting values for these parameters are calculated from the
        data passed to :meth:`~graph_pes.core.GraphPESModel.pre_fit` (typically the
        training data). If ``True``,
        :math:`\varepsilon_i = \sigma_{Z_i} \cdot \varepsilon_{\text{raw},i}`,
        where :math:`\varepsilon_i` is the per-atom local energy prediction,
        and :math:`\sigma_{Z_i}` is the scaling factor for element :math:`Z_i`.
    """  # noqa: E501

    def __init__(self, cutoff: float, auto_scale: bool):
        super().__init__(cutoff)

        # TODO: default this to 1 and make non-trainable + non-fittable
        self.per_element_scaling: PerElementParameter | None
        if auto_scale:
            self.per_element_scaling = PerElementParameter.of_length(
                1,
                default_value=1.0,
                requires_grad=True,
            )
        else:
            self.per_element_scaling = None

    @abstractmethod
    def predict_raw_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Predict the (unscaled) local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        torch.Tensor
            The per-atom local energy predictions, with shape :code:`(N,)`.
        """

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        if self.per_element_scaling is not None:
            return [self.per_element_scaling]
        return []

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[keys.LabelKey],
    ) -> dict[keys.LabelKey, torch.Tensor]:
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

        if keys.FORCES in properties:
            graph[keys._POSITIONS].requires_grad_(True)

        local_energies = self.predict_raw_energies(graph).squeeze()
        if self.per_element_scaling is not None:
            scales = self.per_element_scaling[
                graph[keys.ATOMIC_NUMBERS]
            ].squeeze()
            local_energies = local_energies * scales

        if keys.LOCAL_ENERGIES in properties:
            predictions[keys.LOCAL_ENERGIES] = local_energies

        # TODO: scale
        energy = sum_per_structure(local_energies, graph)
        if keys.ENERGY in properties:
            predictions[keys.ENERGY] = energy
        # use the autograd machinery to auto-magically
        # calculate forces and stress from the energy
        if keys.FORCES in properties:
            dE_dR = differentiate(energy, graph[keys._POSITIONS])
            predictions[keys.FORCES] = -dE_dR

        if keys.STRESS in properties:
            if has_cell(graph):
                stress = differentiate(energy, change_to_cell)
                cell_volume = torch.det(graph[keys.CELL])
                if is_batch(graph):
                    cell_volume = cell_volume.view(-1, 1, 1)
                predictions[keys.STRESS] = stress / cell_volume
            else:
                predictions[keys.STRESS] = torch.tensor(torch.nan)

        graph[keys._POSITIONS] = existing_positions
        graph[keys.CELL] = existing_cell

        # TODO: move that to differntiate? i.e. no need to keep secdon pass
        # also check grad enabled if want force or stress
        if not self.training:
            for key, value in predictions.items():
                predictions[key] = value.detach()

        return predictions
