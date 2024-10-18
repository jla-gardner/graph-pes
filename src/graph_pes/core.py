from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence

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
    guess_per_element_mean_and_var,
    has_cell,
    is_batch,
    sum_per_structure,
    to_batch,
    trim_edges,
)
from .nn import PerElementParameter, UniformModuleDict
from .util import differentiate


class OutputInferrer(nn.Module, ABC):
    def before(self, graph: AtomicGraph):
        """
        Perform any necessary operations before the forward pass.

        Typically, this involves adding some temporary state to the graph,
        or setting up gradient tracking for some graph attributes.

        By default, this method does nothing.

        Parameters
        ----------
        graph
            The graph representation of the structure.
        """
        pass

    @abstractmethod
    def after(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graph: AtomicGraph,
    ):
        """
        Adjust the current predictions for the given ``graph`` in-place.

        Remove any temporary state that was added in the ``before`` method.

        Parameters
        ----------
        current_predictions
            The current predictions for the given ``graph``.
        graph
            The graph representation of the structure.
        """

    def pre_fit(self, graphs: LabelledBatch):
        """
        Pre-fit the output adapter to the training data.

        Parameters
        ----------
        graphs
            The training data.
        """
        pass


class InferTotalEnergy(OutputInferrer):
    def after(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graph: AtomicGraph,
    ):
        predictions[keys.ENERGY] = sum_per_structure(
            predictions[keys.LOCAL_ENERGIES],
            graph,
        )


class _InferForcesState(NamedTuple):
    positions_required_grad: bool


class InferForces(OutputInferrer):
    def __init__(self):
        super().__init__()
        self._temp_state: _InferForcesState | None = None

    def before(self, graph: AtomicGraph):
        # save temporary state
        self._temp_state = _InferForcesState(
            positions_required_grad=graph[keys._POSITIONS].requires_grad
        )

        # ensure gradients are tracked
        graph[keys._POSITIONS].requires_grad_(True)

    def after(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graph: AtomicGraph,
    ):
        # add forces to the predictions
        predictions[keys.FORCES] = -differentiate(
            predictions[keys.ENERGY], graph[keys._POSITIONS]
        )

        # remove temporary state in a TorchScript friendly way
        state = self._temp_state
        assert state is not None
        self._temp_state = None

        # make sure the positions are in the same state as
        # they were in before
        graph[keys._POSITIONS].requires_grad_(state.positions_required_grad)


class _InferStressState(NamedTuple):
    positions_required_grad: bool
    cell_required_grad: bool


class InferStress(OutputInferrer):
    def __init__(self):
        super().__init__()
        self._temp_state: _InferStressState | None = None

    def before(self, graph: AtomicGraph):
        if not has_cell(graph):
            raise ValueError("Can't predict stress without cell information.")

        existing_cell = graph[keys.CELL]

        # save temporary state
        self._temp_state = _InferStressState(
            positions_required_grad=graph[keys._POSITIONS].requires_grad,
            cell_required_grad=graph[keys.CELL].requires_grad,
        )

        # set up stress calculation
        change_to_cell = torch.zeros_like(existing_cell)
        change_to_cell.requires_grad_(True)
        symmetric_change = 0.5 * (
            change_to_cell + change_to_cell.transpose(-1, -2)
        )  # (n_structures, 3, 3) if batched, else (3, 3)
        scaling = torch.eye(3, device=existing_cell.device) + symmetric_change
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

    def after(
        self,
        predictions: dict[keys.LabelKey, torch.Tensor],
        graph: AtomicGraph,
    ):
        # use auto-grad to calculate stress
        stress = differentiate(predictions[keys.ENERGY], graph[keys.CELL])
        predictions[keys.STRESS] = stress

        # remove temporary state in a TorchScript friendly way
        state = self._temp_state
        assert state is not None
        self._temp_state = None

        # put things back to how they were before
        graph[keys.CELL].requires_grad_(state.cell_required_grad)
        graph[keys._POSITIONS].requires_grad_(state.positions_required_grad)


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all models of the PES that act on
    graph-representations (:class:`~graph_pes.graphs.AtomicGraph`)
    of atomic structures.

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
        auto_scale_local_energies: bool,
    ):
        super().__init__()

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self._has_been_pre_fit: torch.Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(0))

        if auto_scale_local_energies:
            self.local_energies_scaler = LocalEnergiesScaler()
        else:
            self.local_energies_scaler = None

        # setup up the output enhancers
        self.implemented_properties = implemented_properties
        if "local_energies" not in implemented_properties:
            raise ValueError(
                'All GraphPESModel\'s must implement a "local_energies" '
                "prediction."
            )
        self.output_inferrers: UniformModuleDict[OutputInferrer] = (
            UniformModuleDict()
        )
        if "energy" not in implemented_properties:
            self.output_inferrers["energy"] = InferTotalEnergy()
        if "forces" not in implemented_properties:
            self.output_inferrers["forces"] = InferForces()
        if "stress" not in implemented_properties:
            self.output_inferrers["stress"] = InferStress()

    @abstractmethod
    def forward(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
        """
        The model's forward pass. Generate all properties for the given graph
        that are in this model's :attr:`implemented_properties` list.
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

        # before anything, remove unnecessary edges:
        graph = trim_edges(graph, self.cutoff.item())

        # only select the relevant inferrers, in
        # last in first out order (stress and force requires energy)
        inferrers = []
        for key in "energy", "forces", "stress":
            if key in properties and key not in self.implemented_properties:
                inferrers.append(self.output_inferrers[key])

        # set up (in reverse order!)
        for inferrer in reversed(inferrers):
            inferrer.before(graph)

        # get the raw output from the model
        output = self(graph)

        # optionally scale the local energies
        if self.local_energies_scaler is not None:
            output["local_energies"] = self.local_energies_scaler(
                output["local_energies"], graph
            )

        # infer the remaining properties
        for inferrer in inferrers:
            inferrer.after(output, graph)

        # tidy up if in eval mode
        if not self.training:
            output: dict[keys.LabelKey, torch.Tensor] = {
                k: v.detach() for k, v in output.items()
            }

        # return the output
        return output

    # TODO: change to pre-fit all.
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
            for inferrer in self.output_inferrers.values():
                inferrer.pre_fit(graph_batch)

            self._has_been_pre_fit = torch.tensor(1)
            self.model_specific_pre_fit(graph_batch)

        # 3. finally, register all per-element parameters (no harm in doing this
        #    multiple times)
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

    # add type hints for mypy etc.
    def __call__(self, graph: AtomicGraph) -> dict[keys.LabelKey, torch.Tensor]:
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


class LocalEnergiesScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_element_scaling = PerElementParameter.of_length(
            1,
            default_value=1.0,
            requires_grad=True,
        )

    def forward(
        self,
        local_energies: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        scales = self.per_element_scaling[graph[keys.ATOMIC_NUMBERS]].squeeze()
        return local_energies * scales

    @torch.no_grad()
    def pre_fit(self, graphs: LabelledBatch):
        """
        Pre-fit the output adapter to the training data.

        Parameters
        ----------
        graphs
            The training data.
        """
        if "energy" not in graphs:
            warnings.warn(
                "No energy data found in training data: can't estimate "
                "per-element scaling factors for local energies.",
                stacklevel=2,
            )

        means, variances = guess_per_element_mean_and_var(
            graphs["energy"], graphs
        )
        for Z, var in variances.items():
            self.per_element_scaling[Z] = torch.sqrt(torch.tensor(var))

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        return [self.per_element_scaling]
