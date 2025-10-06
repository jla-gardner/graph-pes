from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final, Sequence, final

import torch
from ase.data import chemical_symbols
from torch import nn

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    has_cell,
    is_batch,
    replace,
    sum_per_structure,
    to_batch,
    trim_edges,
)
from graph_pes.data.datasets import GraphDataset
from graph_pes.utils.logger import logger

from .utils.misc import differentiate, differentiate_all
from .utils.nn import PerElementParameter

if TYPE_CHECKING:
    from graph_pes.utils.calculator import GraphPESCalculator


class GraphPESModel(nn.Module, ABC):
    r"""
    All models implemented in ``graph-pes`` are subclasses of
    :class:`~graph_pes.GraphPESModel`.

    These models make predictions (via the
    :meth:`~graph_pes.GraphPESModel.predict` method) of the
    following properties:

    .. list-table::
            :header-rows: 1

            * - Key
              - Single graph
              - Batch of graphs
              - Units
            * - :code:`"local_energies"`
              - :code:`(N,)`
              - :code:`(N,)`
              - :code:`[energy]`
            * - :code:`"energy"`
              - :code:`()`
              - :code:`(M,)`
              - :code:`[energy]`
            * - :code:`"forces"`
              - :code:`(N, 3)`
              - :code:`(N, 3)`
              - :code:`[energy / length]`
            * - :code:`"stress"`
              - :code:`(3, 3)`
              - :code:`(M, 3, 3)`
              - :code:`[energy / length^3]`
            * - :code:`"virial"`
              - :code:`(3, 3)`
              - :code:`(M, 3, 3)`
              - :code:`[energy]`

    assuming an input of an :class:`~graph_pes.AtomicGraph` representing a
    single structure composed of ``N`` atoms, or an
    :class:`~graph_pes.AtomicGraph` composed of ``M`` structures and containing
    a total of ``N`` atoms. (see :func:`~graph_pes.atomic_graph.is_batch` for
    more information about batching).

    Note that ``graph-pes`` makes no assumptions as to the actual units of
    the ``energy`` and ``length`` quantities - these will depend on the
    labels the model has been trained on (e.g. could be ``eV`` and ``Å``,
    ``kcal/mol`` and ``nm`` or even ``J`` and ``m``).

    Implementations must override the
    :meth:`~graph_pes.GraphPESModel.forward` method to generate a
    dictionary of predictions for the given graph. As a minimum, this must
    include a per-atom energy contribution (``"local_energies"``).

    For any other properties not returned by the forward pass,
    the :meth:`~graph_pes.GraphPESModel.predict` method will automatically
    infer these properties from the local energies as required:

    * ``"energy"``: as the sum of the local energies per structure.
    * ``"forces"``: as the negative gradient of the energy with respect to the
      atomic positions.
    * ``"stress"``: as the negative gradient of the energy with respect to a
      symmetric expansion of the unit cell, normalised by the cell volume.
      In keeping with convention, a negative stress indicates the system is
      under static compression (wants to expand).
    * ``"virial"``: as ``-stress * volume``. A negative virial indicates the
      system is under static tension (wants to contract).

    For more details on how these are calculated, see :doc:`../theory`.

    :class:`~graph_pes.GraphPESModel` objects save various peices of extra
    metadata to the ``state_dict`` via the
    :meth:`~graph_pes.GraphPESModel.get_extra_state` and
    :meth:`~graph_pes.GraphPESModel.set_extra_state` methods.
    If you want to save additional extra state to the ``state_dict`` of your
    model, please implement the :meth:`~graph_pes.GraphPESModel.extra_state`
    property and corresponding setter to ensure that you do not overwrite
    these extra metadata items.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model.
    implemented_properties
        The property predictions that the model implements in the forward pass.
        Must include at least ``"local_energies"``.
    three_body_cutoff
        The cutoff radius for this model's three-body interactions, if
        applicable.
    """

    def __init__(
        self,
        cutoff: float,
        implemented_properties: list[PropertyKey],
        three_body_cutoff: float | None = None,
    ):
        super().__init__()

        self._GRAPH_PES_VERSION: Final[str] = "0.2.0"

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.three_body_cutoff: torch.Tensor
        self.register_buffer(
            "three_body_cutoff", torch.tensor(three_body_cutoff or 0)
        )
        self._has_been_pre_fit: torch.Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(0))

        self.implemented_properties = implemented_properties
        if "local_energies" not in implemented_properties:
            raise ValueError(
                'All GraphPESModel\'s must implement a "local_energies" '
                "prediction."
            )

    @abstractmethod
    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        """
        The model's forward pass. Generate all properties for the given graph
        that are in this model's ``implemented_properties`` list.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.

        Returns
        -------
        dict[PropertyKey, torch.Tensor]
            A dictionary mapping each implemented property to a tensor of
            predictions (see above for the expected shapes). Use
            :func:`~graph_pes.atomic_graph.is_batch` to check if the
            graph is batched in the forward pass.
        """
        ...

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[PropertyKey],
    ) -> dict[PropertyKey, torch.Tensor]:
        """
        Generate (optionally batched) predictions for the given
        ``properties`` and  ``graph``.

        This method returns a dictionary mapping each requested
        ``property`` to a tensor of predictions, relying on the model's
        :meth:`~graph_pes.GraphPESModel.forward` implementation
        together with :func:`torch.autograd.grad` to automatically infer any
        missing properties.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        properties
            The properties to predict. Can be any combination of
            ``"energy"``, ``"forces"``, ``"stress"``, ``"virial"``, and
            ``"local_energies"``.
        """

        # before anything, remove unnecessary edges:
        graph = trim_edges(graph, self.cutoff.item())

        # and prevent altering the original graph
        graph = replace(
            graph,
            R=graph.R.clone().detach(),
            cell=graph.cell.clone().detach(),
        )

        # check to see if we need to infer any properties
        infer_forces = (
            "forces" in properties
            and "forces" not in self.implemented_properties
        )
        infer_stress_information = (
            # we need to infer stress information if we're asking for
            # the stress or the virial and the model doesn't
            # implement either of them direct
            ("stress" in properties or "virial" in properties)
            and "stress" not in self.implemented_properties
            and "virial" not in self.implemented_properties
        )
        if infer_stress_information and not has_cell(graph):
            raise ValueError("Can't predict stress without cell information.")
        infer_energy = (
            any([infer_stress_information, infer_forces])
            or "energy" not in self.implemented_properties
        )

        # inference specific set up
        if infer_stress_information:
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

            change_to_cell = torch.zeros_like(graph.cell)
            change_to_cell.requires_grad_(True)
            symmetric_change = 0.5 * (
                change_to_cell + change_to_cell.transpose(-1, -2)
            )  # (n_structures, 3, 3) if batched, else (3, 3)
            scaling = torch.eye(3, device=graph.cell.device) + symmetric_change

            # torchscript annoying-ness:
            graph_batch = graph.batch
            if graph_batch is not None:
                scaling_per_atom = torch.index_select(
                    scaling,
                    dim=0,
                    index=graph_batch,
                )  # (n_atoms, 3, 3)

                # to go from (N, 3) @ (N, 3, 3) -> (N, 3), we need un/squeeze:
                # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3) -> (N, 3)
                new_positions = (
                    graph.R.unsqueeze(-2) @ scaling_per_atom
                ).squeeze()
                # (M, 3, 3) @ (M, 3, 3) -> (M, 3, 3)
                new_cell = graph.cell @ scaling

            else:
                # (N, 3) @ (3, 3) -> (N, 3)
                new_positions = graph.R @ scaling
                new_cell = graph.cell @ scaling

            # change to positions will be a tensor of all 0's, but will allow
            # gradients to flow backwards through the energy calculation
            # and allow us to calculate the stress tensor as the gradient
            # of the energy wrt the change in cell.

            graph = replace(graph, R=new_positions, cell=new_cell)

        else:
            change_to_cell = torch.zeros_like(graph.cell)

        if infer_forces:
            graph.R.requires_grad_(True)

        # get the implemented properties
        predictions = self(graph)

        if infer_energy:
            if "local_energies" not in predictions:
                raise ValueError("Can't infer energy without local energies.")

            predictions["energy"] = sum_per_structure(
                predictions["local_energies"],
                graph,
            )

        # use the autograd machinery to auto-magically
        # calculate forces and stress from the energy

        cell_volume = torch.det(graph.cell)
        if is_batch(graph):
            cell_volume = cell_volume.view(-1, 1, 1)

        # ugly triple if loops to be efficient with autograd
        # while also Torchscript compatible
        if infer_forces and infer_stress_information:
            assert "energy" in predictions
            dE_dR, dE_dC = differentiate_all(
                predictions["energy"],
                [graph.R, change_to_cell],
                keep_graph=self.training,
            )
            predictions["forces"] = -dE_dR
            predictions["virial"] = -dE_dC
            predictions["stress"] = dE_dC / cell_volume

        elif infer_forces:
            assert "energy" in predictions
            dE_dR = differentiate(
                predictions["energy"],
                graph.R,
                keep_graph=self.training,
            )
            predictions["forces"] = -dE_dR
        elif infer_stress_information:
            assert "energy" in predictions
            dE_dC = differentiate(
                predictions["energy"],
                change_to_cell,
                keep_graph=self.training,
            )
            predictions["virial"] = -dE_dC
            predictions["stress"] = dE_dC / cell_volume

        # finally, we might not have needed autograd to infer stress/virial
        # if the other was implemented on the base class:
        if not infer_stress_information:
            if (
                "stress" in properties
                and "stress" not in self.implemented_properties
            ):
                predictions["stress"] = -predictions["virial"] / cell_volume
            if (
                "virial" in properties
                and "virial" not in self.implemented_properties
            ):
                predictions["virial"] = -predictions["stress"] * cell_volume

        # make sure we don't leave auxiliary predictions
        # e.g. local_energies if we only asked for energy
        #      or energy if we only asked for forces
        predictions = {prop: predictions[prop] for prop in properties}

        # tidy up if in eval mode
        if not self.training:
            predictions = {k: v.detach() for k, v in predictions.items()}

        # return the output
        return predictions  # type: ignore

    @torch.no_grad()
    def pre_fit_all_components(
        self,
        graphs: Sequence[AtomicGraph],
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
           (inlcuding itself) and calls their :meth:`pre_fit` method (if it exists -
           see for instance :class:`~graph_pes.models.LearnableOffset` for
           an example of a model-specific pre-fit method, and
           :class:`~graph_pes.models.components.scaling.LocalEnergiesScaler` for
           an example of a component-specific pre-fit method).
        2. registers all the unique atomic numbers in the training data with
           all of the model's :class:`~graph_pes.utils.nn.PerElementParameter`
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
        if isinstance(graphs, GraphDataset):
            graphs = list(graphs)
        graph_batch = to_batch(graphs)

        # 2a. if the graph has already been pre-fitted: warn
        if self._has_been_pre_fit:
            model_name = self.__class__.__name__
            logger.warning(
                f"This model ({model_name}) has already been pre-fitted. "
                "This, and any subsequent, call to pre_fit_all_components will "
                "be ignored.",
                stacklevel=2,
            )

        # 2b. if the model has not been pre-fitted: pre-fit
        else:
            if len(graphs) > 10_000:
                logger.warning(
                    f"Pre-fitting on a large dataset ({len(graphs):,} graphs). "
                    "This may take some time. Consider using a smaller, "
                    "representative collection of structures for pre-fitting. "
                    "Set ``max_n_pre_fit`` in your config, or "
                    "see GraphDataset.sample() for more information.",
                    stacklevel=2,
                )

            self.pre_fit(graph_batch)
            # pre-fit any sub-module with a pre_fit method
            for module in self.modules():
                if hasattr(module, "pre_fit") and module is not self:
                    module.pre_fit(graph_batch)

            self._has_been_pre_fit = torch.tensor(1)

        # 3. finally, register all per-element parameters (no harm in doing this
        #    multiple times)
        for param in self.parameters():
            if isinstance(param, PerElementParameter):
                param.register_elements(torch.unique(graph_batch.Z).tolist())

    def pre_fit(self, graphs: AtomicGraph) -> None:
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
    def __call__(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        return super().__call__(graph)

    def get_all_PES_predictions(
        self, graph: AtomicGraph
    ) -> dict[PropertyKey, torch.Tensor]:
        """
        Get all the properties that the model can predict
        for the given ``graph``.
        """
        properties: list[PropertyKey] = [
            "energy",
            "forces",
            "local_energies",
        ]
        if has_cell(graph):
            properties.extend(["stress", "virial"])
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

    def predict_virial(self, graph: AtomicGraph) -> torch.Tensor:
        """Convenience method to predict just the virial."""
        return self.predict(graph, ["virial"])["virial"]

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

    @torch.jit.unused
    @property
    def device(self) -> torch.device:
        return self.cutoff.device

    @torch.jit.unused
    @final
    def get_extra_state(self) -> dict[str, Any]:
        """
        Get the extra state of this instance. Please override the
        :meth:`~graph_pes.GraphPESModel.extra_state` property to add extra
        state here.
        """
        return {
            "_GRAPH_PES_VERSION": self._GRAPH_PES_VERSION,
            "extra": self.extra_state,
        }

    @torch.jit.unused
    @final
    def set_extra_state(self, state: dict[str, Any]) -> None:  # type: ignore
        """
        Set the extra state of this instance using a dictionary mapping strings
        to values returned by the :meth:`~graph_pes.GraphPESModel.extra_state`
        property setter to add extra state here.
        """
        version = state.pop("_GRAPH_PES_VERSION", None)
        if version is not None:
            current_version = self._GRAPH_PES_VERSION
            if version != current_version:
                warnings.warn(
                    "You are attempting to load a state dict corresponding "
                    f"to graph-pes version {version}, but the current version "
                    f"of this model is {current_version}. This may cause "
                    "errors when loading the model.",
                    stacklevel=2,
                )
            self._GRAPH_PES_VERSION = version  # type: ignore

        # user defined extra state
        self.extra_state = state["extra"]

    @torch.jit.unused
    @property
    def extra_state(self) -> Any:
        """
        Override this property to add extra state to the model's
        ``state_dict``.
        """
        return {}

    @torch.jit.unused
    @extra_state.setter
    def extra_state(self, state: Any) -> None:
        """
        Set the extra state of this instance using a value returned by the
        :meth:`~graph_pes.GraphPESModel.extra_state` property.
        """
        pass

    @torch.jit.unused
    def ase_calculator(
        self, device: torch.device | str | None = None, skin: float = 1.0
    ) -> "GraphPESCalculator":
        """
        Return an ASE calculator wrapping this model. See
        :class:`~graph_pes.utils.calculator.GraphPESCalculator` for more
        information.

        Parameters
        ----------
        device
            The device to use for the calculator. If ``None``, the device of the
            model will be used.
        skin
            The skin to use for the neighbour list. If all atoms have moved less
            than half of this distance between calls to `calculate`, the
            neighbour list will be reused, saving (in some cases) significant
            computation time.
        """
        from graph_pes.utils.calculator import GraphPESCalculator

        return GraphPESCalculator(self, device=device, skin=skin)

    @torch.jit.unused
    def torch_sim_model(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
    ):
        """
        Return a model suitable for use with the
        `torch_sim <https://github.com/Radical-AI/torch-sim>`__ package.

        Internally, we set this model to evaluation mode, and wrap it in a
        class that is suitable for use with the ``torch_sim`` package.

        Parameters
        ----------
        device
            The device to use for the model. If ``None``, the model will be
            placed on the best device available.
        dtype
            The dtype to use for the model.
        compute_forces
            Whether to compute forces. Set this to ``False`` if you only need
            to generate energies within the ``torch_sim`` integrator.
        compute_stress
            Whether to compute stress. Set this to ``False`` if you don't
            need stress information from the model within the ``torch_sim``
            integrator.
        """
        import importlib.util

        if importlib.util.find_spec("torch_sim") is None:
            raise ImportError(
                "torch_sim is not installed. Please install it using "
                "pip install torch-sim-atomistic"
            )
        from torch_sim.models.graphpes import GraphPESWrapper

        return GraphPESWrapper(
            self.eval(),
            device=device,
            dtype=dtype,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
        )
