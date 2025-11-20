from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Final, Sequence, final

import torch
from ase.data import chemical_symbols
from torch import nn

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, to_batch
from graph_pes.data.datasets import GraphDataset
from graph_pes.utils.logger import logger

from .utils.nn import PerElementParameter


class GraphPropertyModel(nn.Module, ABC):
    """
    Base class for all models that make predictions of arbitrary properties
    from graph input.

    :class:`~graph_pes.GraphPropertyModel` objects save various pieces of extra
    metadata to the ``state_dict`` via the
    :meth:`~graph_pes.GraphPropertyModel.get_extra_state` and
    :meth:`~graph_pes.GraphPropertyModel.set_extra_state` methods.
    If you want to save additional extra state to the ``state_dict`` of your
    model, please implement the
    :meth:`~graph_pes.GraphPropertyModel.extra_state`
    property and corresponding setter to ensure that you do not overwrite
    these extra metadata items.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model.
    implemented_properties
        The properties that the model implements in the forward pass.
    three_body_cutoff
        The cutoff radius for this model's three-body interactions,
        if applicable.
    """

    def __init__(
        self,
        cutoff: float,
        implemented_properties: list[PropertyKey],
        three_body_cutoff: float | None,
    ):
        super().__init__()

        self._GRAPH_PES_VERSION: Final[str] = "0.2.4"

        self.cutoff: torch.Tensor
        self.register_buffer("cutoff", torch.tensor(cutoff))

        self._has_been_pre_fit: torch.Tensor
        self.register_buffer("_has_been_pre_fit", torch.tensor(0))

        self.three_body_cutoff: torch.Tensor
        self.register_buffer(
            "three_body_cutoff", torch.tensor(three_body_cutoff or 0)
        )

        self.implemented_properties = implemented_properties

    @abstractmethod
    def forward(self):
        """
        The model's forward pass. Generate all properties for the given graph
        that are in this model's ``implemented_properties`` list.
        """
        ...

    @abstractmethod
    def predict(self, graph: AtomicGraph, properties: list[PropertyKey]):
        """
        Generate (optionally batched) predictions for the given
        ``properties`` and  ``graph``.

        This method returns a dictionary mapping each requested
        ``property`` to a tensor of predictions, relying on the model's
        :meth:`~graph_pes.GraphPropertyModel.forward` implementation
        together with :func:`torch.autograd.grad` to automatically infer any
        missing properties.
        """
        pass

    # add type hints for mypy etc.
    def __call__(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        return super().__call__(graph)

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
           (including itself) and calls their :meth:`pre_fit` method (if it exists -
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
        :meth:`~graph_pes.GraphPropertyModel.extra_state` property to add extra
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
        to values returned by the
        :meth:`~graph_pes.GraphPropertyModel.extra_state` property setter to add
        extra state here.
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


class GraphTensorModel(GraphPropertyModel):
    r"""
    Base class for all models that make predictions of
    arbitrary rank, per-atom tensors from graph input.

    Parameters
    ----------
    cutoff
        The cutoff radius for the model.
    implemented_properties
        The property predictions that the model implements in the forward pass.
        Must include ``"tensor"``.
    """

    def __init__(
        self,
        cutoff: float,
        implemented_properties: list[PropertyKey],
    ):
        super().__init__(
            cutoff=cutoff,
            implemented_properties=implemented_properties,
            three_body_cutoff=None,
        )

    @abstractmethod
    def forward(self):
        pass

    def predict(self, graph: AtomicGraph, properties: list[PropertyKey]):
        predictions = self(graph)
        predictions = {prop: predictions[prop] for prop in properties}

        # tidy up if in eval mode
        if not self.training:
            predictions = {k: v.detach() for k, v in predictions.items()}

        return predictions
