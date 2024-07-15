from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import torch

from graph_pes.core import GraphPESModel
from graph_pes.data.dataset import LabelledGraphDataset
from graph_pes.graphs.graph_typing import AtomicGraph, LabelledGraph
from graph_pes.graphs.operations import to_batch
from graph_pes.models.pre_fit import guess_per_element_mean_and_var
from graph_pes.nn import PerElementParameter


class AutoScaledPESModel(GraphPESModel, ABC):
    """
    An abstract base class for all PES models implementations that are best
    suited to making raw predictions that with ~unit variance. By inheriting
    from this sub-class (as opposed to directly GraphPESModel) and implementing
    :meth:`predict_unscaled_energies` (as opposed to
    :meth:`predict_local_energies`), the model will automatically scale the
    raw predictions (unit variance) by the per-element scaling factors
    as calculated from the training data.
    """

    def __init__(self):
        super().__init__()
        self.per_element_scaling = PerElementParameter.of_length(
            1,
            default_value=1.0,
        )

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        raw_energies = self.predict_unscaled_energies(graph).squeeze()
        scales = self.per_element_scaling[graph["atomic_numbers"]].squeeze()
        return raw_energies * scales

    @abstractmethod
    def predict_unscaled_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Predict the unscaled energies for the given graph.

        Parameters
        ----------
        graph
            The input graph

        Returns
        -------
        torch.Tensor
            The unscaled, local energies with shape ``(n_atoms,)``.
        """

    @torch.no_grad()
    def pre_fit(
        self,
        graphs: LabelledGraphDataset | Sequence[LabelledGraph],
    ):
        _was_already_prefit = self._has_been_pre_fit.item()
        super().pre_fit(graphs)

        if _was_already_prefit:
            return

        if isinstance(graphs, LabelledGraphDataset):
            graphs = list(graphs)
        graph_batch = to_batch(graphs)

        # use Ridge regression to calculate standard deviations in the
        # per-element contributions to the total energy
        if "energy" in graph_batch:
            _, variances = guess_per_element_mean_and_var(
                graph_batch["energy"], graph_batch
            )
            for Z, var in variances.items():
                self.per_element_scaling[Z] = var**0.5

        else:
            model_name = self.__class__.__name__
            warnings.warn(
                f"{model_name} is not able to calculate per-element scaling "
                "factors because the training data does not contain energy "
                "labels. The model will not scale the raw predictions.",
                stacklevel=2,
            )
