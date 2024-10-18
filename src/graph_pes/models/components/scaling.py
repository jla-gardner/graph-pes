from __future__ import annotations

import warnings

import torch
from graph_pes.graphs import AtomicGraph, LabelledBatch, keys
from graph_pes.graphs.operations import guess_per_element_mean_and_var
from graph_pes.nn import PerElementParameter
from torch import nn


class LocalEnergiesScaler(nn.Module):
    """
    Scale the local energies by a per-element scaling factor.

    See :func:`graph_pes.graphs.operations.guess_per_element_mean_and_var` for
    how the scaling factors are estimated from the training data.
    """

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

    # add typing for mypy etc
    def __call__(
        self, local_energies: torch.Tensor, graph: AtomicGraph
    ) -> torch.Tensor:
        return super().__call__(local_energies, graph)

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
            return

        means, variances = guess_per_element_mean_and_var(
            graphs["energy"], graphs
        )
        for Z, var in variances.items():
            self.per_element_scaling[Z] = torch.sqrt(torch.tensor(var))

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        return [self.per_element_scaling]
