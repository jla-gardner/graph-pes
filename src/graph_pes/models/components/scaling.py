from __future__ import annotations

import warnings

import torch
from torch import nn

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.nn import PerElementParameter
from graph_pes.utils.shift_and_scale import guess_per_element_mean_and_var


class LocalEnergiesScaler(nn.Module):
    """
    Scale the local energies by a per-element scaling factor.

    See :func:`~graph_pes.utils.shift_and_scale.guess_per_element_mean_and_var`
    for how the scaling factors are estimated from the training data.
    """

    def __init__(self):
        super().__init__()
        self.per_element_scaling = PerElementParameter.of_length(
            1,
            default_value=1.0,
            requires_grad=True,
        )
        """
        The per-element scaling factors. 
        (:class:`~graph_pes.utils.nn.PerElementParameter`)
        """

    def forward(
        self,
        local_energies: torch.Tensor,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        """
        Scale the local energies by the per-element scaling factor.
        """
        scales = self.per_element_scaling[graph.Z].squeeze()
        return local_energies.squeeze() * scales

    # add typing for mypy etc
    def __call__(
        self, local_energies: torch.Tensor, graph: AtomicGraph
    ) -> torch.Tensor:
        return super().__call__(local_energies, graph)

    @torch.no_grad()
    def pre_fit(self, graphs: AtomicGraph):
        """
        Pre-fit the per-element scaling factors.

        Parameters
        ----------
        graphs
            The training data.
        """
        if "energy" not in graphs.properties:
            warnings.warn(
                "No energy data found in training data: can't estimate "
                "per-element scaling factors for local energies.",
                stacklevel=2,
            )
            return

        means, variances = guess_per_element_mean_and_var(
            graphs.properties["energy"], graphs
        )
        for Z, var in variances.items():
            self.per_element_scaling[Z] = torch.sqrt(torch.tensor(var))

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        """The ``per_element_scaling`` parameter should not be decayed."""
        return [self.per_element_scaling]

    def __repr__(self):
        return self.per_element_scaling._repr(alias=self.__class__.__name__)
