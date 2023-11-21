from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch, sum_per_structure
from graph_pes.transform import (
    Chain,
    Identity,
    PerSpeciesOffset,
    PerSpeciesScale,
)
from torch import nn


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all graph-based models of the PES that
    make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E = f(\mathcal{G}) = \sum_i \varepsilon(\mathcal{N}_i)

    To create such a model, implement :meth:`predict_local_energies`,
    which takes a :class:`AtomicGraph` and returns a per-atom prediction
    of the local energy. Under the hood, :class:`GraphPESModel` uses
    these predictions to compute the total energy of the structure
    (complete with a learnable per-species shift and scale) in the
    forward pass.

    .. note::
        All :class:`GraphPESModel` instances are also instances of
        :class:`torch.nn.Module`. This allows for easy optimisation
        of parameters, and automated save/load functionality.
    """

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Predict the (standardized) local energy for each atom in the structure,
        as represented by `graph`.

        Parameters
        ----------
        graph : AtomicGraph
            The graph representation of the structure.
        """

    def forward(self, graph: AtomicGraph | AtomicGraphBatch):
        """
        Predict the total energy of the structure/s.

        Parameters
        ----------
        graph : AtomicGraph
            The graph representation of the structure/s.
        """

        local_energies = self.predict_local_energies(graph).squeeze()
        local_energies = self._energy_transforms["local"](local_energies, graph)

        total_E = sum_per_structure(local_energies, graph)
        total_E = self._energy_transforms["total"](total_E, graph)

        return total_E

    def __init__(self):
        super().__init__()
        self._energy_transforms = nn.ModuleDict(
            {
                "local": Chain(
                    [PerSpeciesScale(), PerSpeciesOffset()], trainable=True
                ),
                "total": Identity(),
            }
        )

    def __repr__(self):
        # modified from torch.nn.Module.__repr__
        # changes:
        # - don't print any modules that start with _

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            if key.startswith("_"):
                continue
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __add__(self, other: GraphPESModel) -> GraphPESModel:
        return Ensemble([self, other], mean=False)


class Ensemble(GraphPESModel):
    def __init__(self, models: list[GraphPESModel], mean: bool = True):
        super().__init__()
        self.models: list[GraphPESModel] = nn.ModuleList(models)  # type: ignore
        self.mean = mean

    def predict_local_energies(self, graph: AtomicGraph | AtomicGraphBatch):
        s = sum(m.predict_local_energies(graph) for m in self.models)
        return s / len(self.models) if self.mean else s


def energy_and_forces(pes: GraphPESModel, structure: AtomicGraph):
    """
    Evaluate the `pes`  on `structure` to obtain both the
    total energy and the forces on each atom.

    Parameters
    ----------
    pes : PES
        The PES to use.
    structure : AtomicStructure
        The atomic structure to evaluate.

    Returns
    -------
    EnergyAndForces
        The energy of the structure and forces on each atom.
    """

    # TODO handle the case where isolated atoms are present
    # such that the gradient of energy wrt their positions
    # is zero.

    # use the autograd machinery to auto-magically
    # calculate forces for (almost) free
    structure._positions.requires_grad_(True)
    energy = pes(structure)
    dE_dR = torch.autograd.grad(
        energy.sum(),
        structure._positions,
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    structure._positions.requires_grad_(False)
    return dict(energy=energy.squeeze(), forces=-dE_dR)


# # if materialize_grads:
#         result = tuple(
#             output
#             if output is not None
#             else torch.zeros_like(input, requires_grad=True)
#             for (output, input) in zip(result, t_inputs)
#         # )
