from __future__ import annotations

import warnings

import torch
from torch import Tensor

from graph_pes.core import GraphPESModel
from graph_pes.graphs import AtomicGraph, LabelledBatch
from graph_pes.logger import logger
from graph_pes.models.pre_fit import guess_per_element_mean_and_var
from graph_pes.nn import PerElementParameter


class EnergyOffset(GraphPESModel):
    r"""
    A model that predicts the total energy as a sum of per-species offsets:

    .. math::
        E(\mathcal{G}) = \sum_i \varepsilon_{Z_i}

    where :math:`\varepsilon_{Z_i}` is the energy offset for atomic species
    :math:`Z_i`.

    With our chemistry hat on, this is equivalent to a PerfectGas model:
    there is no contribution to the total energy from the interactions between
    atoms.

    Parameters
    ----------
    fixed_values
        A dictionary of fixed energy offsets for each atomic species.
    trainable
        Whether the energy offsets are trainable parameters.
    """

    def __init__(self, offsets: PerElementParameter):
        super().__init__()
        self._offsets = offsets

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        """
        Index the energy offsets by the atomic numbers in the graph.

        Parameters
        ----------
        graph
            The atomic graph for which to predict the local energies.

        Returns
        -------
        Tensor
            The energy offsets for each atom in the graph. Shape: (n_atoms,)
        """
        return self._offsets[graph["atomic_numbers"]].squeeze()

    def __repr__(self):
        return self._offsets.__repr(alias=self.__class__.__name__)


class FixedOffset(EnergyOffset):
    """
    An :class:`EnergyOffset` model with pre-defined and fixed energy offsets
    for each element.

    Parameters
    ----------
    final_values
        A dictionary of fixed energy offsets for each atomic species.
    """

    def __init__(self, **final_values: float):
        offsets = PerElementParameter.from_dict(
            **final_values,
            requires_grad=False,
        )
        super().__init__(offsets)


class LearnableOffset(EnergyOffset):
    """
    An :class:`EnergyOffset` model with learnable energy offsets for each
    element.

    Parameters
    ----------
    initial_values
        A dictionary of initial energy offsets for each atomic species.
        Leave this empty to guess the offsets from the training data.
    """

    def __init__(self, **initial_values: float):
        offsets = PerElementParameter.from_dict(
            **initial_values,
            requires_grad=True,
        )
        super().__init__(offsets)
        self._values_were_specified = bool(initial_values)

    @torch.no_grad()
    def model_specific_pre_fit(self, graphs: LabelledBatch) -> None:
        """
        Calculate the **mean** energy offsets per element from the training data
        using linear regression.

        **Note**: this will lead to a lack of physicality in the model, since
        there is now no guarantee that the energy of an isolated atom, or the
        dissociation limit, is correctly modelled.

        Parameters
        ----------
        graphs
            The training data.
        """
        if self._values_were_specified:
            logger.debug(
                "Energy offsets were specified by the user. "
                "Skipping the calculation of mean energy offsets."
            )
            return

        if "energy" not in graphs:
            warnings.warn(
                "No energy labels found in the training data. "
                "Can't guess suitable per-element energy offsets for "
                f"{self.__class__.__name__}.",
                stacklevel=2,
            )
            return

        # use ridge regression to estimate the mean energy contribution
        # from each atomic species
        offsets, _ = guess_per_element_mean_and_var(graphs["energy"], graphs)
        for z, offset in offsets.items():
            self._offsets[z] = offset

        logger.warning(f"""
Estimated per-element energy offsets from the training data:
    {self._offsets}
This will lead to a lack of guaranteed physicality in the model, 
since the energy of an isolated atom (and hence the behaviour of 
this model in the dissociation limit) is not guaranteed to be 
correct. Use a FixedOffset energy offset model if you require
and know the reference energy offsets.""")
