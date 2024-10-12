from __future__ import annotations

import warnings

import torch

from graph_pes.core import GraphPESModel
from graph_pes.graphs import AtomicGraph, LabelledBatch, keys
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
    atoms. Hence, this model generates zero for both force and stress
    predictions.

    Parameters
    ----------
    fixed_values
        A dictionary of fixed energy offsets for each atomic species.
    trainable
        Whether the energy offsets are trainable parameters.
    """

    def __init__(self, offsets: PerElementParameter):
        super().__init__(cutoff=0)
        self._offsets = offsets

    def predict(
        self,
        graph: AtomicGraph,
        properties: list[keys.LabelKey],
        training: bool = False,
    ) -> dict[keys.LabelKey, torch.Tensor]:
        predictions: dict[keys.LabelKey, torch.Tensor] = {}

        Z = graph["atomic_numbers"]
        local_energies = self._offsets[Z].squeeze()
        if "local_energies" in properties:
            predictions["local_energies"] = local_energies
        if "energy" in properties:
            predictions["energy"] = local_energies.sum()

        if "forces" in properties:
            predictions["forces"] = torch.zeros_like(graph["_positions"])
        if "stress" in properties:
            predictions["stress"] = torch.zeros((3, 3), device=Z.device)

        return predictions

    def __repr__(self):
        return self._offsets._repr(alias=self.__class__.__name__)


class FixedOffset(EnergyOffset):
    """
    An :class:`~graph_pes.models.offsets.EnergyOffset` model with pre-defined
    and fixed energy offsets for each element.

    Parameters
    ----------
    final_values
        A dictionary of fixed energy offsets for each atomic species.

    Examples
    --------
    >>> model = FixedOffset(H=-1.3, C=-13.0)
    """

    def __init__(self, **final_values: float):
        offsets = PerElementParameter.from_dict(
            **final_values,
            requires_grad=False,
        )
        super().__init__(offsets)


class LearnableOffset(EnergyOffset):
    """
    An :class:`~graph_pes.models.offsets.EnergyOffset` model with
    learnable energy offsets for each element.

    During pre-fitting, for each element in the training data not specified by
    the user, the model will estimate the energy offset from the data using
    ridge regression (see TODO)

    Parameters
    ----------
    initial_values
        A dictionary of initial energy offsets for each atomic species.
        Leave this empty to guess the offsets from the training data.

    Examples
    --------
    Estimate all relevant energy offsets from the training data:

    >>> model = LearnableOffset()
    >>> model.pre_fit(training_data)  # estimates offsets from data

    Specify some initial values for the energy offsets:

    >>> model = LearnableOffset(H=0.0, C=-3.0)
    >>> model.pre_fit(training_data)  # estimates remaining offsets from data
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
