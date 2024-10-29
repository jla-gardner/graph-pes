from __future__ import annotations

import warnings

import torch
from ase.data import atomic_numbers

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.logger import logger
from graph_pes.utils.nn import PerElementParameter
from graph_pes.utils.shift_and_scale import guess_per_element_mean_and_var


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
    offsets
        The energy offsets for each atomic species.
    """

    def __init__(self, offsets: PerElementParameter):
        super().__init__(
            cutoff=0,
            implemented_properties=["local_energies"],
        )
        self._offsets = offsets

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        return {
            "local_energies": self._offsets[graph.Z].squeeze(),
        }

    def non_decayable_parameters(self) -> list[torch.nn.Parameter]:
        """The ``_offsets`` parameter should not be decayed."""
        return [self._offsets]

    def __repr__(self):
        return self._offsets._repr(alias=self.__class__.__name__)


class FixedOffset(EnergyOffset):
    """
    An :class:`~graph_pes.models.offsets.EnergyOffset` model with pre-defined
    and fixed energy offsets for each element. These do not change during
    training. Any element not specified in the ``final_values`` argument will
    be assigned an energy offset of zero.

    Parameters
    ----------
    final_values
        A dictionary mapping element symbols to fixed energy offset values.

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
    ridge regression (see
    :func:`~graph_pes.utils.shift_and_scale.guess_per_element_mean_and_var`).

    Parameters
    ----------
    initial_values
        A dictionary of initial energy offsets for each atomic species.
        Leave this empty to guess the offsets from the training data.

    Examples
    --------
    Estimate all relevant energy offsets from the training data:

    >>> model = LearnableOffset()
    >>> # estimates offsets from data
    >>> model.pre_fit_all_components(training_data)

    Specify some initial values for the energy offsets:

    >>> model = LearnableOffset(H=0.0, C=-3.0)
    >>> # estimate offsets for elements that aren't C or H
    >>> model.pre_fit_all_components(training_data)
    """

    def __init__(self, **initial_values: float):
        offsets = PerElementParameter.from_dict(
            **initial_values,
            requires_grad=True,
        )
        super().__init__(offsets)

        Zs = torch.tensor(
            [atomic_numbers[symbol] for symbol in initial_values],
            dtype=torch.long,
        )
        self.register_buffer("_pre_specified_Zs", Zs)

    @torch.no_grad()
    def pre_fit(self, graphs: AtomicGraph) -> None:
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

        if "energy" not in graphs.properties:
            warnings.warn(
                "No energy labels found in the training data. "
                "Can't guess suitable per-element energy offsets for "
                f"{self.__class__.__name__}.",
                stacklevel=2,
            )
            return

        # use ridge regression to estimate the mean energy contribution
        # from each atomic species
        offsets, _ = guess_per_element_mean_and_var(
            graphs.properties["energy"], graphs
        )
        for z, offset in offsets.items():
            if torch.any(self._pre_specified_Zs == z):
                continue
            self._offsets[z] = offset

        logger.warning(f"""
Estimated per-element energy offsets from the training data:
    {self._offsets}
This will lead to a lack of guaranteed physicality in the model, 
since the energy of an isolated atom (and hence the behaviour of 
this model in the dissociation limit) is not guaranteed to be 
correct. Use a FixedOffset energy offset model if you require
and know the reference energy offsets.""")
