from __future__ import annotations

import time
import warnings
from typing import Iterable, TypeVar, overload

import ase
import numpy
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from graph_pes.atomic_graph import AtomicGraph, PropertyKey, to_batch
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import groups_of, pairs, uniform_repr


def iterate_over_batches(
    structures: Iterable[AtomicGraph | ase.Atoms],
    batch_size: int,
    model: GraphPESModel,
) -> Iterable[AtomicGraph]:
    for group in groups_of(batch_size, structures):
        graphs = [
            AtomicGraph.from_ase(s, model.cutoff.item() + 0.001)
            if isinstance(s, ase.Atoms)
            else s
            for s in group
        ]
        yield to_batch(graphs).to(model.device)


class GraphPESCalculator(Calculator):
    """
    ASE calculator wrapping any :class:`~graph_pes.GraphPESModel`.

    Implements a neighbour list caching scheme (see below) controlled by
    the ``skin`` parameter. Using ``skin > 0.0`` will

    * accelerate MD and minimisations
    * slow down single point calculations

    If you are predomintantly doing single point calculations, use
    ``skin=0``, otherwise, tune the ``skin`` paramter for your use case
    (see below).

    Parameters
    ----------
    model
        The model to wrap
    device
        The device to use for the calculation, e.g. "cpu" or "cuda".
        Defaults to ``None``, in which case the model is not moved
        from its current device.
    skin
        The additional skin to use for neighbour list calculations.
        If all atoms have moved less than half of this distance between
        calls to `calculate`, the neighbour list will be reused, saving
        (in some cases) significant computation time.
    **kwargs
        Properties passed to the :class:`ase.calculators.calculator.Calculator`
        base class.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: GraphPESModel,
        device: torch.device | str | None = None,
        skin: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        device = model.device if device is None else device
        self.model = model.to(device)
        self.model.eval()

        # caching for accelerated MD / calculation
        self._cached_graph: AtomicGraph | None = None
        self._cached_R: numpy.ndarray | None = None
        self._cached_cell: numpy.ndarray | None = None
        self.skin = skin

        # stats
        self.cache_hits = 0
        self.total_calls = 0
        self.nl_timings = []

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | list[PropertyKey] | None = None,
        system_changes: list[str] = all_changes,
    ):
        """
        Calculate the requested properties for the given structure, and store
        them to ``self.results``, as per a normal
        :class:`ase.calculators.calculator.Calculator`.

        Underneath-the-hood, this uses a neighbour list cache to speed up
        repeated calculations on the similar structures (i.e. particularly
        effective for MD and relaxations).
        """
        # handle defaults
        if properties is None:
            properties = ["energy", "forces"]

        # call to base-class to ensure setting of atoms attribute
        super().calculate(atoms, properties, system_changes)
        assert isinstance(self.atoms, ase.Atoms)

        self.total_calls += 1

        graph: AtomicGraph | None = None

        # avoid re-calculating neighbour lists if possible
        if (
            set(system_changes) <= {"positions", "cell"}
            and self._cached_graph is not None
            and self._cached_R is not None
            and self._cached_cell is not None
        ):
            new_R = self.atoms.positions
            new_cell = self.atoms.cell.array
            changes = numpy.linalg.norm(new_R - self._cached_R, axis=-1)
            cell_changes = numpy.linalg.norm(
                new_cell - self._cached_cell, axis=-1
            )
            # cache hit?
            if numpy.all(changes < self.skin / 2) and numpy.all(
                cell_changes < self.skin / 2
            ):
                self.cache_hits += 1
                graph = self._cached_graph._replace(
                    R=torch.tensor(new_R, dtype=self._cached_graph.R.dtype),
                    cell=torch.tensor(
                        new_cell, dtype=self._cached_graph.cell.dtype
                    ),
                )

        # cache miss
        if graph is None:
            tick = time.perf_counter()
            graph = AtomicGraph.from_ase(
                self.atoms, self.model.cutoff.item() + self.skin
            ).to(self.model.device)
            tock = time.perf_counter()
            self.nl_timings.append(tock - tick)
            self._cached_graph = graph
            self._cached_R = graph.R.detach().cpu().numpy()
            self._cached_cell = graph.cell.detach().cpu().numpy()

        graph = graph.to(self.model.device)

        results = {
            k: v.detach().cpu().numpy()
            for k, v in self.model.predict(
                graph,
                properties=properties,  # type: ignore
            ).items()
            if k in properties
        }
        if "energy" in properties:
            results["energy"] = results["energy"].item()
        if "stress" in properties:
            results["stress"] = full_3x3_to_voigt_6_stress(results["stress"])

        self.results = results

    @property
    def cache_hit_rate(self) -> float:
        """
        The ratio of calls to
        :meth:`~graph_pes.utils.calculator.GraphPESCalculator.calculate`
        for which the neighbour list was reused.
        """
        if self.total_calls == 0:
            warnings.warn("No calls to calculate yet", stacklevel=2)
            return 0.0
        return self.cache_hits / self.total_calls

    @property
    def average_nl_timing(self) -> float:
        """The average time taken to calculate the neighbour list in seconds."""
        return numpy.mean(self.nl_timings).item()

    @property
    def total_nl_timing(self) -> float:
        """The total time taken to calculate the neighbour list in seconds."""
        return sum(self.nl_timings)

    def reset_cache_stats(self):
        """Reset the :attr:`cache_hit_rate` statistic."""
        self.cache_hits = 0
        self.total_calls = 0
        self.nl_timings = []

    def calculate_all(
        self,
        structures: Iterable[AtomicGraph | ase.Atoms],
        properties: list[PropertyKey] | None = None,
        batch_size: int = 5,
    ) -> list[dict[PropertyKey, numpy.ndarray | float]]:
        """
        Semantically identical to:

        .. code-block::

            [calc.calculate(structure, properties) for structure in structures]

        but with significant acceleration due to internal batching.

        Parameters
        ----------
        structures
            A list of :class:`~graph_pes.AtomicGraph` or
            :class:`ase.Atoms` objects.
        properties
            The properties to predict.
        batch_size
            The number of structures to predict at once.
        """
        # Convert structures to graphs and handle defaults
        if properties is None:
            properties = ["energy", "forces"]

        # Batched prediction
        tensor_results: list[dict[PropertyKey, torch.Tensor]] = []
        for batch in iterate_over_batches(structures, batch_size, self.model):
            predictions: dict[PropertyKey, torch.Tensor] = {
                k: v.detach().cpu()
                for k, v in self.model.predict(batch, properties).items()
            }
            separated = _seperate(predictions, batch)
            tensor_results.extend(separated)

        results: list[dict[PropertyKey, numpy.ndarray | float]] = [
            {k: to_numpy(v) for k, v in r.items()} for r in tensor_results
        ]

        # Convert stress tensors to Voigt notation
        for r in results:
            for key in ["stress", "virial"]:
                if key in r:
                    r[key] = full_3x3_to_voigt_6_stress(r[key])

        return results

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            model=self.model,
            device=self.model.device,
            skin=self.skin,
        )


## utils ##

T = TypeVar("T")
TensorLike = TypeVar("TensorLike", torch.Tensor, numpy.ndarray)


def to_numpy(t: torch.Tensor) -> numpy.ndarray | float:
    n = t.detach().cpu().numpy()
    return n.item() if n.shape == () else n


def _seperate(
    batched_prediction: dict[PropertyKey, TensorLike],
    batch: AtomicGraph,
) -> list[dict[PropertyKey, TensorLike]]:
    preds_list = []
    assert batch.ptr is not None

    for idx, (start, stop) in enumerate(pairs(batch.ptr)):
        preds = {}

        # per-structure properties
        if "energy" in batched_prediction:
            preds["energy"] = batched_prediction["energy"][idx]
        if "stress" in batched_prediction:
            preds["stress"] = batched_prediction["stress"][idx]
        if "virial" in batched_prediction:
            preds["virial"] = batched_prediction["virial"][idx]

        # per-atom properties
        if "forces" in batched_prediction:
            preds["forces"] = batched_prediction["forces"][start:stop]
        if "local_energies" in batched_prediction:
            preds["local_energies"] = batched_prediction["local_energies"][
                start:stop
            ]

        preds_list.append(preds)

    return preds_list


@overload
def merge_predictions(  # type: ignore
    predictions: list[dict[PropertyKey, numpy.ndarray | float]],
) -> dict[PropertyKey, numpy.ndarray]: ...
@overload
def merge_predictions(
    predictions: list[dict[PropertyKey, torch.Tensor | float]],
) -> dict[PropertyKey, torch.Tensor]: ...
def merge_predictions(
    predictions: list[dict[PropertyKey, TensorLike | float]],
) -> dict[PropertyKey, TensorLike]:
    """
    Take a list of property predictions and merge them
    in a sensible way. Implemented for both :class:`torch.Tensor`
    and :class:`numpy.ndarray`.

    Parameters
    ----------
    predictions
        A list of property predictions each corresponding to a single
        structure.

    Examples
    --------
    >>> predictions = [
    ...     {"energy": np.array(1.0), "forces": np.array([[1, 2], [3, 4]])},
    ...     {"energy": np.array(2.0), "forces": np.array([[5, 6], [7, 8]])},
    ... ]
    >>> merge_predictions(predictions)
    {'energy': array([1., 2.]), 'forces': array([[1, 2], [3, 4], [5, 6], [7, 8]])}
    """  # noqa: E501
    if not predictions:
        return {}

    eg = next(iter(predictions[0].values()))
    if isinstance(eg, torch.Tensor):
        stack = torch.stack
        cat = torch.cat
    else:
        stack = numpy.stack
        cat = numpy.concatenate

    merged: dict[PropertyKey, TensorLike] = {}

    # stack per-structure properties along new axis
    for key in ["energy", "stress"]:
        if key in predictions[0]:
            merged[key] = stack([p[key] for p in predictions])  # type: ignore

    # concatenat per-atom properties along the first axis
    for key in ["forces", "local_energies"]:
        if key in predictions[0]:
            merged[key] = cat([p[key] for p in predictions])  # type: ignore

    return merged
