from __future__ import annotations

import warnings
from typing import Iterable, TypeVar

import ase
import numpy
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from graph_pes.atomic_graph import AtomicGraph, PropertyKey, has_cell, to_batch
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import groups_of, pairs


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

        # cache stats
        self.cache_hits = 0
        self.total_calls = 0

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
            # cache hit
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
            graph = AtomicGraph.from_ase(
                self.atoms, self.model.cutoff.item() + self.skin
            ).to(self.model.device)
            self._cached_graph = graph
            self._cached_R = graph.R.detach().cpu().numpy()
            self._cached_cell = graph.cell.detach().cpu().numpy()

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

    def reset_cache_stats(self):
        """Reset the :attr:`cache_hit_rate` statistic."""
        self.cache_hits = 0
        self.total_calls = 0

    def calculate_all(
        self,
        structures: Iterable[AtomicGraph | ase.Atoms],
        properties: list[PropertyKey] | None = None,
        batch_size: int = 5,
    ) -> list[dict[PropertyKey, numpy.ndarray]]:
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

        Examples
        --------
        >>> calculator = GraphPESCalculator(model, device="cuda")
        >>> structures = [Atoms(...), Atoms(...), Atoms(...)]
        >>> predictions = calculator.calculate_all(
        ...     structures,
        ...     properties=["energy", "forces"],
        ...     batch_size=2,
        ... )
        >>> print(predictions)
        [{'energy': array(...), 'forces': array(...)},
         {'energy': array(...), 'forces': array(...)},
         {'energy': array(...), 'forces': array(...)}]
        """

        _, tensor_results = self._calculate_all_keep_tensor(
            structures, properties, batch_size
        )

        results = [to_numpy(r) for r in tensor_results]

        if "stress" in results[0]:
            for r in results:
                r["stress"] = full_3x3_to_voigt_6_stress(r["stress"])

        return results

    def _calculate_all_keep_tensor(
        self,
        structures: Iterable[AtomicGraph | ase.Atoms],
        properties: list[PropertyKey] | None = None,
        batch_size: int = 5,
    ) -> tuple[
        list[AtomicGraph],
        list[dict[PropertyKey, torch.Tensor]],
    ]:
        # defaults
        graphs = [
            AtomicGraph.from_ase(s, self.model.cutoff.item() + 0.001)
            if isinstance(s, ase.Atoms)
            else s
            for s in structures
        ]
        if properties is None:
            properties = ["energy", "forces"]
            if all(map(has_cell, graphs)):
                properties.append("stress")

        # batched prediction
        results: list[dict[PropertyKey, torch.Tensor]] = []
        for batch in map(to_batch, groups_of(batch_size, graphs)):
            predictions = self.model.predict(batch, properties)
            seperated = _seperate(predictions, batch)
            results.extend(seperated)

        return graphs, results


## utils ##

T = TypeVar("T")
TensorLike = TypeVar("TensorLike", torch.Tensor, numpy.ndarray)


def to_numpy(results: dict[T, torch.Tensor]) -> dict[T, numpy.ndarray]:
    return {key: tensor.detach().numpy() for key, tensor in results.items()}


def _seperate(
    batched_prediction: dict[PropertyKey, TensorLike],
    batch: AtomicGraph,
) -> list[dict[PropertyKey, TensorLike]]:
    preds_list = []

    for idx, (start, stop) in enumerate(pairs(batch.other["ptr"])):
        preds = {}

        # per-structure properties
        if "energy" in batched_prediction:
            preds["energy"] = batched_prediction["energy"][idx]
        if "stress" in batched_prediction:
            preds["stress"] = batched_prediction["stress"][idx]

        # per-atom properties
        if "forces" in batched_prediction:
            preds["forces"] = batched_prediction["forces"][start:stop]
        if "local_energies" in batched_prediction:
            preds["local_energies"] = batched_prediction["local_energies"][
                start:stop
            ]

        preds_list.append(preds)

    return preds_list


Array = TypeVar("Array", torch.Tensor, numpy.ndarray)


def merge_predictions(
    predictions: list[dict[PropertyKey, Array]],
) -> dict[PropertyKey, Array]:
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

    merged: dict[PropertyKey, Array] = {}

    # stack per-structure properties along new axis
    for key in ["energy", "stress"]:
        if key in predictions[0]:
            merged[key] = stack([p[key] for p in predictions])  # type: ignore

    # concatenat per-atom properties along the first axis
    for key in ["forces", "local_energies"]:
        if key in predictions[0]:
            merged[key] = cat([p[key] for p in predictions])  # type: ignore

    return merged
