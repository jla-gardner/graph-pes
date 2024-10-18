from __future__ import annotations

import warnings
from typing import Dict, Sequence, TypeVar, no_type_check, overload

import torch
from sklearn.linear_model import Ridge
from torch import Tensor

from graph_pes.logger import logger
from graph_pes.util import left_aligned_mul

from . import keys
from .graph_typing import (
    AtomicGraph,
    AtomicGraphBatch,
    LabelledBatch,
    LabelledGraph,
)

############################### BATCHING ###############################


def is_batch(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a batch of atomic graphs?

    Parameters
    ----------
    graph
        The graph to check.
    """

    return "batch" in graph


@overload
def to_batch(graphs: Sequence[LabelledGraph]) -> LabelledBatch: ...
@overload
def to_batch(graphs: Sequence[AtomicGraph]) -> AtomicGraphBatch: ...
@torch.no_grad()
def to_batch(
    graphs: Sequence[AtomicGraph | LabelledGraph],
) -> AtomicGraphBatch | LabelledBatch:
    """
    Collate a sequence of atomic graphs into a single batch object.

    Parameters
    ----------
    graphs
        The graphs to collate.
    """
    # easy properties: just cat these together
    Z = torch.cat([g[keys.ATOMIC_NUMBERS] for g in graphs])
    positions = torch.cat([g[keys._POSITIONS] for g in graphs])
    neighbour_offsets = torch.cat(
        [g[keys._NEIGHBOUR_CELL_OFFSETS] for g in graphs]
    )

    # stack cells along a new batch dimension
    cells = torch.stack([g[keys.CELL] for g in graphs])

    # standard way to caculaute the batch and ptr properties
    batch = torch.cat(
        [
            torch.full_like(g[keys.ATOMIC_NUMBERS], fill_value=i)
            for i, g in enumerate(graphs)
        ]
    )
    ptr = torch.tensor(
        [0] + [g[keys.ATOMIC_NUMBERS].shape[0] for g in graphs]
    ).cumsum(dim=0)

    # use the ptr to increment the neighbour index appropriately
    neighbour_index = torch.cat(
        [g[keys.NEIGHBOUR_INDEX] + ptr[i] for i, g in enumerate(graphs)], dim=1
    )

    graph = with_nice_repr(
        {
            keys.ATOMIC_NUMBERS: Z,
            keys._POSITIONS: positions,
            keys.NEIGHBOUR_INDEX: neighbour_index,
            keys.CELL: cells,
            keys._NEIGHBOUR_CELL_OFFSETS: neighbour_offsets,
            keys.BATCH: batch,
            keys.PTR: ptr,
        }
    )

    # - per structure labels are concatenated along a new batch axis (0)
    for label in [keys.ENERGY, keys.STRESS]:
        if label in graphs[0]:
            graph[label] = torch.stack([g[label] for g in graphs])

    # - per atom labels are concatenated along the first axis
    for key in [keys.FORCES, keys.LOCAL_ENERGIES]:
        if key in graphs[0]:
            graph[key] = torch.cat([g[key] for g in graphs])

    return graph  # type: ignore


############################### PROPERTIES ###############################


def number_of_atoms(graph: AtomicGraph) -> int:
    """
    Get the number of atoms in the ``graph``.

    Parameters
    ----------
    graph
        The atomic graph
    """

    return graph[keys.ATOMIC_NUMBERS].shape[0]


def number_of_edges(graph: AtomicGraph) -> int:
    """
    Get the number of edges in the ``graph``.

    Parameters
    ----------
    graph
        The atomic graph
    """

    return graph[keys.NEIGHBOUR_INDEX].shape[1]


def has_cell(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a structure with a defined unit cell?

    Parameters
    ----------
    graph
        The graph to check.
    """

    # TODO shouldn't this also check for at least one edge vector
    # across the periodic boundary?
    return bool(torch.any(graph[keys.CELL] != 0).item())


def neighbour_vectors(graph: AtomicGraph) -> Tensor:
    """
    Get the vector between each pair of atoms specified in the
    ``graph``'s ``"neighbour_index"`` property, respecting periodic
    boundary conditions where present.

    Parameters
    ----------
    graph
        The graph to extract the neighbour vectors from.
    """

    positions = graph[keys._POSITIONS]

    if not is_batch(graph):
        batch = torch.zeros_like(graph["atomic_numbers"])
        cell = graph["cell"].unsqueeze(0)
    else:
        batch = graph["batch"]  # type: ignore
        cell = graph["cell"]

    # avoid tuple de-structuring to keep torchscript happy
    i, j = graph["neighbour_index"][0], graph["neighbour_index"][1]
    cell_per_edge = cell[batch[i]]
    distance_offsets = torch.einsum(
        "kl,klm->km",
        graph["_neighbour_cell_offsets"].float(),
        cell_per_edge,
    )
    neighbour_positions = positions[j] + distance_offsets
    return neighbour_positions - positions[i]


def neighbour_distances(graph: AtomicGraph) -> Tensor:
    """
    Get the distance between each pair of atoms specified in the
    ``graph``'s ``neighbour_index`` property, respecting periodic
    boundary conditions where present.

    Parameters
    ----------
    graph
        The graph to extract the neighbour distances from.
    """
    return torch.linalg.norm(neighbour_vectors(graph), dim=-1)


def number_of_structures(batch: AtomicGraph) -> int:
    """
    Get the number of structures in the ``batch``.

    Parameters
    ----------
    batch
        The batch to get the number of structures for.
    """

    if not is_batch(batch):
        return 1
    return batch[keys.PTR].shape[0] - 1  # type: ignore


def structure_sizes(batch: AtomicGraph) -> Tensor:
    """
    Get the number of atoms in each structure in the ``batch``, of shape
    ``(S,)`` where ``S`` is the number of structures.

    Parameters
    ----------
    batch
        The batch to get the structure sizes for.

    Examples
    --------
    >>> len(graphs)
    3
    >>> [number_of_atoms(g) for g in graphs]
    [3, 4, 5]
    >>> structure_sizes(to_batch(graphs))
    tensor([3, 4, 5])
    """

    if not is_batch(batch):
        return torch.scalar_tensor(number_of_atoms(batch))

    return batch[keys.PTR][1:] - batch[keys.PTR][:-1]  # type: ignore


def number_of_neighbours(
    graph: AtomicGraph,
    include_central_atom: bool = True,
) -> Tensor:
    """
    Get a tensor, ``T``, of shape ``(N,)``, where ``N`` is the number of atoms
    in the ``graph``, such that ``T[i]`` gives the number of neighbours of atom
    ``i``. If ``include_central_atom`` is ``True``, then the central atom is
    included in the count.

    Parameters
    ----------
    graph
        The graph to get the number of neighbours for.
    include_central_atom
        Whether to include the central atom in the count.
    """

    return sum_over_neighbours(
        torch.ones_like(graph[keys.NEIGHBOUR_INDEX][0]),
        graph,
    ) + int(include_central_atom)


############################### ACTIONS ###############################


def is_local_property(x: Tensor, graph: AtomicGraph) -> bool:
    """
    Is the property ``x`` local to each atom in the ``graph``?

    Parameters
    ----------
    x
        The property to check.
    graph
        The graph to check the property for.
    """

    return len(x.shape) > 0 and x.shape[0] == number_of_atoms(graph)


def trim_edges(graph: AtomicGraph, cutoff: float) -> AtomicGraph:
    """
    Remove edges from the graph where the distance between the atoms
    is greater than the ``cutoff``.

    Parameters
    ----------
    graph
        The graph to trim the edges of.
    cutoff
        The maximum distance between atoms to keep the edge.
    """
    # unfortunately this function is a bit ugly: since this is part of
    # the forward pass, we need to ensure that this is torchscriptable
    # Hence we have a bunch of `type: ignore`s so that the external API
    # is clean.

    _RMAX: str = "_rmax"

    # make a shallow copy of the graph to prevent modifying the original
    graph_dict: dict[str, torch.Tensor] = dict(graph)  # type: ignore

    if _RMAX in graph_dict:
        existing_cutoff = graph_dict[_RMAX].item()
        if existing_cutoff < cutoff:
            warnings.warn(
                f"Graph already has a cutoff of {graph_dict[_RMAX]} which is "
                "less than the requested cutoff of {cutoff}.",
                stacklevel=2,
            )
            return graph
        elif existing_cutoff == cutoff:
            return graph

    distances = neighbour_distances(graph_dict)  # type: ignore
    mask = distances <= cutoff

    graph_dict[keys.NEIGHBOUR_INDEX] = graph_dict[keys.NEIGHBOUR_INDEX][:, mask]
    graph_dict[keys._NEIGHBOUR_CELL_OFFSETS] = graph_dict[
        keys._NEIGHBOUR_CELL_OFFSETS
    ][mask, :]
    graph_dict[_RMAX] = torch.tensor(cutoff, dtype=graph[keys._POSITIONS].dtype)

    return graph_dict  # type: ignore


def sum_over_neighbours(p: Tensor, graph: AtomicGraph) -> Tensor:
    r"""
    Shape-preserving sum over neighbours of a per-edge property, :math:`p_{ij}`,
    to get a per-atom property, :math:`P_i`:

    .. math::
        P_i = \sum_{j \in \mathcal{N}_i} p_{ij}

    where:

    * :math:`\mathcal{N}_i` is the set of neighbours of atom :math:`i`.
    * :math:`p_{ij}` is the property of the edge between atoms :math:`i` and
      :math:`j`.
    * :math:`p` is of shape :code:`(E, ...)` and :math:`P` is of shape
      :code:`(N, ...)` where :math:`E` is the number of edges and :math:`N` is
      the number of atoms. :code:`...` denotes any number of additional
      dimensions, including none.
    * :math:`P_i` = 0 if :math:`|\mathcal{N}_i| = 0`.

    Parameters
    ----------
    p
        The per-edge property to sum.
    graph
        The graph to sum the property for.
    """

    N = number_of_atoms(graph)
    central_atoms = graph[keys.NEIGHBOUR_INDEX][0]  # shape: (E,)

    # optimised implementations for common cases
    if p.dim() == 1:
        zeros = torch.zeros(N, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(0, central_atoms, p)

    elif p.dim() == 2:
        C = p.shape[1]
        zeros = torch.zeros(N, C, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(0, central_atoms.unsqueeze(1).expand(-1, C), p)

    shape = (N,) + p.shape[1:]
    zeros = torch.zeros(shape, dtype=p.dtype, device=p.device)

    if p.shape[0] == 0:
        # return all zeros if there are no atoms
        return zeros

    # create `index`, where index.shape = p.shape
    # and (index[e] == central_atoms[e]).all()
    ones = torch.ones_like(p)
    index = left_aligned_mul(ones, central_atoms).long()
    return zeros.scatter_add(0, index, p)


def sum_per_structure(x: Tensor, graph: AtomicGraph) -> Tensor:
    r"""
    Shape-preserving sum of a per-atom property, :math:`p`, to get a
    per-structure property, :math:`P`:

    If a single structure, containing ``N`` atoms, is used, then
    :math:`P = \sum_i p_i`, where:

    * :math:`p_i` is of shape ``(N, ...)``
    * :math:`P` is of shape ``(...)``
    * ``...`` denotes any
      number of additional dimensions, including ``None``.

    If a batch of ``S`` structures, containing a total of ``N`` atoms, is
    used, then :math:`P_k = \sum_{k \in K} p_k`, where:

    * :math:`K` is the collection of all atoms in structure :math:`k`
    * :math:`p_i` is of shape ``(N, ...)``
    * :math:`P` is of shape ``(S, ...)``
    * ``...`` denotes any
      number of additional dimensions, including ``None``.

    Parameters
    ----------
    x
        The per-atom property to sum.
    graph
        The graph to sum the property for.

    Examples
    --------
    Single graph case:

    >>> import torch
    >>> from ase.build import molecule
    >>> from graph_pes.data import sum_per_structure, to_atomic_graph
    >>> water = molecule("H2O")
    >>> graph = to_atomic_graph(water, cutoff=1.5)
    >>> # summing over a vector gives a scalar
    >>> sum_per_structure(torch.ones(3), graph)
    tensor(3.)
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(3, 2, 3), graph).shape
    torch.Size([2, 3])

    Batch case:

    >>> import torch
    >>> from ase.build import molecule
    >>> from graph_pes.data import sum_per_structure, to_atomic_graph, to_batch
    >>> water = molecule("H2O")
    >>> graph = to_atomic_graph(water, cutoff=1.5)
    >>> batch = to_batch([graph, graph])
    >>> batch
    AtomicGraphBatch(structures: 2, atoms: 6, edges: 8, has_cell: False)
    >>> # summing over a vector gives a tensor
    >>> sum_per_structure(torch.ones(6), graph)
    tensor([3., 3.])
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(6, 3, 4), graph).shape
    torch.Size([2, 3, 4])
    """

    if is_batch(graph):
        batch = graph[keys.BATCH]  # type: ignore
        shape = (number_of_structures(graph),) + x.shape[1:]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return zeros.scatter_add(0, batch, x)
    else:
        return x.sum(dim=0)


def index_over_neighbours(
    x: Tensor,  # [n_atoms, dim]
    graph: AtomicGraph,
) -> Tensor:  # [n_edges, dim]
    return x[graph["neighbour_index"][1]]


def guess_per_element_mean_and_var(
    per_structure_quantity: torch.Tensor,
    batch: AtomicGraphBatch,
    min_variance: float = 0.01,
) -> tuple[dict[int, float], dict[int, float]]:
    r"""
    Guess the per-element mean (:math:`\mu_Z`) and variance (:math:`\sigma_Z^2`)
    of a per-structure quantity using ridge regression under the following assumptions:

    1. the per-structure property, :math:`P`, is a summation over local
       properties of its components atoms: :math:`P = \sum_{i=1}^{N} p_{Z_i}`.
    2. the per-atom properties, :math:`p_{Z_i}`, are independent and identically
       distributed (i.i.d.) for each atom of type :math:`Z_i` according to a
       normal distribution: :math:`p_{Z_i} \sim \mathcal{N}(\mu_{Z_i}, \sigma_{Z_i}^2)`.

    Parameters
    ----------
    per_structure_quantity
        The per-structure quantity to guess the per-element mean and variance of.
    batch
        The batch of graphs to use for guessing the per-element mean and variance.

    Returns
    -------
    means
        A dictionary mapping atomic numbers to per-element means.
    variances
        A dictionary mapping atomic numbers to per-element variances.
    """  # noqa: E501

    # extract the atomic numbers to tensor N such that:
    # N[structure, Z] is the number of atoms of atomic number Z in structure
    unique_Zs = torch.unique(batch["atomic_numbers"])  # (n_Z,)
    N = torch.zeros(number_of_structures(batch), len(unique_Zs))  # (batch, n_Z)
    for i, Z in enumerate(unique_Zs):
        N[:, i] = sum_per_structure(
            (batch["atomic_numbers"] == Z).float(), batch
        )

    # calculate the per-element mean
    # use Ridge rather than LinearRegression to avoid singular matrices
    # when e.g. only one structure contains an atom of a given type...
    ridge = Ridge(fit_intercept=False, alpha=0.00001)
    ridge.fit(N.numpy(), per_structure_quantity)
    mu_Z = torch.tensor(ridge.coef_)
    means = {int(Z): float(mu) for Z, mu in zip(unique_Zs, mu_Z)}

    # calculate the per-element variance
    residuals = per_structure_quantity - N @ mu_Z
    # assuming that the squared residuals are a sum of the independent
    # variances for each atom, we can estimate these variances again
    # using Ridge regression
    ridge.fit(N.numpy(), residuals**2)
    var_Z = ridge.coef_
    # avoid negative variances by clipping to min value
    variances = {
        int(Z): max(float(var), min_variance)
        for Z, var in zip(unique_Zs, var_Z)
    }

    logger.debug(f"Per-element means: {means}")
    logger.debug(f"Per-element variances: {variances}")

    return means, variances


###################### making things look nice ######################


def available_labels(graph: AtomicGraph) -> list[keys.LabelKey]:
    """Get the labels that are available on the ``graph``."""
    return [label for label in keys.ALL_LABEL_KEYS if label in graph]


class _AtomicGraph_Impl(dict):
    @no_type_check
    def __repr__(self):
        info = {}

        if is_batch(self):
            name = "AtomicGraphBatch"
            info["structures"] = number_of_structures(self)
        else:
            name = "AtomicGraph"

        info["atoms"] = number_of_atoms(self)
        info["edges"] = self[keys.NEIGHBOUR_INDEX].shape[1]
        info["has_cell"] = has_cell(self)

        labels = available_labels(self)
        if labels:
            info["labels"] = labels

        info_str = ", ".join(f"{k}: {v}" for k, v in info.items())
        return f"{name}({info_str})"


T = TypeVar("T", AtomicGraph, AtomicGraphBatch, Dict[str, torch.Tensor])


def with_nice_repr(graph: T) -> T:
    """
    Add a nice __repr__ method to the AtomicGraph or AtomicGraphBatch
    object.

    This is useful for debugging purposes, as it allows you to see
    a summary of the graph when printing it.

    Parameters
    ----------
    graph
        The graph to add the nice __repr__ method to.
    """
    return _AtomicGraph_Impl(graph)  # type: ignore
