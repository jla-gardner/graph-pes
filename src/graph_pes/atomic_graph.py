import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    Final,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    Union,
    cast,
)

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from typing_extensions import TypeAlias

from graph_pes.utils.misc import (
    all_equal,
    is_being_documented,
    left_aligned_div,
    left_aligned_mul,
    uniform_repr,
)

DEFAULT_CUTOFF: Final[float] = 5.0


PropertyKey: TypeAlias = Literal["local_energies", "forces", "energy", "stress"]
ALL_PROPERTY_KEYS: Final[List[PropertyKey]] = [
    "local_energies",
    "forces",
    "energy",
    "stress",
]

if not TYPE_CHECKING and not is_being_documented():
    # torchscript doesn't handle TypedDicts or Literal types:
    # at run-time, we just use less specific, but still correct, types
    Properties: TypeAlias = Dict[str, torch.Tensor]  # noqa: UP006 <- torchscript issue
    PropertyKey: TypeAlias = str


class AtomicGraph(NamedTuple):
    # special members of other: cutoff, batch, ptr

    Z: torch.Tensor
    R: torch.Tensor
    cell: torch.Tensor

    neighbour_list: torch.Tensor
    neighbour_cell_offsets: torch.Tensor

    properties: Dict[PropertyKey, torch.Tensor]
    other: Dict[str, torch.Tensor]  # noqa: UP006 <- torchscript issue

    def to(self, device: Union[torch.device, str]) -> "AtomicGraph":
        properties: dict[PropertyKey, torch.Tensor] = {
            k: v.to(device) for k, v in self.properties.items()
        }
        return AtomicGraph(
            Z=self.Z.to(device),
            R=self.R.to(device),
            cell=self.cell.to(device),
            neighbour_list=self.neighbour_list.to(device),
            neighbour_cell_offsets=self.neighbour_cell_offsets.to(device),
            properties=properties,
            other={k: v.to(device) for k, v in self.other.items()},
        )

    @classmethod
    def create(
        cls,
        Z: torch.Tensor,
        R: torch.Tensor,
        cell: Union[torch.Tensor, None] = None,
        neighbour_list: Union[torch.Tensor, None] = None,
        neighbour_cell_offsets: Union[torch.Tensor, None] = None,
        properties: Union[Dict[PropertyKey, torch.Tensor], None] = None,
        other: Union[Dict[str, torch.Tensor], None] = None,
    ) -> "AtomicGraph":
        if cell is None:
            cell = torch.zeros(3, 3)
        if neighbour_list is None:
            neighbour_list = torch.zeros(2, 0)
        if neighbour_cell_offsets is None:
            neighbour_cell_offsets = torch.zeros(0, 3)
        if properties is None:
            properties = {}
        if other is None:
            other = {}
        return cls(
            Z=Z,
            R=R,
            cell=cell,
            neighbour_list=neighbour_list,
            neighbour_cell_offsets=neighbour_cell_offsets,
            properties=properties,
            other=other,
        )

    @classmethod
    def from_ase(
        cls,
        structure: ase.Atoms,
        cutoff: float = DEFAULT_CUTOFF,
        property_mapping: Union[Mapping[str, PropertyKey], None] = None,
        others_to_include: Union[Sequence[str], None] = None,
    ) -> "AtomicGraph":
        r"""
        Convert an ASE Atoms object to an AtomicGraph.

        Parameters
        ----------
        structure
            The ASE Atoms object.
        cutoff
            The cutoff distance for neighbour finding.
        property_mapping
            An optional mapping of the form ``{key_on_structure:
            key_for_graph}`` defining how relevant properties are labelled on
            the ASE Atoms object. If not provided, this function will extract
            all of ``"energy"``, ``"forces"``, or ``"stress"`` from the
            ``.info`` and ``.arrays`` dicts if they are present.
        others_to_include
            An optional list of other ``.info``/``.arrays`` keys to include in
            the graph's ``other`` dict. The corresponding values will be
            converted to :class:`torch.Tensor`\ s.

        Example
        -------

        .. code-block:: python

            >>> from ase.build import molecule
            >>> from graph_pes import AtomicGraph
            >>> # create a a structure with some extra info
            >>> atoms = molecule("H2O")
            >>> atoms.info["DFT_energy"] = -10.0
            >>> atoms.info["unique_id"] = 1234

            >>> # default behaviour:
            >>> AtomicGraph.from_ase(atoms)
            AtomicGraph(atoms=3, edges=6, has_cell=False, cutoff=5.0)

            >>> # specify how to map properties, and other things to include
            >>> AtomicGraph.from_ase(
            ...     atoms,
            ...     property_mapping={
            ...         "DFT_energy": "energy",
            ...     },
            ...     others_to_include=["unique_id"],
            ... )
            AtomicGraph(
                atoms=3,
                edges=6,
                has_cell=False,
                cutoff=5.0,
                properties=['energy'],
                other=['unique_id']
            )
        """

        dtype = torch.get_default_dtype()

        # structure
        Z = torch.tensor(structure.numbers, dtype=torch.long)
        R = torch.tensor(structure.positions, dtype=dtype)
        cell = torch.tensor(structure.cell.array, dtype=dtype)

        # neighbour list
        i, j, offsets = neighbor_list("ijS", structure, cutoff)
        neighbour_list = torch.tensor(np.vstack([i, j]), dtype=torch.long)
        neighbour_cell_offsets = torch.tensor(offsets, dtype=dtype)

        # properties
        properties: dict[PropertyKey, torch.Tensor] = {}
        other: dict[str, torch.Tensor] = {"cutoff": torch.tensor(cutoff)}

        if property_mapping is None:
            all_keys = set(structure.info) | set(structure.arrays)
            property_mapping = {
                k: cast(PropertyKey, k)
                for k in ["energy", "forces", "stress"]
                if k in all_keys
            }
        if others_to_include is None:
            others_to_include = []

        for key, value in list(structure.info.items()) + list(
            structure.arrays.items()
        ):
            if key in property_mapping:
                property_key = property_mapping[key]
                properties[property_key] = torch.tensor(value, dtype=dtype)
            elif key in others_to_include:
                other[key] = torch.tensor(value, dtype=dtype)

        missing = set(
            structure_key
            for structure_key, graph_key in property_mapping.items()
            if graph_key not in properties
        )
        if missing:
            raise ValueError(f"Unable to find properties: {missing}")

        return cls.create(
            Z=Z,
            R=R,
            cell=cell,
            neighbour_list=neighbour_list,
            neighbour_cell_offsets=neighbour_cell_offsets,
            properties=properties,
            other=other,
        )

    def __repr__(self):
        info = {}

        if is_batch(self):
            name = "AtomicGraphBatch"
            info["structures"] = self.other["batch"].max().item() + 1
        else:
            name = "AtomicGraph"

        info["atoms"] = number_of_atoms(self)
        info["edges"] = number_of_edges(self)
        info["has_cell"] = has_cell(self)
        if "cutoff" in self.other:
            info["cutoff"] = self.other["cutoff"].item()
        if self.properties:
            info["properties"] = available_properties(self)
        actual_other = {
            k: v
            for k, v in self.other.items()
            if k not in {"cutoff", "batch", "ptr"}
        }
        if actual_other:
            info["other"] = list(actual_other.keys())

        return uniform_repr(name, **info, indent_width=4)


############################### BATCHING ###############################


def to_batch(
    graphs: Sequence[AtomicGraph],
) -> AtomicGraph:
    """
    Collate a sequence of atomic graphs into a single batch object.

    Parameters
    ----------
    graphs
        The graphs to collate.
    """
    if any(is_batch(g) for g in graphs):
        raise ValueError("Cannot recursively batch graphs")

    # easy properties: just cat these together
    Z = torch.cat([g.Z for g in graphs])
    R = torch.cat([g.R for g in graphs])
    neighbour_offsets = torch.cat([g.neighbour_cell_offsets for g in graphs])

    # stack cells along a new batch dimension
    cells = torch.stack([g.cell for g in graphs])

    # standard way to caculaute the batch and ptr properties
    batch = torch.cat(
        [torch.full_like(g.Z, fill_value=i) for i, g in enumerate(graphs)]
    )
    ptr = torch.tensor([0] + [g.Z.shape[0] for g in graphs]).cumsum(dim=0)

    # use the ptr to increment the neighbour index appropriately
    neighbour_list = torch.cat(
        [g.neighbour_list + ptr[i] for i, g in enumerate(graphs)], dim=1
    )

    properties: dict[PropertyKey, torch.Tensor] = {}
    # - per structure labels are concatenated along a new batch axis (0)
    for label in ["energy", "stress"]:
        if label in graphs[0].properties:
            properties[label] = torch.stack(
                [g.properties[label] for g in graphs]
            )

    # - per atom labels are concatenated along the first axis
    for key in ["forces", "local_energies"]:
        if key in graphs[0].properties:
            properties[key] = torch.cat([g.properties[key] for g in graphs])

    # - finally, add in the other stuff: this is a bit tricky
    #   since we need to try and infer whether these are per-atom
    #   or per-structure
    other = {"batch": batch, "ptr": ptr}

    for key in graphs[0].other:
        if key in ["batch", "ptr"]:
            continue

        # special handling for cutoff
        if key == "cutoff":
            cutoffs = [g.other["cutoff"] for g in graphs]
            if not all_equal(cutoffs):
                warnings.warn(
                    "Attempting to batch graphs with different cutoffs: "
                    f"{cutoffs}. Using the maximum cutoff.",
                    stacklevel=2,
                )
            other["cutoff"] = torch.tensor(cutoffs).max()
            continue

        # if all of the value tensors have the same number of entries
        # as atoms in the structure, we'll treat them as per-atom
        if all(is_local_property(g.other[key], g) for g in graphs):
            other[key] = torch.cat([g.other[key] for g in graphs])
        # otherwise, we'll assume they're per-structure
        else:
            other[key] = torch.stack([g.other[key] for g in graphs])

    return AtomicGraph(
        Z=Z,
        R=R,
        cell=cells,
        neighbour_list=neighbour_list,
        neighbour_cell_offsets=neighbour_offsets,
        properties=properties,
        other=other,
    )


def is_batch(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a batch of atomic graphs?

    Parameters
    ----------
    graph
        The graph to check.
    """

    return "batch" in graph.other and "ptr" in graph.other


############################### PROPERTIES ###############################


def number_of_atoms(graph: AtomicGraph) -> int:
    """
    Get the number of atoms in the ``graph``.
    """

    return graph.Z.shape[0]


def number_of_edges(graph: AtomicGraph) -> int:
    """
    Get the number of edges in the ``graph``.
    """

    return graph.neighbour_list.shape[1]


def has_cell(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a structure with a defined unit cell?
    """

    return not torch.allclose(graph.cell, torch.zeros_like(graph.cell))


def neighbour_vectors(graph: AtomicGraph) -> torch.Tensor:
    """
    Get the vector between each pair of atoms specified in the
    ``graph``'s ``"neighbour_list"`` property, respecting periodic
    boundary conditions where present.
    """

    # to simplify the logic below, we'll expand
    # a single graph into a batch of one
    if is_batch(graph):
        batch = graph.other["batch"]
        cell = graph.cell
    else:
        batch = torch.zeros_like(graph.Z)
        cell = graph.cell.unsqueeze(0)

    # avoid tuple de-structuring to keep torchscript happy
    i, j = graph.neighbour_list[0], graph.neighbour_list[1]  # (E,)
    cell_per_edge = cell[batch[i]]  # (E, 3, 3)
    distance_offsets = torch.einsum(
        "kl,klm->km",
        graph.neighbour_cell_offsets.to(cell_per_edge.dtype),
        cell_per_edge,
    )  # (E, 3)
    neighbour_positions = graph.R[j] + distance_offsets  # (E, 3)
    return neighbour_positions - graph.R[i]  # (E, 3)


def neighbour_distances(graph: AtomicGraph) -> torch.Tensor:
    """
    Get the distance between each pair of atoms specified in the
    ``graph``'s ``neighbour_list`` property, respecting periodic
    boundary conditions where present.
    """
    return torch.linalg.norm(neighbour_vectors(graph), dim=-1)


def number_of_structures(graph: AtomicGraph) -> int:
    """
    Get the number of structures in the ``graph``.
    """

    if not is_batch(graph):
        return 1
    return graph.other["ptr"].shape[0] - 1


def structure_sizes(batch: AtomicGraph) -> torch.Tensor:
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

    return batch.other["ptr"][1:] - batch.other["ptr"][:-1]


def number_of_neighbours(
    graph: AtomicGraph,
    include_central_atom: bool = True,
) -> torch.Tensor:
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
        torch.ones_like(graph.neighbour_list[0]),
        graph,
    ) + int(include_central_atom)


def available_properties(graph: AtomicGraph) -> List[PropertyKey]:
    """Get the labels that are available on the ``graph``."""
    return [cast(PropertyKey, k) for k in graph.properties]


############################### ACTIONS ###############################


def is_local_property(x: torch.Tensor, graph: AtomicGraph) -> bool:
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

    existing_cutoff = graph.other.get(
        "cutoff", torch.tensor(float("inf"))
    ).item()
    if existing_cutoff < cutoff:
        warnings.warn(
            f"Graph already has a cutoff of {existing_cutoff} which is "
            f"less than the requested cutoff of {cutoff}.",
            stacklevel=2,
        )
        return graph
    elif existing_cutoff == cutoff:
        return graph

    distances = neighbour_distances(graph)
    mask = distances <= cutoff
    neighbour_list = graph.neighbour_list[:, mask]
    neighbour_cell_offsets = graph.neighbour_cell_offsets[mask, :]

    other = graph.other.copy()
    other["cutoff"] = torch.tensor(cutoff)

    # can't use _replace here due to TorchScript
    return AtomicGraph(
        Z=graph.Z,
        R=graph.R,
        cell=graph.cell,
        neighbour_list=neighbour_list,
        neighbour_cell_offsets=neighbour_cell_offsets,
        properties=graph.properties,
        other=other,
    )


def sum_over_neighbours(p: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
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
    central_atoms = graph.neighbour_list[0]  # shape: (E,)

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


def sum_per_structure(x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
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
        batch = graph.other["batch"]
        shape = (number_of_structures(graph),) + x.shape[1:]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return zeros.scatter_add(0, batch, x)
    else:
        return x.sum(dim=0)


def index_over_neighbours(x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
    """
    Index a per-atom property, :math:`x`, over the neighbours of each atom in
    the ``graph``.
    """
    return x[graph.neighbour_list[1]]


def divide_per_atom(x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
    """
    Divide a per-structure property, :math:`X`, by the number of atoms in each
    structure to get a per-atom property, :math:`x`:

    .. math::
        x_i = \frac{X_k}{N_k}

    where:

    * :math:`X` is of shape ``(S, ...)``
    * :math:`x` is of shape ``(N, ...)``
    * :math:`S` is the number of structures
    * :math:`N` is the number of atoms
    """
    return left_aligned_div(x, structure_sizes(graph))
