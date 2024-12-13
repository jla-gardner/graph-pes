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
    Tuple,
    Union,
    cast,
)

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from ase.stress import voigt_6_to_full_3x3_stress
from load_atoms.utils import remove_calculator
from typing_extensions import TypeAlias

from graph_pes.utils.misc import (
    all_equal,
    is_being_documented,
    left_aligned_div,
    left_aligned_mul,
    to_significant_figures,
    uniform_repr,
)

DEFAULT_CUTOFF: Final[float] = 5.0


PropertyKey: TypeAlias = Literal[
    "local_energies", "forces", "energy", "stress", "virial"
]
ALL_PROPERTY_KEYS: Final[List[PropertyKey]] = [
    "local_energies",
    "forces",
    "energy",
    "stress",
    "virial",
]

if not TYPE_CHECKING and not is_being_documented():
    # torchscript doesn't handle TypedDicts or Literal types:
    # at run-time, we just use less specific, but still correct, types
    Properties: TypeAlias = Dict[str, torch.Tensor]
    PropertyKey: TypeAlias = str


class AtomicGraph(NamedTuple):
    r"""
    An :class:`AtomicGraph` represents an atomic structure.
    Each node corresponds to an atom, and each directed edge links a central
    atom to a "bonded" neighbour.

    We implement such graphs as (immutable) :class:`~typing.NamedTuple`\ s.
    This allows for easy serialisation and compatibility with PyTorch,
    TorchScript and other libraries. These objects, and all functions that
    operate on them, are compatible with both isolated and periodic structures.

    Batches of multiple structures are represented by a single
    :class:`AtomicGraph` object containing multiple disjoint subgraphs. See
    :func:`~graph_pes.atomic_graph.to_batch` for more information.

    Properties
    ++++++++++

    Below we assume the graph contains ``N`` atoms, ``E`` edges (and ``S``
    structures if the graph is batched).

    We use two examples to illustrate the various properties:


    .. dropdown:: Water molecule

        .. code-block:: python

            >>> from ase.build import molecule
            >>> from load_atoms import view
            >>> water = molecule("H2O")
            >>> view(water, show_bonds=True)

        .. raw:: html
            :file: ../../docs/source/_static/water.html

        We can see that there are 3 atoms, with 2 bonds (and therefore 4
        directed edges):

        .. code-block:: python

            >>> from graph_pes import AtomicGraph
            >>> water_graph = AtomicGraph.from_ase(water, cutoff=1.2)
            >>> water_graph
            AtomicGraph(atoms=3, edges=4, has_cell=False, cutoff=1.2)

    .. dropdown:: Sodium crystal

        .. code-block:: python

            >>> from ase.build import bulk
            >>> from load_atoms import view
            >>> sodium = bulk("Na")
            >>> view(sodium.repeat(3), show_bonds=True)

        .. raw:: html
            :file: ../../docs/source/_static/Na.html

        This structure has a single atom within a periodic cell. If you
        look closely, you can see that this atom has 8 nearest neighbours.
        Only "source" atoms within the unit cell are included in the
        neighbour list, and hence there are 8 edges:

        .. code-block:: python

            >>> sodium_graph = AtomicGraph.from_ase(sodium, cutoff=3.7)
            >>> sodium_graph
            AtomicGraph(atoms=1, edges=8, has_cell=True, cutoff=3.7)
    """

    Z: torch.Tensor
    """
    The atomic numbers of the atoms in the graph, of shape ``(N,)``.

    .. code-block:: python

        >>> water_graph.Z
        tensor([8, 1, 1])
    
    .. code-block:: python

        >>> sodium_graph.Z
        tensor([11])
    """

    R: torch.Tensor
    """
    The cartesian positions of the atoms in the graph, of shape ``(N, 3)``.

    .. code-block:: python

        >>> water_graph.R
        tensor([[ 0.0000,  0.0000,  0.1193],
                [ 0.0000,  0.7632, -0.4770],
                [ 0.0000, -0.7632, -0.4770]])
    
    .. code-block:: python
        
        >>> sodium_graph.R
        tensor([[0., 0., 0.]])
    """

    cell: torch.Tensor
    """
    The unit cell vectors, of shape ``(3, 3)`` for a single structure, or
    ``(S, 3, 3)`` for a batched graph. If the structure is non-periodic,
    this will be all zeros.

    .. code-block:: python

        >>> water_graph.cell
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
    
    .. code-block:: python
        
        >>> sodium_graph.cell
        tensor([[-2.1150,  2.1150,  2.1150],
                [ 2.1150, -2.1150,  2.1150],
                [ 2.1150,  2.1150, -2.1150]])
    """

    neighbour_list: torch.Tensor
    """
    A neighbour list, of shape ``(2, E)``, where 
    ``i, j = graph.neighbour_list[:, k]`` is the ``k``'th directed edge
    in the graph, linking atom ``i`` to atom ``j``.

    .. code-block:: python

        >>> water_graph.neighbour_list
        tensor([[0, 0, 1, 2],
                [1, 2, 0, 0]])
    
    .. code-block:: python
        
        >>> sodium_graph.neighbour_list
        tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])
    """

    neighbour_cell_offsets: torch.Tensor
    """
    The offsets of the neighbours of each atom in units of the unit cell
    vectors, of shape ``(E, 3)``, such that:

    .. code-block:: python

        # k-th edge
        i, j = graph.neighbour_list[:, k]
        kth_displacement_vector = (
            graph.R[j] 
            + graph.neighbour_cell_offsets[k] @ graph.cell
            - graph.R[i]
        )

    In the case of an isolated, non-periodic structure, these will be all zeros.

    .. code-block:: python

        >>> water_graph.neighbour_cell_offsets
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
    
    .. code-block:: python

        >>> sodium_graph.neighbour_cell_offsets
        tensor([[ 0.,  0.,  1.],
                [ 0.,  1.,  0.],
                [ 1.,  1.,  1.],
                [ 1.,  0.,  0.],
                [-1.,  0.,  0.],
                [-1., -1., -1.],
                [ 0., -1.,  0.],
                [ 0.,  0., -1.]])

    """

    properties: Dict[PropertyKey, torch.Tensor]
    """
    A dictionary containing potential energy surface (PES) related properties
    of the graph.

    .. list-table::
        :header-rows: 1

        * - Key
          - Shape
          - Description
        * - :code:`"local_energies"`
          - :code:`(N,)`
          - contribution to the total energy from each atom
        * - :code:`"energy"`
          - ``()`` 
          
            (``(S,)`` if batched)
          - total energy of the structure
        * - :code:`"forces"`
          - :code:`(N, 3)`
          - force on each atom
        * - :code:`"stress"`
          - ``(3, 3)`` 
          
            (``(S, 3, 3)`` if batched)
          - stress tensor (see :doc:`../theory`)
        * - :code:`"virial"`
          - ``(3, 3)`` 
          
            (``(S, 3, 3)`` if batched)
          - virial stress tensor (see :doc:`../theory`)
    """

    cutoff: float
    """
    The cutoff distance used to create the neighbour list for this graph.
    """

    other: Dict[str, torch.Tensor]
    """
    A dictionary containing any other additional information about the graph.
    Feel free to populate this as you wish.
    """

    batch: Union[torch.Tensor, None] = None
    """
    A tensor of shape ``(N,)`` indicating the index ``i`` of the structure
    within the batch that each atom belongs to. Not present for a single
    structure.
    """

    ptr: Union[torch.Tensor, None] = None
    """
    A tensor of shape ``(S + 1,)`` indicating the index ``i`` of the first
    atom in each structure within the batch. Not present for a single structure.
    """

    @classmethod
    def from_ase(
        cls,
        structure: ase.Atoms,
        cutoff: float = DEFAULT_CUTOFF,
        property_mapping: Union[Mapping[str, PropertyKey], None] = None,
        others_to_include: Union[Sequence[str], None] = None,
    ) -> "AtomicGraph":
        r"""
        Convert an :class:`ase.Atoms` object to an :class:`AtomicGraph`.

        Parameters
        ----------
        structure
            The :class:`ase.Atoms` object to convert.
        cutoff
            The cutoff distance for neighbour finding.
        property_mapping
            An optional mapping of the form ``{key_on_structure:
            key_for_graph}`` defining how relevant properties are labelled on
            the :class:`ase.Atoms` object. If not provided, this function will
            extract all of ``"energy"``, ``"forces"``, ``"stress"``, or
            ``"virial"`` from the ``.info`` and ``.arrays`` dicts if they are
            present.
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

        # account for strange behaviour in ase 3.23.0+ whereby
        # properties are sometimes removed from the atoms.info/arrays dicts
        # to ensure we don't change the atoms object that the user passes in
        # we first make a copy, ensuring that the calculator is also copied over
        calc = structure.calc
        structure = structure.copy()
        structure.calc = calc
        # then we remove the calculator and copy over the properties to the
        # relevant info/arrays dicts (with help from load_atoms.utils)
        remove_calculator(structure)

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
        other: dict[str, torch.Tensor] = {}

        if property_mapping is None:
            all_keys = set(structure.info) | set(structure.arrays)
            property_mapping = {
                k: cast(PropertyKey, k)
                for k in ["energy", "forces", "stress", "virial"]
                if k in all_keys
            }
        if others_to_include is None:
            others_to_include = []

        for key, value in list(structure.info.items()) + list(
            structure.arrays.items()
        ):
            if key in property_mapping:
                property = property_mapping[key]
                # ensure stress is always 3x3, not voigt notation
                if property in ["stress", "virial"] and value.reshape(
                    -1
                ).shape == (6,):
                    value = voigt_6_to_full_3x3_stress(value)
                properties[property] = torch.tensor(value, dtype=dtype)

            elif key in others_to_include:
                other[key] = torch.tensor(value, dtype=dtype)

        missing = set(
            structure_key
            for structure_key, graph_key in property_mapping.items()
            if graph_key not in properties
        )
        if missing:
            raise ValueError(f"Unable to find properties: {missing}")

        return cls.create_with_defaults(
            Z=Z,
            R=R,
            cell=cell,
            neighbour_list=neighbour_list,
            neighbour_cell_offsets=neighbour_cell_offsets,
            properties=properties,
            other=other,
            cutoff=cutoff,
        )

    @classmethod
    def create_with_defaults(
        cls,
        Z: torch.Tensor,
        R: torch.Tensor,
        cell: Union[torch.Tensor, None] = None,
        neighbour_list: Union[torch.Tensor, None] = None,
        neighbour_cell_offsets: Union[torch.Tensor, None] = None,
        properties: Union[Dict[PropertyKey, torch.Tensor], None] = None,
        other: Union[Dict[str, torch.Tensor], None] = None,
        cutoff: float = 0.0,
    ) -> "AtomicGraph":
        """
        Create an :class:`AtomicGraph`, populating missing values with defaults.

        Parameters
        ----------
        Z
            The atomic numbers.
        R
            The cartesian positions.
        cell
            The unit cell. Defaults to ``torch.zeros(3, 3)``.
        neighbour_list
            The neighbour list. Defaults to ``torch.zeros(2, 0)``.
        neighbour_cell_offsets
            The neighbour cell offsets. Defaults to ``torch.zeros(0, 3)``.
        properties
            The properties. Defaults to ``{}``.
        other
            The other information. Defaults to ``{}``.
        """

        if cell is None:
            cell = torch.zeros(3, 3, device=R.device).float()
        if neighbour_list is None:
            neighbour_list = torch.zeros(2, 0, device=R.device).long()
        if neighbour_cell_offsets is None:
            neighbour_cell_offsets = torch.zeros(0, 3, device=R.device).float()
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
            cutoff=cutoff,
        )

    def to(self, device: Union[torch.device, str]) -> "AtomicGraph":
        """Move this graph to the specified device."""

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
            cutoff=self.cutoff,
            batch=self.batch.to(device) if self.batch is not None else None,
            ptr=self.ptr.to(device) if self.ptr is not None else None,
        )

    def __repr__(self):
        info = {}

        if self.batch is not None:
            name = "AtomicGraphBatch"
            info["structures"] = self.batch.max().item() + 1
        else:
            name = "AtomicGraph"

        info["atoms"] = number_of_atoms(self)
        info["edges"] = number_of_edges(self)
        info["has_cell"] = has_cell(self)
        info["cutoff"] = to_significant_figures(self.cutoff, 3)
        if self.properties:
            info["properties"] = available_properties(self)
        if self.other:
            info["other"] = list(self.other.keys())

        return uniform_repr(name, **info, indent_width=4)


############################### BATCHING ###############################


def to_batch(
    graphs: Sequence[AtomicGraph],
) -> AtomicGraph:
    """
    Collate a sequence of atomic graphs into a single batch object.

    The ``Z``, ``R``, ``neighbour_list``, and ``neighbour_cell_offsets``
    properties are concatenated along the first axis, while the ``cell``
    property is stacked along a new batch dimension.

    Values in the ``"other"`` dictionary are concatenated along the first axis
    if they appear to be a per-atoms property (i.e. their first dimension
    matches the number of atoms in the structure). Otherwise, they are stacked
    along a new batch dimension.

    Parameters
    ----------
    graphs
        The graphs to collate.

    Examples
    --------

    A basic example:

    >>> from ase.build import molecule
    >>> from graph_pes import AtomicGraph, to_batch
    >>> graphs = [
    ...     AtomicGraph.from_ase(molecule("H2O")),
    ...     AtomicGraph.from_ase(molecule("CH4")),
    ... ]
    >>> batch = to_batch(graphs)
    >>> batch.batch  # H20 has 3 atoms, CH4 has 5
    tensor([0, 0, 0, 1, 1, 1, 1, 1])
    >>> batch.ptr  # offset of first atom of each graph
    tensor([0, 3, 8])
    >>> batch.Z.shape
    torch.Size([8])
    >>> batch.R.shape
    torch.Size([8, 3])
    >>> batch.cell.shape
    torch.Size([2, 3, 3])
    """
    if any(is_batch(g) for g in graphs):
        raise ValueError("Cannot recursively batch graphs")

    # easy properties: just cat these together
    Z = torch.cat([g.Z for g in graphs])
    R = torch.cat([g.R for g in graphs])
    neighbour_offsets = torch.cat([g.neighbour_cell_offsets for g in graphs])

    # stack cells along a new batch dimension
    if not all_equal([has_cell(g) for g in graphs]):
        warnings.warn(
            "Attempting to batch a colleciton of graphs where only some "
            "have a defined unit cell. This may lead to unexpected results.",
            stacklevel=2,
        )
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

    # handle cutoff
    cutoffs = [g.cutoff for g in graphs]
    if not all_equal(cutoffs):
        warnings.warn(
            "Attempting to batch graphs with different cutoffs: "
            f"{cutoffs}. Setting graph.cutoff to the maximum.",
            stacklevel=2,
        )
    cutoff = max(cutoffs)

    properties: dict[PropertyKey, torch.Tensor] = {}
    # - per structure labels are concatenated along a new batch axis (0)
    for key in ["energy", "stress", "virial"]:
        key = cast(PropertyKey, key)
        if all(key in g.properties for g in graphs):
            properties[key] = torch.stack([g.properties[key] for g in graphs])

    # - per atom labels are concatenated along the first axis
    for key in ["forces", "local_energies"]:
        key = cast(PropertyKey, key)
        if all(key in g.properties for g in graphs):
            properties[key] = torch.cat([g.properties[key] for g in graphs])

    # - finally, add in the other stuff: this is a bit tricky
    #   since we need to try and infer whether these are per-atom
    #   or per-structure
    other: dict[str, torch.Tensor] = {}
    for key in graphs[0].other:
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
        cutoff=cutoff,
        batch=batch,
        ptr=ptr,
    )


def is_batch(graph: AtomicGraph) -> bool:
    """
    Does ``graph`` represent a batch of atomic graphs?

    Parameters
    ----------
    graph
        The graph to check.
    """

    return graph.batch is not None


############################### PROPERTIES ###############################


def get_cell_volume(graph: AtomicGraph) -> float:
    """
    Get the volume of the unit cell.
    """

    return torch.det(graph.cell).abs().item()


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
    batch: torch.Tensor = torch.zeros_like(graph.Z)
    cell: torch.Tensor = graph.cell.unsqueeze(0)

    # torchscript annoying-ness:
    graph_batch = graph.batch
    if graph_batch is not None:
        cell = graph.cell
        batch = graph_batch

    # avoid tuple de-structuring to keep torchscript happy
    i, j = graph.neighbour_list[0], graph.neighbour_list[1]  # (E,)
    if i.shape[0] == 0:
        return torch.zeros(0, 3, device=graph.R.device)

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
    # torchscript annoying-ness:
    graph_ptr = graph.ptr
    if graph_ptr is None:
        return 1
    return graph_ptr.shape[0] - 1


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
    # torchscript annoying-ness:
    graph_ptr = batch.ptr
    if graph_ptr is None:
        return torch.scalar_tensor(number_of_atoms(batch))

    return graph_ptr[1:] - graph_ptr[:-1]


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


from graph_pes.utils.misc import angle_spanned_by


def neighbour_triplets(graph: AtomicGraph) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Find all the triplets :math:`(i, j, k)` such that
    :math:`k` and :math:`j` are neighbours of :math:`i`,
    and:

    * :math:`j \neq k` for non periodic graphs
    * :math:`j, k` do not refer to the same image of an atom (respecting pbcs)
      for periodic graphs

    Returns
    -------
    triplet_idxs
        A ``(Y, 3)`` shaped tensor indicating the triplets, such that

        .. code-block:: python

            # get the y'th triplet
            i, j, k = triplet_idxs[y]

    triplet_vectors
        A ``(Y, 2, 3)`` shaped tensor indicating the vectors, such that

        .. code-block:: python

            # get the y'th vector triplet
            v_ij, v_ik = vector_triplets[y]

    Examples
    --------
    Note that the triplets are permutation sensitive, and hence
    both ``(i, j, k)`` and ``(i, k, j)`` are included:

    >>> graph = AtomicGraph.from_ase(molecule("H2O"))
    >>> triplet_idxs, triplet_vectors = neighbour_triplets(graph)
    >>> triplet_idxs
    tensor([[0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]])
    """

    vectors = neighbour_vectors(graph)

    # graph.neighbour_list is an (2, E) shaped tensor
    # indicating the indices of the neighbours of each atom
    # the first row is the central atom index
    # the second row is the neighbour atom index
    # the indices are 0-indexed
    central_atoms = graph.neighbour_list[0]
    neighbour_atoms = graph.neighbour_list[1]

    # we want to find all the triplets (i, j, k) such that
    # k and j are neighbours of i, and j != k (respecting pbc ghost atoms)
    triplet_idxs = []
    triplets_vectors = []

    for i in torch.unique(central_atoms):
        mask = central_atoms == i
        relevant_neighbours = neighbour_atoms[mask]
        relevant_vectors = vectors[mask]

        N = relevant_neighbours.shape[0]
        _idx = torch.cartesian_prod(
            torch.arange(N),
            torch.arange(N),
        )  # (N**2, 2)
        _idx = _idx[_idx[:, 0] != _idx[:, 1]]  # (N**2 - N, 2)

        kj = relevant_neighbours[_idx]  # (N**2 - N, 2)
        ikj = torch.hstack([i.expand(kj.shape[0]).view(-1, 1), kj])
        triplet_idxs.append(ikj)

        vkj = relevant_vectors[_idx]  # (N**2 - N, 2, 3)
        triplets_vectors.append(vkj)

    if len(triplet_idxs) == 0:
        return (
            torch.zeros(0, 3, device=graph.R.device).long(),
            torch.zeros(0, 2, 3, device=graph.R.device),
        )

    triplet_idxs = torch.vstack(triplet_idxs)
    triplets_vectors = torch.vstack(triplets_vectors)

    return triplet_idxs, triplets_vectors


def triplet_bond_descriptors(
    graph: AtomicGraph,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    For each triplet :math:`(i, j, k)`, get the bond angle :math:`\theta_{jk}`
    (in radians) and the two bond lengths :math:`r_{ij}` and :math:`r_{ik}`.

    Returns
    -------
    triplet_idxs
        The triplet indices, :math:`(i, j, k)`, of shape ``(Y, 3)``.
    angle
        The bond angle :math:`\theta_{jk}`, shape ``(Y,)``.
    r_ij
        The bond length :math:`r_{ij}`, shape ``(Y,)``.
    r_ik
        The bond length :math:`r_{ik}`, shape ``(Y,)``.

    Examples
    --------
    >>> graph = AtomicGraph.from_ase(molecule("H2O"))
    >>> angle, r_ij, r_ik = triplet_bond_descriptors(graph)
    >>> torch.rad2deg(angle)
    tensor([103.9999, 103.9999,  38.0001,  38.0001,  38.0001,  38.0001])
    """

    triplet_idxs, vector_triplets = neighbour_triplets(graph)

    if vector_triplets.shape[0] == 0:
        return (
            triplet_idxs,
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
            torch.zeros(0, device=graph.R.device).float(),
        )

    v1 = vector_triplets[:, 0]
    v2 = vector_triplets[:, 1]

    return (
        triplet_idxs,
        angle_spanned_by(v1, v2),
        torch.linalg.vector_norm(v1, dim=-1),
        torch.linalg.vector_norm(v2, dim=-1),
    )


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
    Return a new graph with edges trimmed to be no longer than the ``cutoff``.
    Leaves the original graph unchanged.

    Parameters
    ----------
    graph
        The graph to trim the edges of.
    cutoff
        The maximum distance between atoms to keep the edge.
    """

    existing_cutoff = graph.cutoff
    if existing_cutoff + 1e-5 < cutoff:
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

    # can't use _replace here due to TorchScript
    return AtomicGraph(
        Z=graph.Z,
        R=graph.R,
        cell=graph.cell,
        neighbour_list=neighbour_list,
        neighbour_cell_offsets=neighbour_cell_offsets,
        properties=graph.properties,
        other=graph.other,
        cutoff=cutoff,
        batch=graph.batch,
        ptr=graph.ptr,
    )


def sum_over_central_atom_index(
    p: torch.Tensor,
    central_atom_index: torch.Tensor,
    graph: AtomicGraph,
) -> torch.Tensor:
    r"""
    Efficient, shape-preserving sum of a property, :math:`p`, defined over a
    ``central_atom_index``, to get a per-atom property, :math:`P`,
    such that:

    .. code-block:: python

        # i in central_atom_index
        P[i] == torch.sum(p[central_atom_index == i], dim=0)
        # i not in central_atom_index
        P[i] == torch.zeros_like(p[0])

    .. seealso::

        :func:`sum_over_neighbours` for the explicit case where
        ``p`` is a per-edge property.

    Parameters
    ----------
    p
        The property to sum, of shape ``(Y, ...)``.
    central_atom_index
        The central atoms relevant to each element of ``p``, of shape ``(Y,)``.
    graph
        The graph to sum the property for.

    Returns
    -------
    P: torch.Tensor
        The summed property, of shape ``(N, ...)``.
    """

    N = number_of_atoms(graph)

    # optimised implementations for common cases
    if p.dim() == 1:
        zeros = torch.zeros(N, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(0, central_atom_index, p)

    elif p.dim() == 2:
        C = p.shape[1]
        zeros = torch.zeros(N, C, dtype=p.dtype, device=p.device)
        return zeros.scatter_add(
            0,
            central_atom_index.unsqueeze(1).expand(-1, C),
            p,
        )

    shape = (N,) + p.shape[1:]
    zeros = torch.zeros(shape, dtype=p.dtype, device=p.device)

    if p.shape[0] == 0:
        # return all zeros if there are no atoms
        return zeros

    # create `index`, where index.shape = p.shape
    # and (index[e] == central_atoms[e]).all()
    ones = torch.ones_like(p)
    index = left_aligned_mul(ones, central_atom_index).long()
    return zeros.scatter_add(0, index, p)


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

    return sum_over_central_atom_index(p, graph.neighbour_list[0], graph)


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
    >>> from graph_pes.atomic_graph import sum_per_structure, AtomicGraph
    >>> water = molecule("H2O")
    >>> graph = AtomicGraph.from_ase(water, cutoff=1.5)
    >>> # summing over a vector gives a scalar
    >>> sum_per_structure(torch.ones(3), graph)
    tensor(3.)
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(3, 2, 3), graph).shape
    torch.Size([2, 3])

    Batch case:

    >>> import torch
    >>> from ase.build import molecule
    >>> from graph_pes.atomic_graph import sum_per_structure, AtomicGraph, to_batch
    >>> water = molecule("H2O")
    >>> graph = AtomicGraph.from_ase(water, cutoff=1.5)
    >>> batch = to_batch([graph, graph])
    >>> batch
    AtomicGraphBatch(structures: 2, atoms: 6, edges: 8, has_cell: False)
    >>> # summing over a vector gives a tensor
    >>> sum_per_structure(torch.ones(6), graph)
    tensor([3., 3.])
    >>> # summing over higher order tensors gives a tensor
    >>> sum_per_structure(torch.ones(6, 3, 4), graph).shape
    torch.Size([2, 3, 4])
    """  # noqa: E501

    # torchscript annoying-ness:
    graph_batch = graph.batch
    if graph_batch is not None:
        shape = (number_of_structures(graph),) + x.shape[1:]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return zeros.scatter_add(0, graph_batch, x)
    else:
        return x.sum(dim=0)


def index_over_neighbours(x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
    """
    Index a per-atom property, :math:`x`, over the neighbours of each atom in
    the ``graph``.
    """
    return x[graph.neighbour_list[1]]


def divide_per_atom(x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
    r"""
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
