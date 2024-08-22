from __future__ import annotations

from typing import TYPE_CHECKING, Dict, TypedDict

import torch
from torch import Tensor
from typing_extensions import TypeAlias

from graph_pes.util import _is_being_documented


class AtomicGraph(TypedDict):
    r"""
    An :class:`AtomicGraph` represents an atomic structure.
    Each node corresponds to an atom, and each directed edge links a central
    atom to a "bonded" neighbour.

    We implement such graphs as simple dictionaries from property strings to
    PyTorch :class:`Tensors <torch.Tensor>`. This allows for easy serialisation
    and compatibility with PyTorch, TorchScript and other libraries.

    The following properties completely define any (unlabelled) atomic graph
    with :code:`N` atoms and :code:`E` edges:

    .. list-table::
        :header-rows: 1

        * - Key
          - Shape
          - Property
        * - :code:`"atomic_numbers"`
          - :code:`(N,)`
          - atomic number (Z) of each atom
        * - :code:`"neighbour_index"`
          - :code:`(2, E)`
          - indices of each directed neighbour pair
        * - :code:`"_positions"`
          - :code:`(N, 3)`
          - cartesian position of each atom
        * - :code:`"cell"`
          - :code:`(3, 3)`
          - | unit cell vectors
            | (``torch.zeros(3, 3)`` for isolated systems)
        * - :code:`"_neighbour_cell_offsets"`
          - :code:`(E, 3)`
          - | offset of each neighbour pair in
            | units of the unit cell vectors


    Direct access of the position and cell-offset properties is discouraged,
    as indicated by the leading underscore. We provide implementations for a
    wide range of derived properties, such as
    :func:`~graph_pes.graphs.operations.neighbour_distances`, that correctly
    handle periodic boundary conditions and other subtleties.

    Example
    -------

    >>> from ase.build import molecule
    >>> from graph_pes.data import to_atomic_graph
    >>> ethanol = molecule("CH3CH2OH")
    >>> graph = to_atomic_graph(ethanol, cutoff=1.5)
    >>> graph  # we implement a custom __repr__
    AtomicGraph(atoms: 9, edges: 14, has_cell: False)
    >>> for key, value in graph.items():
    ...     print(f"{key:>23}", tuple(value.shape))
             atomic_numbers (9,)
                       cell (3, 3)
                v_positions (9, 3)
            neighbour_index (2, 14)
    _neighbour_cell_offsets (14, 3)
    """

    # required properties
    atomic_numbers: Tensor
    cell: Tensor
    neighbour_index: Tensor
    _positions: Tensor
    _neighbour_cell_offsets: Tensor


class AtomicGraphBatch(AtomicGraph):
    """
    We represent a batch of atomic graphs as a single larger graph containing
    disjoint subgraphs corresponding to each structure.

    Together with all the properties defined on :class:`AtomicGraph`,
    an :class:`AtomicGraphBatch` containing ``S`` structures and a total
    of ``N`` atoms has two additional properties:

    .. list-table::
        :header-rows: 1

        * - Key
          - Shape
          - Property
        * - :code:`"batch"`
          - :code:`(N,)`
          - index of the structure for each atom
        * - :code:`"ptr"`
          - :code:`(S+1,)`
          - | pointer to the start and end
            | of each structure in the batch

    Example
    -------
    >>> from ase.build import molecule
    >>> from graph_pes.data import to_atomic_graphs, to_batch
    >>> structures = [molecule("H2O"), molecule("CH4")]
    >>> graphs = to_atomic_graphs(structures, cutoff=1.5)
    >>> batch = to_batch(graphs)
    >>> batch  # we implement a custom __repr__
    AtomicGraphBatch(structures: 2, atoms: 8, edges: 12, has_cell: False)
    >>> batch["atomic_numbers"]
    tensor([8, 1, 1, 6, 1, 1, 1, 1])
    >>> batch["batch"]
    tensor([0, 0, 0, 1, 1, 1, 1, 1])
    >>> batch["ptr"]
    tensor([0, 3, 8])
    """

    batch: Tensor
    ptr: Tensor


class Labels(TypedDict):
    energy: Tensor
    forces: Tensor
    stress: Tensor


class LabelledGraph(AtomicGraph, Labels):
    """
    A :class:`LabelledGraph` is an :class:`AtomicGraph` with additional
    property labels indexed by :class:`~graph_pes.graphs.keys.LabelKey` s:

    .. list-table::
        :header-rows: 1

        * - Key
          - Shape
          - Property
        * - :code:`"energy"`
          - :code:`()`
          - total energy
        * - :code:`"forces"`
          - :code:`(N, 3)`
          - force on each atom
        * - :code:`"stress"`
          - :code:`(3, 3)`
          - stress tensor
    """


class LabelledBatch(AtomicGraphBatch, Labels):
    """
    A :class:`LabelledBatch` is an :class:`AtomicGraphBatch` with additional
    property labels:

    .. list-table::
        :header-rows: 1

        * - Key
          - Shape
          - Property
        * - :code:`"energy"`
          - :code:`(S,)`
          - total energy
        * - :code:`"forces"`
          - :code:`(N, 3)`
          - force on each atom
        * - :code:`"stress"`
          - :code:`(S, 3, 3)`
          - stress tensor
    """


# Torchscript compilation is used in graph_pes to serialise models for
# deployment in e.g. LAMMPs, and also sometimes as a means to accelerate
# training.
#
# When such compilation takes place, Torchscript uses the run-time type
# hint information to determine the form of the data structures and
# required functions that act on these.
#
# Unfortunately, Torchscript does not support:
# - TypedDicts
# - Literal types
# and so we resort below to defining the AtomicGraph (and other derived) types
# as TypedDicts when type checking is enabled (i.e. when we write code in
# an IDE), and as vanilla dictionaries types at run time.
#
# Further to this, in order to enable nicer printing of the AtomicGraph
# objects, we actually subclass the dictionary type and override the __repr__
# method, see `graph_pes.graphs.operations._AtomicGraph_Impl` and
# `graph_pes.graphs.operations.with_nice_repr` for more details.
#
# It is important to realise that this (and any other custom
# behaviour implemented on the _AtomicGraph_Impl class) will not be available
# when the AtomicGraph is compiled to Torchscript and used in e.g. LAMMPs.
# Hence we don't put any logic on this _AtomicGraph_Impl class.
#
#
# Type definitions:
# -----------------
# when people are writing code, we want correct types
# and so use TypedDicts to raise warnings in IDEs
#
# but at run time, we want @torch.jit.script to work, and this requires
# vanilla dictionaries with no pre-defined keys:
if not (TYPE_CHECKING or _is_being_documented()):
    AtomicGraph: TypeAlias = Dict[str, torch.Tensor]
    LabelledGraph: TypeAlias = Dict[str, torch.Tensor]
    AtomicGraphBatch: TypeAlias = Dict[str, torch.Tensor]
    LabelledBatch: TypeAlias = Dict[str, torch.Tensor]
