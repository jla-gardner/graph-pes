from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class AtomicGraph(TypedDict):
    r"""
    An :class:`AtomicGraph` is a representation of an atomic structure.
    Each node corresponds to an atom, and each edge links two atoms that
    are “bonded”.

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
          - indices of each neighbour pair
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
    as indicated by the leading underscore. We implement a wide range of
    derived properties, such as :func:`~graph_pes.data.neighbour_distances`,
    that correctly handle periodic boundary conditions and other subtleties.

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
    property labels:

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
