
Atomic Graphs
=============

We describe atomic graphs using the :class:`~graph_pes.AtomicGraph` class.
For convenient ways to create instances of such graphs from :class:`~ase.Atoms` objects,
see :meth:`~graph_pes.AtomicGraph.from_ase`.


Definition
----------

.. autoclass:: graph_pes.AtomicGraph()
    :show-inheritance:
    :members:


Batching
--------

A batch of :class:`~graph_pes.AtomicGraph` instances is itself represented by a single
:class:`~graph_pes.AtomicGraph` instance, containing multiple disjoint subgraphs.

:class:`~graph_pes.AtomicGraph` batches are created using :func:`~graph_pes.atomic_graph.to_batch`:

.. autofunction:: graph_pes.atomic_graph.to_batch
.. autofunction:: graph_pes.atomic_graph.is_batch

If you need to define custom batching logic for a field in the ``other`` property,
you can use :func:`~graph_pes.atomic_graph.register_custom_batcher`:

.. autofunction:: graph_pes.atomic_graph.register_custom_batcher

Derived Properties
------------------

We define a number of derived properties of atomic graphs. These
work for both isolated and batched :class:`~graph_pes.AtomicGraph` instances.

.. autofunction:: graph_pes.atomic_graph.number_of_atoms
.. autofunction:: graph_pes.atomic_graph.number_of_edges
.. autofunction:: graph_pes.atomic_graph.has_cell
.. autofunction:: graph_pes.atomic_graph.neighbour_vectors
.. autofunction:: graph_pes.atomic_graph.neighbour_distances
.. autofunction:: graph_pes.atomic_graph.number_of_neighbours
.. autofunction:: graph_pes.atomic_graph.available_properties
.. autofunction:: graph_pes.atomic_graph.number_of_structures
.. autofunction:: graph_pes.atomic_graph.structure_sizes
.. autofunction:: graph_pes.atomic_graph.neighbour_triplets
.. autofunction:: graph_pes.atomic_graph.triplet_bond_descriptors

Graph Operations
----------------

We define a number of operations that act on :class:`torch.Tensor` instances conditioned on the graph structure.
All of these are fully compatible with batched :class:`~graph_pes.AtomicGraph` instances, and with ``TorchScript`` compilation.

.. autofunction:: graph_pes.atomic_graph.is_local_property
.. autofunction:: graph_pes.atomic_graph.index_over_neighbours
.. autofunction:: graph_pes.atomic_graph.sum_over_central_atom_index
.. autofunction:: graph_pes.atomic_graph.sum_over_neighbours
.. autofunction:: graph_pes.atomic_graph.sum_per_structure
.. autofunction:: graph_pes.atomic_graph.divide_per_atom
.. autofunction:: graph_pes.atomic_graph.trim_edges
