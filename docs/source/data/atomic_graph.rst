
Atomic Graphs
=============

We describe atomic graphs using the :class:`~graph_pes.AtomicGraph` class.
For convenient ways to create instances of such graphs from :class:`~ase.Atoms` objects,
see :func:`~graph_pes.data.io.to_atomic_graph`.


Definition
----------

.. autoclass:: graph_pes.AtomicGraph()



Derived Properties
------------------

We define a number of derived properties of atomic graphs. These
also work for :class:`~graph_pes.AtomicGraphBatch` instances.

.. autofunction:: graph_pes.atomic_graph.number_of_atoms
.. autofunction:: graph_pes.atomic_graph.number_of_edges
.. autofunction:: graph_pes.atomic_graph.has_cell
.. autofunction:: graph_pes.atomic_graph.neighbour_vectors
.. autofunction:: graph_pes.atomic_graph.neighbour_distances
.. autofunction:: graph_pes.atomic_graph.number_of_neighbours
.. autofunction:: graph_pes.atomic_graph.available_properties


Graph Operations
----------------

We define a number of operations that act on :class:`torch.Tensor` instances conditioned on the graph structure.
All of these are fully compatible with :class:`~graph_pes.AtomicGraphBatch` instances, and with ``TorchScript`` compilation.

.. autofunction:: graph_pes.atomic_graph.trim_edges
.. autofunction:: graph_pes.atomic_graph.sum_over_neighbours
.. autofunction:: graph_pes.atomic_graph.index_over_neighbours
.. autofunction:: graph_pes.atomic_graph.is_local_property
