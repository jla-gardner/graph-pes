
Atomic Graphs
=============

We describe atomic graphs using the :class:`~graph_pes.graphs.AtomicGraph` class.
For convenient ways to create instances of such graphs, see :func:`to_atomic_graph`.


Definition
----------

.. autoclass:: graph_pes.graphs.AtomicGraph()
.. autoclass:: graph_pes.graphs.LabelledGraph()


Creation
--------

.. autofunction:: graph_pes.data.io.to_atomic_graph
.. autofunction:: graph_pes.data.io.to_atomic_graphs


Derived Properties
------------------

We define a number of derived properties of atomic graphs. These
also work for :class:`~graph_pes.graphs.AtomicGraphBatch` instances.

.. autofunction:: graph_pes.graphs.operations.number_of_atoms
.. autofunction:: graph_pes.graphs.operations.number_of_edges
.. autofunction:: graph_pes.graphs.operations.has_cell
.. autofunction:: graph_pes.graphs.operations.neighbour_vectors
.. autofunction:: graph_pes.graphs.operations.neighbour_distances


Graph Operations
----------------

.. autofunction:: graph_pes.graphs.operations.sum_over_neighbours
.. autofunction:: graph_pes.graphs.operations.is_local_property