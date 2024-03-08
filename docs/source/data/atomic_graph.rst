
Atomic Graphs
=============

We describe atomic graphs using the :class:`~graph_pes.data.AtomicGraph` class.
For convenient ways to create instances of such graphs, see :func:`~graph_pes.data.to_atomic_graph`.


Definition
----------

.. autoclass:: graph_pes.data.AtomicGraph()
.. autoclass:: graph_pes.data.LabelledGraph()


Creation
--------

.. autofunction:: graph_pes.data.to_atomic_graph
.. autofunction:: graph_pes.data.to_atomic_graphs


Derived Properties
------------------

We define a number of derived properties of atomic graphs. These
also work for :class:`~graph_pes.data.AtomicGraphBatch` instances.

.. autofunction:: graph_pes.data.number_of_atoms
.. autofunction:: graph_pes.data.number_of_edges
.. autofunction:: graph_pes.data.has_cell
.. autofunction:: graph_pes.data.neighbour_vectors
.. autofunction:: graph_pes.data.neighbour_distances


Graph Operations
----------------

.. autofunction:: graph_pes.data.sum_over_neighbours
.. autofunction:: graph_pes.data.is_local_property