
Atomic Graphs
=============

We describe atomic graphs using the :class:`~graph_pes.graphs.AtomicGraph` class.
For convenient ways to create instances of such graphs from :class:`~ase.Atoms` objects,
see :func:`~graph_pes.data.io.to_atomic_graph`.


Definition
----------

.. autoclass:: graph_pes.graphs.AtomicGraph()
.. autoclass:: graph_pes.graphs.LabelledGraph()

.. class:: graph_pes.graphs.keys.LabelKey

    Type alias for ``Literal["energy", "forces", "stress"]``.

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
.. autofunction:: graph_pes.graphs.operations.number_of_neighbours
.. autofunction:: graph_pes.graphs.operations.available_labels


Graph Operations
----------------

We define a number of operations that act on :class:`torch.Tensor` instances conditioned on the graph structure.
All of these are fully compatible with :class:`~graph_pes.graphs.AtomicGraphBatch` instances, and with ``TorchScript`` compilation.

.. autofunction:: graph_pes.graphs.operations.trim_edges
.. autofunction:: graph_pes.graphs.operations.sum_over_neighbours
.. autofunction:: graph_pes.graphs.operations.index_over_neighbours
.. autofunction:: graph_pes.graphs.operations.is_local_property
.. autofunction:: graph_pes.graphs.operations.guess_per_element_mean_and_var
