Functional Models
=================

If you have a function that takes an :class:`~graph_pes.graphs.AtomicGraph` 
and returns a prediction of the local energies, you can use ``graph-pes`` to get
force and stress predictions "for free" by wrapping it in a
:class:`~graph_pes.models.functional.FunctionalModel`:

.. autoclass:: graph_pes.models.functional.FunctionalModel()
   :show-inheritance:

