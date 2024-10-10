Functional Models
=================


If you have a function that takes an :class:`~graph_pes.graphs.AtomicGraph` and returns a scalar, you can use it as an energy model
by wrapping it in :class:`~graph_pes.core.FunctionalModel`:

.. autoclass:: graph_pes.core.FunctionalModel()
   :members: forward
   :show-inheritance:

.. autofunction:: graph_pes.get_predictions
