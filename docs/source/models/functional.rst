Functional Models
=================

If you have a function that takes an :class:`~graph_pes.graphs.AtomicGraph` 
and returns a scalar energy prediction, you can use ``graph-pes`` to get
force and stress predictions "for free" by wrapping it in a
:class:`~graph_pes.models.functional.FunctionalModel`:

.. autoclass:: graph_pes.models.functional.FunctionalModel()
   :members: forward
   :show-inheritance:

.. autofunction:: graph_pes.get_predictions
