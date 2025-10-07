###########
Aggregation
###########

Aggregating some value over one's neighbours is a common operation in graph-based
ML models. ``graph-pes`` provides a base class for such operations, together with
a few common implementations. A common way to specify the aggregation mode to use
in a model is to use a :class:`~graph_pes.models.components.aggregation.NeighbourAggregationMode`
string, which internally is passed to :meth:`~graph_pes.models.components.aggregation.NeighbourAggregation.parse`.


Base Class
----------

.. autoclass:: graph_pes.models.components.aggregation.NeighbourAggregation
   :members:


.. class:: graph_pes.models.components.aggregation.NeighbourAggregationMode

   Type alias for ``Literal["sum", "mean", "constant_fixed", "constant_learnable", "sqrt"]``.


Implementations
---------------

.. autoclass:: graph_pes.models.components.aggregation.SumNeighbours
   :members:

.. autoclass:: graph_pes.models.components.aggregation.MeanNeighbours
   :members:

.. autoclass:: graph_pes.models.components.aggregation.ScaledSumNeighbours
   :members:

.. autoclass:: graph_pes.models.components.aggregation.VariancePreservingSumNeighbours
   :members:
