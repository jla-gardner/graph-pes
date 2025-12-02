Distance Expansions
===================

Available Expansions
--------------------

``graph-pes`` exposes the :class:`~graph_pes.models.components.distances.DistanceExpansion` 
base class, together with implementations of a few common expansions:

.. autoclass:: graph_pes.models.components.distances.Bessel
    :show-inheritance:
.. autoclass:: graph_pes.models.components.distances.GaussianSmearing
    :show-inheritance:
.. autoclass:: graph_pes.models.components.distances.SinExpansion
    :show-inheritance:
.. autoclass:: graph_pes.models.components.distances.ExponentialRBF
    :show-inheritance:


Implementing a new Expansion
----------------------------

.. autoclass:: graph_pes.models.components.distances.DistanceExpansion
    :members: