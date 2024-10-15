Distance Expansions
===================

Available Expansions
--------------------

``graph-pes`` exposes the :class:`DistanceExpansion <graph_pes.models.components.distances.DistanceExpansion>` 
base class, which can be used to implement new distance expansions.
We also provide a few common expansions:

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