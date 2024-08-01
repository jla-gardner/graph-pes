.. _models:

######
Models
######

All models implemented in ``graph_pes`` are subclasses of
:class:`~graph_pes.GraphPESModel`. Pass these models an :class:`~graph_pes.graphs.AtomicGraph` to get
the predicted total energy.
To get a full set of predictions (energy, forces, stress), use :func:`~graph_pes.core.get_predictions`.


.. autoclass:: graph_pes.GraphPESModel
   :members: 
   :show-inheritance:
   :special-members: __call__

.. autofunction:: graph_pes.core.get_predictions



Available Models
================

.. toctree::
   :maxdepth: 3

   offsets
   pairwise
   many-body/root
   
