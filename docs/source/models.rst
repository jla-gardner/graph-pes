.. _models:

######
Models
######

All models implemented in ``graph_pes`` are subclasses of
:class:`~graph_pes.GraphPESModel`. Pass these models an :class:`~graph_pes.data.AtomicGraph` to get
the predicted total energy.
To get a full set of predictions (energy, forces, stress), use :func:`~graph_pes.core.get_predictions`.


.. autoclass:: graph_pes.GraphPESModel
   :members: 
   :show-inheritance:

.. autofunction:: graph_pes.core.get_predictions



Available Models
================

.. toctree::
   :maxdepth: 2

   models/pairwise
   models/schnet
   models/painn


Helper Classes and Functions
============================
.. toctree::
   :maxdepth: 2

   models/ensembling
   models/distances
   models/nn

