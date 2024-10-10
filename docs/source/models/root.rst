.. _models:

######
Models
######

All models implemented in ``graph-pes`` are subclasses of
:class:`~graph_pes.ConservativePESModel`. Pass these models an :class:`~graph_pes.graphs.AtomicGraph` to get
the predicted total energy.
To get a full set of predictions (energy, forces, stress), use :func:`graph_pes.ConservativePESModel.get_predictions`, or its functional
counterpart :func:`graph_pes.get_predictions`.


.. autoclass:: graph_pes.ConservativePESModel
   :members: 
   :show-inheritance:
   :special-members: __call__


Loading Models
==============

.. autofunction:: graph_pes.models.load_model


Available Models
================

.. toctree::
   :maxdepth: 2

   functional
   addition
   offsets
   pairwise
   many-body/root
   