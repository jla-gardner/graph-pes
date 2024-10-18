.. _models:

######
Models
######

All models implemented in ``graph-pes`` are subclasses of
:class:`~graph_pes.core.GraphPESModel`. Implementations should override the
:meth:`~graph_pes.core.GraphPESModel.forward` method.


.. autoclass:: graph_pes.core.GraphPESModel
   :members:
   :show-inheritance:



Loading Models
==============

.. autofunction:: graph_pes.models.load_model


Available Models
================

.. toctree::
   :maxdepth: 2

   addition
   offsets
   pairwise
   many-body/root
   