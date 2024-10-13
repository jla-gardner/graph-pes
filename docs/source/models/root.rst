.. _models:

######
Models
######

All models implemented in ``graph-pes`` are subclasses of
:class:`~graph_pes.core.GraphPESModel`. Implementations should override the
:meth:`~graph_pes.core.GraphPESModel.predict` method.

Inheriting from :class:`~graph_pes.core.LocalEnergyModel` changes this requirement
to implement the :meth:`~graph_pes.core.LocalEnergyModel.predict_raw_energies`
method instead.


.. autoclass:: graph_pes.core.GraphPESModel
   :members:
   :show-inheritance:
   :exclude-members: get_extra_state, set_extra_state


.. autoclass:: graph_pes.core.LocalEnergyModel
   :members: predict_raw_energies
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
   