.. _models:


Models
######


.. autoclass:: graph_pes.GraphPESModel
   :members:
   :show-inheritance:


.. autoclass:: graph_pes.GraphPropertyModel
   :members:
   :show-inheritance:


Loading Models
==============

.. autofunction:: graph_pes.models.load_model
.. autofunction:: graph_pes.models.load_model_component

Freezing Models
===============

.. class:: graph_pes.models.T 

   Type alias for ``TypeVar("T", bound=torch.nn.Module)``.

.. autofunction:: graph_pes.models.freeze
.. autofunction:: graph_pes.models.freeze_matching
.. autofunction:: graph_pes.models.freeze_all_except
.. autofunction:: graph_pes.models.freeze_any_matching


Available Models
================

.. toctree::
   :maxdepth: 2

   addition
   offsets
   pairwise
   many-body/root


Unit Conversion
===============

.. autoclass:: graph_pes.models.UnitConverter
   :show-inheritance:
