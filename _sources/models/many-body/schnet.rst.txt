SchNet
======

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +SchNet:
         channels: 32

Definition
----------

.. autoclass:: graph_pes.models.SchNet
    :show-inheritance:

Components:
------------

.. autoclass:: graph_pes.models.schnet.SchNetInteraction
.. autoclass:: graph_pes.models.schnet.CFConv