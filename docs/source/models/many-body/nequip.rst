NequIP
======

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +NequIP:
         elements: [H, C, N, O]
         features:
            channels: [64, 32, 8]
            l_max: 2
            use_odd_parity: true


Definition
----------

.. autoclass:: graph_pes.models.NequIP
.. autoclass:: graph_pes.models.ZEmbeddingNequIP

Utilities
---------

.. autoclass:: graph_pes.models.e3nn.nequip.SimpleIrrepSpec()
    :show-inheritance:
.. autoclass:: graph_pes.models.e3nn.nequip.CompleteIrrepSpec()
    :show-inheritance: