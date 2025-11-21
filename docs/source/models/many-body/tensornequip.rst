TensorNequIP
======

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +TensorNequIP:
         elements: [Si, O]
         features:
            channels: [64, 64, 64]
            l_max: 2
            use_odd_parity: true
         props: tensor
         target_tensor_irreps: 0e + 1e + 2e
         prune_last_layer: false
         target_method: "direct"


Definition
----------

.. autoclass:: graph_pes.models.TensorNequIP
.. autoclass:: graph_pes.models.ZEmbeddingTensorNequIP
