TensorMACE
##########

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +TensorMACE:
         elements: [Si, O]
         channels: 32
         hidden_irreps: "0e+1o+2e+3o"
         props: tensor
         target_method: "tensor_product"
         target_tensor_irreps: "0e+1e+2e"
         irrep_tp: "3o"
         number_of_tps: 8
         
         

Definition
----------

.. autoclass:: graph_pes.models.TensorMACE
.. autoclass:: graph_pes.models.ZEmbeddingTensorMACE
