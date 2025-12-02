TensorNet
=========

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +TensorNet:
         channels: 32

Definition
----------

.. autoclass:: graph_pes.models.TensorNet
    :show-inheritance:

Components
----------

Below, we use the notation as taken from the `TensorNet paper <https://arxiv.org/abs/2306.06482>`__.

.. autoclass:: graph_pes.models.tensornet.ScalarOutput
.. autoclass:: graph_pes.models.tensornet.VectorOutput
    
