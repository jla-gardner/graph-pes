MACE
####

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +MACE:
         elements: [H, C, N, O]
         

Definition
----------

.. autoclass:: graph_pes.models.MACE
.. autoclass:: graph_pes.models.ZEmbeddingMACE

``ScaleShiftMACE``?
-------------------

To replicate a ``ScaleShiftMACE`` model as defined in the reference `MACE <https://github.com/ACEsuit/mace>`_ implementation, you could use the following config:

.. code-block:: yaml

    model:
        offset:
            +LearnableOffset: {}
        many-body:
            +MACE:
                elements: [H, C, N, O]