EDDP
####

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:


.. code-block:: yaml

   model:
      +EDDP:
         elements: [H, C, N, O]
         cutoff: 5.0
         three_body_cutoff: 3.0

Definition
----------

.. autoclass:: graph_pes.models.EDDP
    :show-inheritance:
