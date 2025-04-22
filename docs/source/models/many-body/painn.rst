PaiNN
#####

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +PaiNN:
        channels: 32

Definition
----------

.. autoclass:: graph_pes.models.PaiNN
    :show-inheritance:
.. autoclass:: graph_pes.models.painn.Interaction
.. autoclass:: graph_pes.models.painn.Update
