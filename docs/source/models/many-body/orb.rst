Orb
###

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +Orb:
        channels: 32

Definition
----------

.. autoclass:: graph_pes.models.Orb
    :show-inheritance:


Helpers
-------

.. class:: graph_pes.models.orb.NormType

    A type alias for a ``Literal["layer", "rms"]``.


.. class:: graph_pes.models.orb.AttentionGate

    A type alias for a ``Literal["sigmoid", "softmax"]``.