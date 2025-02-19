MACE
####


MACE Models
===========

.. autoclass:: graph_pes.models.MACE
.. autoclass:: graph_pes.models.ZEmbeddingMACE

``ScaleShiftMACE``?
===================

To replicate a ``ScaleShiftMACE`` model as defined in the reference `MACE <https://github.com/ACEsuit/mace>`_ implementation, you could use the following config:

.. code-block:: yaml

    model:
        offset:
            +LearnableOffset: {}
        many-body:
            +MACE:
                elements: [H, C, N, O]
                ...