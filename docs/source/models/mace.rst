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
            graph_pes.models.LearnableOffset: {}
        many-body:
            graph_pes.models.MACE:
                elements: [H, C, N, O]
                ...