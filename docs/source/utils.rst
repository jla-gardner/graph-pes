Utils
=====


Shift and Scale
---------------

.. autofunction:: graph_pes.utils.shift_and_scale.guess_per_element_mean_and_var


Sampling
--------

.. autoclass:: graph_pes.utils.sampling.T

.. autoclass:: graph_pes.utils.sampling.SequenceSampler
    :members:

Useful Types and Aliases
------------------------

.. class:: graph_pes.atomic_graph.PropertyKey

    Type alias for ``Literal["energy", "forces", "stress", "virial", "local_energies"]``.