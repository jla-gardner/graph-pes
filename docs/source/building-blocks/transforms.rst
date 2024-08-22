Transforms
==========

.. class:: graph_pes.transform.Transform

    Type alias for ``Callable[[Tensor, AtomicGraph], Tensor]``.

    Transforms map a property, :math:`x`, to a target property, :math:`y`,
    conditioned on an :class:`~graph_pes.graphs.AtomicGraph`, :math:`\mathcal{G}`:

    .. math::

        T: (x; \mathcal{G}) \mapsto y

.. autofunction:: graph_pes.transform.identity
.. autofunction:: graph_pes.transform.divide_per_atom