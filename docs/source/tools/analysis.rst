########
Analysis
########


.. class:: graph_pes.utils.analysis.Transform

    Alias for ``Callable[[Tensor, AtomicGraph], Tensor]``.

    Transforms map a property, :math:`x`, to a target property, :math:`y`,
    conditioned on an :class:`~graph_pes.AtomicGraph`, :math:`\mathcal{G}`:

    .. math::

        T: (x; \mathcal{G}) \mapsto y

    
.. autofunction:: graph_pes.utils.analysis.parity_plot
.. autofunction:: graph_pes.utils.analysis.dimer_curve