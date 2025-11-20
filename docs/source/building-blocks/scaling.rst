Scaling
=======

A commonly used strategy in models of the PES is to scale the raw local energy predictions
by some scale parameter (derived in a :meth:`graph_pes.GraphPropertyModel.pre_fit` step). This has the 
effect of allowing models to output ~unit normally distributed predictions (which is often 
an implicit assumption of e.g. NN components) before having these scaled to the natural scale of the 
labels in question.

.. autoclass:: graph_pes.models.components.scaling.LocalEnergiesScaler
    :members:
