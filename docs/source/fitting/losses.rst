######
Losses
######

In ``graph-pes``, we distinguish between metrics and losses:

* A :class:`~graph_pes.training.loss.Loss` is some function that takes a model, a batch of graphs, and some predictions, and returns a scalar value measuring something that training should seek to minimise. 
  This could be a prediction error, a model weight penalty, or something else.
* A :class:`~graph_pes.training.loss.Metric` is some function that takes two tensors and returns a scalar value measuring the discrepancy between them.


Losses
======

.. autoclass:: graph_pes.training.loss.Loss
    :show-inheritance:
    :members: name, forward, required_properties

.. autoclass:: graph_pes.training.loss.PropertyLoss
    :show-inheritance:

.. autoclass:: graph_pes.training.loss.PerAtomEnergyLoss


Metrics
=======

.. class:: graph_pes.training.loss.Metric

    A type alias for any function that takes two input tensors
    and returns some scalar measure of the discrepancy between them.

    .. code-block:: python

        Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]



.. autoclass:: graph_pes.training.loss.RMSE()
.. autoclass:: graph_pes.training.loss.MAE()
.. autoclass:: graph_pes.training.loss.MSE()



    
Helpers
=======

.. autoclass:: graph_pes.training.loss.TotalLoss
    :show-inheritance:

.. class:: graph_pes.training.loss.MetricName

    A type alias for a ``Literal["RMSE", "MAE", "MSE"]``.

.. autoclass:: graph_pes.training.loss.WeightedLoss
    :members:
