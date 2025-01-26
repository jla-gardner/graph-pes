Callbacks
=========

We have implemented a few useful `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/index.html>`_ callbacks that you can use to monitor your training process: 

.. autoclass:: graph_pes.training.callbacks.WandbLogger

.. autoclass:: graph_pes.training.callbacks.OffsetLogger

.. autoclass:: graph_pes.training.callbacks.ScalesLogger

.. autoclass:: graph_pes.training.callbacks.DumpModel

.. autoclass:: graph_pes.training.callbacks.ModelTimer

Base class
----------

.. autoclass:: graph_pes.training.callbacks.GraphPESCallback