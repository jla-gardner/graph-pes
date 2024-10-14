Batching
========

Definition
----------
.. autoclass:: graph_pes.graphs.AtomicGraphBatch()
.. autoclass:: graph_pes.graphs.LabelledBatch()

Creation
--------
.. autofunction:: graph_pes.graphs.operations.to_batch

.. class:: graph_pes.data.loader.GraphDataLoader

    A data loader for merging :class:`~graph_pes.graphs.AtomicGraph` objects
    into :class:`~graph_pes.graphs.AtomicGraphBatch` objects.

    Parameters
    ++++++++++
    **dataset** (:class:`~graph_pes.data.dataset.LabelledGraphDataset` | :class:`~graph_pes.graphs.LabelledGraph`) - the dataset to load.
    
    **batch_size** (:class:`int`) - the batch size.
    
    **shuffle** (:class:`bool`) - whether to shuffle the dataset.
    
    **kwargs** - additional keyword arguments to pass to the underlying
    :class:`torch.utils.data.DataLoader`.

Batch Operations
----------------
.. autofunction:: graph_pes.graphs.operations.is_batch
.. autofunction:: graph_pes.graphs.operations.sum_per_structure
.. autofunction:: graph_pes.graphs.operations.number_of_structures

