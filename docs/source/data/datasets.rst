Datasets
========

TODO: Write me

.. autoclass:: graph_pes.data.GraphDataset

.. autofunction:: graph_pes.data.load_atoms_dataset


.. autoclass:: graph_pes.data.SequenceDataset
    :members:

.. autoclass:: graph_pes.data.ShuffledDataset
    :members:

.. autoclass:: graph_pes.data.FittingData()
    :members:



.. class:: graph_pes.data.loader.GraphDataLoader

    A data loader for merging :class:`~graph_pes.AtomicGraph` objects
    into batches.

    Parameters
    ++++++++++
    **dataset** (:class:`~graph_pes.data.GraphDataset` | Sequence[:class:`~graph_pes.AtomicGraph`]) - the dataset to load.
    
    **batch_size** (:class:`int`) - the batch size.
    
    **shuffle** (:class:`bool`) - whether to shuffle the dataset.
    
    **kwargs** - additional keyword arguments to pass to the underlying
    :class:`torch.utils.data.DataLoader`.