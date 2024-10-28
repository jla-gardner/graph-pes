Datasets
========

:class:`~graph_pes.data.GraphDataset`\ s are collections of :class:`~graph_pes.AtomicGraph`\ s.
We provide a base class, :class:`~graph_pes.data.GraphDataset`, together with several 
implementations. The most common way to get a dataset of graphs is to use one of 
:func:`~graph_pes.data.load_atoms_dataset`, :func:`~graph_pes.data.file_dataset` or 
:class:`~graph_pes.data.ASEDataset`.

Useful Datasets
---------------

.. autofunction:: graph_pes.data.load_atoms_dataset

.. autofunction:: graph_pes.data.file_dataset

.. autoclass:: graph_pes.data.ASEDataset
    :show-inheritance:

Base Classes
-------------

.. autoclass:: graph_pes.data.datasets.T

.. autoclass:: graph_pes.data.SizedDataset()
    :show-inheritance:
    :special-members: __len__, __getitem__, __iter__

.. autoclass:: graph_pes.data.GraphDataset()
    :show-inheritance:
    :members:
    :special-members: __len__, __getitem__, __iter__

.. autoclass:: graph_pes.data.SequenceDataset
    :show-inheritance:

.. autoclass:: graph_pes.data.ShuffledDataset
    :show-inheritance:

.. autoclass:: graph_pes.data.ReMappedDataset
    :show-inheritance:

.. autoclass:: graph_pes.data.FittingData()
    :show-inheritance:



