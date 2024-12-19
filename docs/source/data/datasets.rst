Datasets
========

:class:`~graph_pes.data.GraphDataset`\ s are collections of :class:`~graph_pes.AtomicGraph`\ s.
We provide a base class, :class:`~graph_pes.data.GraphDataset`, together with several 
implementations. The most common way to get a dataset of graphs is to use 
:func:`~graph_pes.data.load_atoms_dataset` or :func:`~graph_pes.data.file_dataset`.

Useful Datasets
---------------

.. autofunction:: graph_pes.data.load_atoms_dataset

.. autofunction:: graph_pes.data.file_dataset

Base Classes
-------------

.. autoclass:: graph_pes.data.GraphDataset()
    :show-inheritance:
    :members:
    :special-members: __len__, __getitem__, __iter__

.. autoclass:: graph_pes.data.ASEToGraphDataset()
    :show-inheritance:

.. autoclass:: graph_pes.data.DatasetCollection()
    :show-inheritance:


Utilities
---------

.. autoclass:: graph_pes.data.ase_db.ASEDatabase
    :show-inheritance:
    :members:
