``mace-torch``
==============

Installation
------------

To install ``graph-pes`` with support for MACE models, you need to install
the `mace-torch <https://github.com/ACEsuit/mace-torch>`__ package. We recommend doing this in a new environment:

.. code-block:: bash

   conda create -n graph-pes-mace python=3.10
   conda activate graph-pes-mace
   pip install mace-torch graph-pes


Interface
---------

.. autoclass:: graph_pes.interfaces.mace.MACEWrapper
