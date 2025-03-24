``mace-torch``
==============


``graph-pes`` supports the conversion of arbitrary ``mace-torch`` models to :class:`~graph_pes.GraphPESModel` objects via the :class:`~graph_pes.interfaces._mace.MACEWrapper` class.

We also provide two convenience functions to the recently trained ``MACE-MP`` and ``MACE-OFF`` "foundation" models, as well as the ``GO-MACE-23`` model.

If you use any ``mace-torch`` models in your work, please visit the `mace-torch <https://github.com/ACEsuit/mace-torch>`__ repository and cite the following:

.. code-block:: bibtex

    @inproceedings{Batatia2022mace,
        title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
        author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=YPpSngE-ZU}
    }

    @misc{Batatia2022Design,
        title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
        author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
        year = {2022},
        number = {arXiv:2205.06643},
        eprint = {2205.06643},
        eprinttype = {arxiv},
        doi = {10.48550/arXiv.2205.06643},
        archiveprefix = {arXiv}
    }



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

.. autofunction:: graph_pes.interfaces.mace_mp
.. autofunction:: graph_pes.interfaces.mace_off
.. autofunction:: graph_pes.interfaces.go_mace_23

.. autoclass:: graph_pes.interfaces._mace.MACEWrapper
