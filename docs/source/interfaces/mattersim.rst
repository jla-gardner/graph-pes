``mattersim``
=============


``graph-pes`` allows you fine-tune and use the ``mattersim`` series of models in the same way as any other :class:`~graph_pes.GraphPESModel`, either via the Python API:


.. code-block:: python

   from graph_pes.interfaces import mattersim
   model = mattersim("mattersim-v1.0.0-1m")
   model.predict_energy(graph)

... or within a ``graph-pes-train`` configuration file:

.. code-block:: yaml

   model:
      +mattersim:
         load_path: "mattersim-v1.0.0-5m"



If you use any ``mattersim`` models in your work, please visit the `mattersim <https://github.com/microsoft/mattersim/tree/main>`__ repository and cite the following:

.. code-block:: bibtex

    @article{yang2024mattersim,
        title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
        author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
        year={2024},
        eprint={2405.04967},
        archivePrefix={arXiv},
        primaryClass={cond-mat.mtrl-sci},
        url={https://arxiv.org/abs/2405.04967},
        journal={arXiv preprint arXiv:2405.04967}
    }




Installation
------------

To install ``graph-pes`` with support for ``mattersim`` models, you need to install
the `mattersim <https://github.com/microsoft/mattersim/tree/main>`__ package. We recommend doing this in a new environment:

.. code-block:: bash

   conda create -n graph-pes-mattersim python=3.9
   conda activate graph-pes-mattersim
   pip install graph-pes
   pip install --upgrade mattersim


Interface
---------

.. autofunction:: graph_pes.interfaces.mattersim

