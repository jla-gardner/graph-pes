Example configs
===============

Realistic config
----------------

A realistic config for training a :class:`~graph_pes.models.MACE` model on the `C-GAP-20U dataset <https://jla-gardner.github.io/load-atoms/datasets/C-GAP-20U.html>`__:

.. literalinclude:: ../../../../configs/realistic.yaml
    :language: yaml



Kitchen sink config
-------------------

A `"kitchen sink"` config that attempts to specify every possible option:

.. literalinclude:: ../../../../configs/kitchen-sink.yaml
    :language: yaml


Default config
--------------

For reference, here are the default config options used in ``graph-pes-train``:

.. literalinclude:: ../../../../src/graph_pes/config/training-defaults.yaml
    :language: yaml
