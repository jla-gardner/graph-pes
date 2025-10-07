################
Stillinger-Weber
################

Use this empirical model directly via the Python API:

.. code-block:: python

   from graph_pes.models import StillingerWeber
   model = StillingerWeber()
   model.predict_energy(graph)

   # or monatomic water
   model = StillingerWeber.monatomic_water()
   model.predict_energy(graph)

or within a ``graph-pes-train`` configuration file to :doc:`train a new model <../../cli/graph-pes-train/root>`.

.. code-block:: yaml

   model:
      +StillingerWeber:
        sigma: 3


Definition
----------

.. autoclass:: graph_pes.models.StillingerWeber
    :members: monatomic_water



