###########
Transforms
###########

.. Transforming properties defined on graphs has several useful properties
.. when training PES models. For instance, adding a per-species shift to 
.. energy predictions...

.. autoclass :: graph_pes.transform.Transform
   :members:
   :private-members:

Available Transforms
====================

.. autoclass :: graph_pes.transform.PerAtomShift
   :members:
.. autoclass :: graph_pes.transform.PerAtomScale
   :members:


Useful Other Transforms
=======================

.. autoclass :: graph_pes.transform.DividePerAtom()
.. autoclass :: graph_pes.transform.Chain
.. autoclass :: graph_pes.transform.Identity()




