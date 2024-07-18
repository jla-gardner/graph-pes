import ase.io
from graph_pes.analysis import parity_plot
from graph_pes.data.io import to_atomic_graphs
from graph_pes.data.utils import random_split
from graph_pes.models import LennardJones
from graph_pes.training.manual import train_the_model
from graph_pes.transform import divide_per_atom

# 1. load some (labelled) structures using ASE
structures = ase.io.read("structures.xyz", index=":")
assert isinstance(structures, list) and "energy" in structures[0].info

# 2. convert to graphs using a radius cutoff
graphs = to_atomic_graphs(structures, cutoff=5.0)
train, val, test = random_split(graphs, [100, 25, 25])

# 3. define the model
model = LennardJones()

# 4. train
train_the_model(model, train, val, trainer_options=dict(max_epochs=100))

# 5. evaluate
parity_plot(model, test, units="eV / atom", transform=divide_per_atom)
