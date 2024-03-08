from ase.io import read
from graph_pes.analysis import parity_plot
from graph_pes.data import random_split, to_atomic_graphs
from graph_pes.models.zoo import LennardJones
from graph_pes.training import train_model
from graph_pes.transform import DividePerAtom

# 1. load some (labelled) structures
structures = read("structures.xyz", index=":")
assert "energy" in structures[0].info

# 2. convert to graphs (e.g. using a radius cutoff)
graphs = to_atomic_graphs(structures, cutoff=5.0)
train, val, test = random_split(graphs, [100, 25, 25])

# 3. define the model
model = LennardJones()

# 4. train
train_model(model, train, val, max_epochs=100)

# 5. evaluate
parity_plot(model, test, units="eV / atom", transform=DividePerAtom())
