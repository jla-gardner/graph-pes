from ase.io import read
from graph_pes.data.io import to_atomic_graphs
from graph_pes.data.loader import GraphDataLoader
from graph_pes.data.utils import random_split
from graph_pes.models.pairwise import LennardJones
from graph_pes.training.loss import MAE, RMSE, Loss

# 1. Define and initialize the model
model = LennardJones(epsilon=1.0, sigma=1.0)

# 2. Load the training data
structures = read("data/training_data.xyz", ":100")
graphs = to_atomic_graphs(structures, cutoff=5.0)
train, val = random_split(graphs, [90, 10], seed=42)

# 3. Define total loss
total_loss = Loss("energy", MAE()) * 10 + Loss("forces", RMSE())

# 4. Fit summary statistics and transforms
model._energy_summation.fit_to_graphs(train)
for loss in total_loss.losses:
    loss.fit_transform(train)

# 5. Train the model
train_loader, val_loader = GraphDataLoader(train), GraphDataLoader(val)

# typical PyTorch training loop
...
