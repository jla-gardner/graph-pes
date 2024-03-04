# `graph-pes` - Potential Energy Surfaces on Graphs

`graph-pes` is a Python framework for training Potential Energy Surface (PES) 
models that operate on graph representations of atomic structures.
Under-the-hood, this relies on Pytorch/Geometric for efficient tensor operations.

Batteries are included:
- **easy data manipulations** : see docs relating to `AtomicGraph`, intuitive batching, easy conversion from ase etc.
- **easy construction of PES models** : implement `predict_local_energies` , easy to save, load and share
- **useful primitives** : PerSpeciesParameter
- **easy training** : forces, energies, well conditioned losses etc.
- **analysis** : easy to plot, analyse and compare models

## Installation

```bash
pip install graph-pes
```

## Minimal example

```python
from ase.io import read
from graph_pes.data import convert_to_atomic_graphs
from graph_pes.models.pairwise import LennardJones
from graph_pes.training import train_model

# 1. load some (labelled) structures
structures = read("structures.xyz", index=":10")
assert "energy" in structures[0].info

# 2. convert to graphs (e.g. using a radius cutoff)
graphs = convert_to_atomic_graphs(structures, cutoff=5.0)

# 3. define the model
model = LennardJones()

# 4. train
train_model(model, graphs, max_epochs=100)
```