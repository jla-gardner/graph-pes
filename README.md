<div align="center">
    <a href="https://jla-gardner.github.io/graph-pes/">
        <img src="docs/source/_static/logo-text.svg" width="90%"/>
    </a>

`graph-pes` is a framework built to accelerate the development of machine-learned potential energy surface (PES) models that act on graph representations of atomic structures.

Links: [Google Colab Quickstart](https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb) - [Documentation](https://jla-gardner.github.io/graph-pes/) - [PyPI](https://pypi.org/project/graph-pes/)

[![PyPI](https://img.shields.io/pypi/v/graph-pes)](https://pypi.org/project/graph-pes/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/graph-pes.svg)](https://github.com/conda-forge/graph-pes-feedstock)
[![Tests](https://github.com/jla-gardner/graph-pes/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/jla-gardner/graph-pes/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/jla-gardner/graph-pes/branch/main/graph/badge.svg)](https://codecov.io/gh/jla-gardner/graph-pes)
[![GitHub last commit](https://img.shields.io/github/last-commit/jla-gardner/load-atoms)]()

</div>


## Features

- Experiment with new model architectures by inheriting from our `GraphPESModel` [base class](https://jla-gardner.github.io/graph-pes/models/root.html).
- [Train your own](https://jla-gardner.github.io/graph-pes/quickstart/implement-a-model.html) or existing model architectures (e.g., [SchNet](https://jla-gardner.github.io/graph-pes/models/many-body/schnet.html), [NequIP](https://jla-gardner.github.io/graph-pes/models/many-body/nequip.html), [PaiNN](https://jla-gardner.github.io/graph-pes/models/many-body/pinn.html), [MACE](https://jla-gardner.github.io/graph-pes/models/many-body/mace.html), [TensorNet](https://jla-gardner.github.io/graph-pes/models/many-body/tensornet.html), etc.).
- Easily configure distributed training, learning rate scheduling, weights and biases logging, and other features using our `graph-pes-train` [command line interface](https://jla-gardner.github.io/graph-pes/cli/graph-pes-train.html).
- Use our data-loading pipeline within your [own training loop](https://jla-gardner.github.io/graph-pes/quickstart/custom-training-loop.html).
- Run molecular dynamics simulations via [LAMMPS](https://jla-gardner.github.io/graph-pes/tools/lammps.html) (or [ASE](https://jla-gardner.github.io/graph-pes/tools/ase.html)) using `pair_style graph_pes` and any `GraphPESModel` - all classes and functions in `graph-pes` are [TorchScriptable](https://pytorch.org/docs/stable/jit.html) 🔥

## Quickstart

```bash
pip install -q graph-pes
wget https://tinyurl.com/graph-pes-minimal-config -O config.yaml
graph-pes-train config.yaml
```

Alternatively, for a 0-install quickstart experience, please see [this Google Colab](https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb), which you can also find in our [documentation](https://jla-gardner.github.io/graph-pes/quickstart/quickstart.html).


## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request on the [GitHub repository](https://github.com/jla-gardner/graph-pes).
