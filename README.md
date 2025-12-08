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

## Statement of need

`graph-pes` is a toolkit for building, training, and deploying machine-learned potential energy surfaces (PES) models that act on graph representations of atomic structures.

As a researcher who wants to train and use existing MLIPs, you can use the `graph-pes-train` command to train many different architectures from scratch on your own data, or fine-tune several existing foundation models. Once trained, you can use our drivers to run optimisations, single point energy calculations, and molecular dynamics simulations with your model with a variety of existing tools (`LAMMPS`, `ASE`, and `torch-sim`).

As a researcher wanting to work on MLIP methodology, `graph-pes` makes implementing new architectures easy, allows you to experiment with various different training strategies, and provides a clean, well-documented API for building things yourself.


## Features

- Experiment with new model architectures by inheriting from our `GraphPESModel` [base class](https://jla-gardner.github.io/graph-pes/models/root.html).
- [Train your own](https://jla-gardner.github.io/graph-pes/quickstart/implement-a-model.html) or existing model architectures (e.g., [SchNet](https://jla-gardner.github.io/graph-pes/models/many-body/schnet.html), [NequIP](https://jla-gardner.github.io/graph-pes/models/many-body/nequip.html), [PaiNN](https://jla-gardner.github.io/graph-pes/models/many-body/pinn.html), [MACE](https://jla-gardner.github.io/graph-pes/models/many-body/mace.html), [TensorNet](https://jla-gardner.github.io/graph-pes/models/many-body/tensornet.html), [OrB](https://jla-gardner.github.io/graph-pes/models/many-body/orb.html) etc.).
- Use and fine-tune foundation models via a unified interface: [MACE-MP0](https://jla-gardner.github.io/graph-pes/interfaces/mace.html), [MACE-OFF](https://jla-gardner.github.io/graph-pes/interfaces/mace.html), [MatterSim](https://jla-gardner.github.io/graph-pes/interfaces/mattersim.html), [GO-MACE](https://jla-gardner.github.io/graph-pes/interfaces/mace.html), [Egret](https://jla-gardner.github.io/graph-pes/interfaces/mace.html#graph_pes.interfaces.egret) and [Orb v2/3](https://jla-gardner.github.io/graph-pes/interfaces/orb.html).
- Easily configure distributed training, learning rate scheduling, weights and biases logging, and other features using our `graph-pes-train` [command line interface](https://jla-gardner.github.io/graph-pes/cli/graph-pes-train/root.html).
- Use our data-loading pipeline within your [own training loop](https://jla-gardner.github.io/graph-pes/quickstart/custom-training-loop.html).
- Run molecular dynamics simulations with any `GraphPESModel` using [torch-sim](https://jla-gardner.github.io/graph-pes/tools/torch-sim.html), [LAMMPS](https://jla-gardner.github.io/graph-pes/tools/lammps.html) or [ASE](https://jla-gardner.github.io/graph-pes/tools/ase.html)

## Quickstart

```bash
pip install -q graph-pes
wget https://tinyurl.com/graph-pes-minimal-config -O config.yaml
graph-pes-train config.yaml
```

Alternatively, for a 0-install quickstart experience, please see [this Google Colab](https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb), which you can also find in our [documentation](https://jla-gardner.github.io/graph-pes/quickstart/quickstart.html).


## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request on the [GitHub repository](https://github.com/jla-gardner/graph-pes). 
We use `uv` to manage dependencies and run commands. Install it [here](https://docs.astral.sh/uv/), and sync the dependencies using `uv sync --all-extras`.

Once you have made your changes, you can:

- run tests locally: `uv run pytest tests/`
- build the documentation: `uv run sphinx-build docs/source docs/build`

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## Citing `graph-pes`

We kindly ask that you cite `graph-pes` in your work if it has been useful to you. 
A manuscript is currently in preparation - in the meantime, please cite the Zenodo DOI found in the [CITATION.cff](CITATION.cff) file.
