<div align="center">
    <a href="https://jla-gardner.github.io/graph-pes/">
        <img src="docs/source/_static/logo-text.svg" width="90%"/>
    </a>
</div>

`graph-pes` is a framework built to accelerate the development of machine-learned potential energy surface (PES) models that act on graph representations of atomic structures.

<div align="center">

Links: [Google Colab Quickstart](https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb) - [Documentation](https://jla-gardner.github.io/graph-pes/) - [PyPI](https://pypi.org/project/graph-pes/)

</div>


## Features

- Experiment with new model architectures by inheriting from our `GraphPESModel` base class.
- Train your own or existing models (e.g., [SchNet](https://jla-gardner.github.io/graph-pes/models/many-body/schnet.html), [NequIP](https://jla-gardner.github.io/graph-pes/models/many-body/nequip.html), [PaiNN](https://jla-gardner.github.io/graph-pes/models/many-body/pinn.html), [MACE](https://jla-gardner.github.io/graph-pes/models/many-body/mace.html), etc.).
- Easily configure distributed training, learning rate scheduling, weights and biases logging, and other features using our `graph-pes-train` [command line interface](https://jla-gardner.github.io/graph-pes/cli/graph-pes-train.html).
- Use our data-loading pipeline within your [own training loop](https://jla-gardner.github.io/graph-pes/quickstart/custom-training-loop.html).
- Run molecular dynamics simulations via [LAMMPS](https://jla-gardner.github.io/graph-pes/tools/lammps.html) (or [ASE](https://jla-gardner.github.io/graph-pes/tools/ase.html)) using any `GraphPESModel` and the `pair_style graph_pes` LAMMPS command.

## Quickstart

```bash
pip install graph-pes
wget https://tinyurl.com/graph-pes-quickstart-cgap17
graph-pes-train quickstart-cgap17.yaml
```

Alternatively, for a 0-install quickstart experience, please see [this Google Colab](https://colab.research.google.com/github/jla-gardner/graph-pes/blob/main/docs/source/quickstart/quickstart.ipynb), which you can also find in our [documentation](https://jla-gardner.github.io/graph-pes/quickstart/quickstart.html).


## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request on the [GitHub repository](https://github.com/jla-gardner/graph-pes).
