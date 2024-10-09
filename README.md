<div align="center">
    <a href="https://jla-gardner.github.io/graph-pes/">
        <img src="docs/source/_static/logo-all.svg" width="90%"/>
    </a>
</div>



`graph-pes` is a framework built to accelerate the development of machine-learned potential energy surface (PES) models that act on graph representations of atomic structures.

## Features

- Experiment with new model architectures by inheriting from our `ConservativePESModel` base class.
- Train your own or existing models (e.g., SchNet, NequIP, PaiNN, MACE, etc.).
- Easily configure distributed training, learning rate scheduling, weights and biases logging, and other features using our `graph-pes-train` command line interface.
- Use our data-loading pipeline within your own training loop.
- Run molecular dynamics simulations via LAMMPS (or ASE) using any `ConservativePESModel` and the `pair_style graph_pes` LAMMPS command.

## Quickstart

For a 0-install quickstart experience, please see [this Google colab](TODO), which you can also find in our documentation.

Want to try this out locally? Run the following commands:

```bash
# optionally create a new environment
conda create -n graph-pes python=3.10
conda activate graph-pes

# install graph-pes
pip install graph-pes

# download a config file
curl -0 TODO

# train a model
graph-pes-train TODO
```



## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request on the [GitHub repository](https://github.com/jla-gardner/graph-pes).

## License

`graph-pes` is licensed under the [MIT License](https://github.com/jla-gardner/graph-pes/blob/main/LICENSE).

## Acknowledgments

`graph-pes` builds upon the following open-source projects:

- [PyTorch](https://pytorch.org/)
- [LAMMPS](https://lammps.org/)
- [ASE](https://wiki.fysik.dtu.dk/ase/)

We are grateful for the contributions of the developers and maintainers of these projects.

---