---
title: 'graph-pes: graph-based machine-learning models for potential energy surfaces'
tags:
  - Python
  - C++
  - machine-learning
  - force fields
  - molecular dynamics
  - graphs
  - chemistry
authors:
  - name: John L. A. Gardner
    orcid: 0009-0006-7377-7146
    affiliation: 1 
  - name: Volker L. Deringer
    orcid: 0000-0001-6873-0278
    affiliation: 1
affiliations:
 - name: Department of Chemistry, Inorganic Chemistry Laboratory, University of Oxford, Oxford OX1 3QR, United Kingdom
   index: 1
date: 10/01/2025
bibliography: paper.bib

---

# Summary

We present `graph-pes`, an open-source toolkit for accelerating the development, training and deployment of machine-learned potential energy surfaces (ML-PESs) that act on graph-based representations of atomic structures. The `graph-pes` toolkit comprises three components:

1. **the `graph_pes` Python package**: a modular Python package containing all the functionality required to build and train graph-based ML-PES models. This includes a mature data pipeline for converting atomic structures into graph representations (`AtomicGraph`s), a fully featured ML-PES base class with automatic force and stress calcuations (`GraphPESModel`), and a suite of common data manipulations and model building blocks.

2. **the `graph-pes-train` command-line interface**: a unified tool for training graph-based ML-PES models on datasets of labelled atomic structures. Several popular model architectures are provided out of the box, but the interface is designed so that custom, end-user defined architectures can also easily be used (alongside custom loss functions, optimizers, datasets, etc.)

3. **the `pair style graph_pes` LAMMPS integration**: a pair style for using LAMMPS [@Thompson-22-02] to perform GPU-accelerated molecular dynamics simulations using any ML-PES model defined and trained using the `graph_pes` package.


# Statement of need

In recent years, machine-learned potential energy surfaces (ML-PESs) have become an indispensable tool for computational chemists. Their high accuracy, favourable scaling with system size, and ease of parallelisation allows ML-PES models to faithfully simulate the dynamics of large systems over long timescales. ML-PESs are thus facilitating the study of complex chemical phenomena at the atomic scale, and hence the generation of novel insight and understanding. [cite]

<!-- A (ground-state, un-charged) chemical structure is completely defined by the positions of its atoms ($R$), and their chemical identities ($Z$). -->

Many "flavours" of ML-PES exist, and with them have arisen a variety of softwares that are tailored to training specific architectures (see below). Given their unique specialisations, these varied softwares fail to conform to a common interface, making it difficult for practitioners to migrate their training and validation setups between different architectures.

`graph_pes` provides a **unified interface and framework** for defining, training, and  working with all ML-PES models that act on graph-based representations of atomic structures. This unified interface has several advantages:

- **graph representations are genral and useful.** A chemical structure is completely defined by the positions of its atoms ($\mathbf{R}$), and their chemical identities ($Z$).[^1] A graph representation of the atomic structure incorporates this complete description, together with a neighbour list ($\mathbf{N}$) indicating which atoms are within the locality of others (for instance using a fixed cutoff radius). The resulting graph ($G = \{\mathbf{R}, Z, \mathbf{N}\}$), is thus an extremely general representation of chemical structure that does not impose any constraints on the architecture of the ML-PES model while easily facilitating the implementation of both local and non-local models.

[^1]: isolated structure only - ignoring cell and periodic boundary conditions. Plus gorund state, uncharged.

- **transferable training and validation pipelines.** The unified interface of the `graph_pes.GraphPESModel` class and the `graph-pes-train` CLI tool allow researchers to easily transfer their training and validation pipelines between different model architectures. For convenience, we have implemented several popular model architectures out of the box (including PaiNN, NequIP, MACE and TensorNet). Training scripts require as little as 2 lines of code to change to the model architecture. LAMMPS input scripts require no changes other than pointing to the new model's file.

- **accelerated ML-PES development.** `graph-pes` provides all the functionality required to quickly design new ML-PES architectures, including common data manipulations, message passing operations and common model building blocks such as distance expansions and neighbour summations. By inheriting from the `graph_pes.GraphPESModel` class, these new architectures can instantly be trained using the `graph-pes-train` CLI tool, and validated using the `pair style graph_pes` LAMMPS pair style.


ML-PES research is not just limited to the development of new model architectures. 


- advanced workflows: 
    - fine-tuning
    - foundation models
    - own training loop

- beyond numerical validation, generation of rdfs, dimer curves, energy volume scans, density isobars etc.: write validation procedure/script once, use forever.

- porting or interfacing

# Related work

`graph-pes` is driving a significant number of projects within the Deringer group, and has already been cited in @Liu-24-12.

The core functionality of `graph-pes` builds upon several existing, open-source packages:

- `PyTorch` [@Paszke-19]

- `ase` [@HjorthLarsen-17-06]

- `LAMMPS` [@Thompson-22-02]

We also use the `e3nn` [@Geiger-22-07] package for implementing the `NequIP` [@Batzner-22-05] and `MACE` [@Batatia-23-01] architectures.

Other note-worthy softwares that offer similar functionality to `graph-pes` include:

- `nequip`

- `mace-torch`

- `deepmd-kit`

- `schnetpack`

- `torchmd`


# Acknowledgements

We thank Krystian Gierczak, Daniel Thomas du Toit and Zoé Faure Beaulieu for early testing and feedback.


# References


