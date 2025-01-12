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

1. **the `graph_pes` Python package**: a modular Python framework containing all the functionality required to build and train graph-based ML-PES models. This includes a mature data pipeline for converting atomic structures into graph representations (`AtomicGraph`s), a fully featured ML-PES base class with automatic force and stress calcuations (`GraphPESModel`), and a suite of common data manipulations and model building blocks.

2. **the `graph-pes-train` command-line interface**: a unified tool for training graph-based ML-PES models on datasets of labelled atomic structures. Several popular model architectures are provided out of the box, but the interface is designed so that custom, end-user defined architectures can also easily be used (alongside custom loss functions, optimizers, datasets, etc.)

3. **the `pair style graph_pes` LAMMPS integration**: a pair style for using LAMMPS [@Thompson-22-02] to drive GPU-accelerated molecular dynamics (MD) simulations using any ML-PES model defined and trained using the `graph_pes` package.


# Statement of need

In recent years, machine-learned potential energy surfaces (ML-PESs) have become an indispensable tool for computational chemists. Their high accuracy, favourable scaling with system size, and ease of parallelisation allows ML-PES models to faithfully simulate the dynamics of large systems over long timescales. ML-PESs are thus facilitating the study of complex chemical phenomena at the atomic scale, in turn driving the generation of novel insight and understanding. [cite]

Many "flavours" of ML-PES exist, and with them have arisen a variety of softwares that are tailored to training specific architectures (see below). Given their unique specialisations, these varied softwares fail to conform to a common interface, making it difficult for practitioners to migrate their training and validation setups between different model architectures.

`graph_pes` provides a **unified interface and framework** for defining, training, and  working with all ML-PES models that act on graph-based representations of atomic structures. This unified interface has several advantages:

- **graph representations are completely general.** A chemical structure is completely defined by the positions of its atoms ($\mathbf{R}$), and their chemical identities ($Z$).[^1] A graph representation of the atomic structure incorporates this complete description, together with a neighbour list ($\mathbf{N}$) indicating which atoms are within the locality of others (for instance using a fixed cutoff radius). The resulting graph ($G = \{\mathbf{R}, Z, \mathbf{N}\}$), is thus an extremely general representation of chemical structure that does not impose any constraints on the architecture of the ML-PES model while easily facilitating the implementation of both local and non-local models.

[^1]: assuming that the structure is isolated, uncharged and in its electronic ground state. Defining a periodic structure requires the trivial addition of a unit cell and periodic boundary conditions.

- **transferable training and validation pipelines.** The unified interface of the `graph_pes.GraphPESModel` class and the `graph-pes-train` CLI tool allow researchers to easily transfer their training and validation pipelines between different model architectures. For convenience, we have implemented several popular model architectures out of the box (including PaiNN [@Schutt-21-06], NequIP [@Batzner-22-05], MACE [@Batatia-23-01] and TensorNet [@Simeon-23-06]). Training scripts require as little as 1 line of code to swap between model architecture, while LAMMPS input scripts require no changes other than pointing to the new model's file.

- **accelerated ML-PES development.** `graph-pes` provides all the functionality required to quickly design new ML-PES architectures, including common and well-documented: data manipulations, message passing operations and model building blocks such as distance expansions and neighbour summations. By inheriting from the `graph_pes.GraphPESModel` class, these new architectures can instantly be trained using the `graph-pes-train` CLI tool, and used to drive MD using the `pair style graph_pes` command.

ML-PES research is not just limited to the development of new model architectures. Among other important research topics, `graph-pes` provides salient features that are relevant to the following research topics:

- **model fine-tuning.** Various pre-train/fine-tune strategies have been proposed for improving the accuraacy and robustness of ML-PES models. Model fine-tuning in `graph-pes` can be performed using the `graph-pes-train` CLI tool, where users can optionally specify which of the model's parameters should be frozen during training.

- **universal force-fields.** A topical and recent area of research is the development of universal force-fields that can be used to describe the potential energy surface of a wide range of systems. `graph-pes` provides access to the `MACE-OFF` and `MACE-MP0` fuondation models for ease of use and fine-tuning. We use these models as an example of how to incorporate externally trained models into the `graph-pes` framework, either by porting the model's implementation into the `graph_pes` package, or by constructing a custom, lightweight interface to a 3rd party package (in this case, `mace-torch` [@Batatia-23-01]).

- **customised training procedures.** The `graph-pes-train` CLI tool supports user-defined optimizers, loss functions, and datasets through the flexible plugin system provided by the `data2objects` package. [@data2objects] Beyond this, we provide well documented examples of how to implement custom, architecture-agnostic training procedures in simple Python scripts.

- **beyond numerical validation.** While it is common to benchmark ML-PES models using numerical validation metrics (such as energy and force RMSEs), more extensive and physically motivated validation routines are important. The *unified*, *varied* and *architecture-agnostic* functionalities that `graph-pes` provide (including an `ase` Calculator interface and `LAMMPS` pair style) allow researchers to define a validation procedure once, and then use it for all their ML-PES models and architectures, and to share this validation procedure with the wider community. To aid in these validation procedures, we provide some utility functions for plotting dimer curves, energy volume scans and parity plots. 


# Related work

`graph-pes` is driving a significant number of projects within the Deringer group, and has already been cited in @Liu-24-12.

The core functionality of `graph-pes` builds upon several existing, open-source packages. We use the `Tensor` data structure, `nn` module and `autograd` functionality from `PyTorch` [@Paszke-19] for data representation, model definition and automatic differentiation respectively. We use the `PyTorch Lightning` [@Lightning] to implement our core training loop, as well as for advanced features such as learning rate scheduling, stochastic weight averaging and more.
We use the `ase` [@HjorthLarsen-17-06] package for reading serialised atomic structures from disc, and for converting them into graph representations. We use the `LAMMPS` [@Thompson-22-02] framework for creating the `pair style graph_pes` command for driving MD simulations.
We also use the `e3nn` [@Geiger-22-07] package for implementing the `NequIP` [@Batzner-22-05] and `MACE` [@Batatia-23-01] architectures.

Relevant packages that offer training and validation functionaltiy for specific ML-PES architectures include: `nequip` [@Batzner-22-05], `mace-torch` [@Batatia-23-01], `deepmd-kit` [@Wang-18-07; @Zeng-23-08], `schnetpack` [@schutt2019schnetpack; @schutt2023schnetpack] and `torchmd-net` [@TorchMDNet].


# Acknowledgements

We thank Krystian Gierczak, Daniel Thomas du Toit and Zoé Faure Beaulieu for early testing and feedback.


# References


