---
title: 'graph-pes: graph-based machine-learning models for potential-energy surfaces'
tags:
  - Python
  - C++
  <! -- where do we use C++? may be obvious to experts but interesting to mention in the text briefly for non-experts? -->
  - machine learning
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
 - name: Inorganic Chemistry Laboratory, Department of Chemistry, University of Oxford, Oxford OX1 3QR, United Kingdom
   index: 1
date: 10/01/2025
bibliography: paper.bib

---

# Summary

We present `graph-pes`, an open-source toolkit for accelerating the development, training, and deployment of machine-learned potential-energy surface (ML-PES) models that act on graph-based representations of atomic structures. The `graph-pes` toolkit comprises three components:

1. **The `graph_pes` Python package**: a modular Python framework containing all the functionality required to build, train, and evaluate graph-based PES models. The package includes a mature data pipeline for converting atomic structures into graph representations (`AtomicGraph`s), a fully featured ML-PES base class with automatic force and stress calcuations (`GraphPESModel`), and a suite of common data manipulation routines and model building blocks.

2. **The `graph-pes-train` command-line interface** (CLI): a unified tool for training graph-based ML-PES models on datasets of labelled atomic structures. Several popular model architectures are provided out of the box, but the interface is designed so that custom, end-user defined architectures can also easily be used (alongside custom loss functions, optimisers, datasets, etc.).
<!-- unclear (to me) here whether the Python pacakge also allows for training, or training must be with the CLI. Is this a bit like quip/quippy (you can call quip on the command line and this is very computationally efficient, but people may prefer using the quippy Python wrappers which make it much easier to develop)? -->

3. **Molecular-dynamics drivers** for popular MD engines that allow any ML-PES model that has been defined and trained using `graph_pes` to be used in GPU-accelerated MD simulations. These drivers currently include a `pair style` for use in LAMMPS, [@Thompson-22-02] a `GraphPESCalculator` for use in ASE, [@Larsen-17-06] and an integration with the `torch-sim` package. [@torch-sim]


# Statement of need

In recent years, machine-learned PES models, commonly referred to as machine-learned interatomic potentials (MLIPs), have become central tools for computational chemistry and materials science. 
<!-- cite Adv. Mater. 2019 -->
These models are trained on quantum-mechanical data, but scale much more favourably with system size, and so they make it possible to simulate the dynamics of large systems (millions of atoms and more) over extended timescales. In this way, MLIPs are facilitating the study of complex chemical phenomena at the atomic scale, in turn driving the generation of novel insight and understanding. 

Many "flavours" of ML-PES exist, and with them have arisen a variety of software packages that are tailored to training specific architectures (see below). Given their unique specialisations, these individual software implementations do not normally conform to a common interface, making it difficult for practitioners to migrate their training and validation pipelines between different model architectures.

`graph_pes` provides a **unified interface and software framework** for defining, training, and  working with all ML-PES models that act on graph-based representations of atomic structures. This unified interface has several advantages:

- **Completely general graph representations.** A chemical structure is completely defined by the positions of its atoms ($\mathbf{R}$), and their chemical identities ($Z$).[^1] A graph representation of the atomic structure incorporates this complete description, together with a neighbour list ($\mathbf{N}$) indicating which atoms are within the local environment of others (defined, for instance, using a fixed cut-off radius). The resulting graph, $G = \{\mathbf{R}, Z, \mathbf{N}\}$, is thus an extremely general representation of chemical structure that does not impose any constraints on the architecture of the ML-PES model while easily facilitating the implementation of both local and non-local models.
<!-- I think this needs brief explanation, i.e. what do we mean by non-local?-->

[^1]: assuming that the structure is isolated, uncharged, and in its electronic ground state. Defining a periodic structure requires the trivial addition of a unit cell and periodic boundary conditions.

- **Transferable training and validation pipelines.** The unified interface of the `graph_pes.GraphPESModel` class and the `graph-pes-train` CLI tool allow researchers to easily transfer their training and validation pipelines between different model architectures. For convenience, we have implemented several popular model architectures out of the box (including PaiNN [@Schutt-21-06], EDDP [@Pickard-22-07], NequIP [@Batzner-22-05], MACE [@Batatia-23-01] and TensorNet [@Simeon-23-06]). 
<!-- can we make clear here that this is a full new implementation based on e3nn, not just an interface -->
Training scripts require as little as 1 line of code to swap between model architecture, while LAMMPS input scripts, ASE calculators, and `torch-sim` simulations require no changes other than pointing to the new model's file.

- **Accelerated MLIP development.** `graph-pes` provides all the functionality required to quickly adapt MLIP architectures and even design new ones from scratch, including common and well-documented: data manipulations, message passing operations and model building blocks such as distance expansions and neighbour summations. 
<!-- unclear here - language-wise it sounds like the "common and well-documented" refers to the architectures, but that sounds wrong -->
By inheriting from the `graph_pes.GraphPESModel` class, these new architectures can instantly be trained using the `graph-pes-train` CLI tool, and used to drive MD and other tasks, for example, using the `pair style graph_pes` command in LAMMPS.

Research in the field of MLIPs is not just limited to the development of new model architectures. Among other important research topics, `graph-pes` provides salient features that are relevant to the following research directions:


- **Customised training procedures.** The `graph-pes-train` CLI tool supports user-defined optimisers, loss functions, and datasets through the flexible plugin system provided by the `data2objects` package. [@data2objects] Beyond this, we provide well-documented examples of how to implement custom, architecture-agnostic training procedures in simple Python scripts.
<!-- re-ordering slightly. I think customised training is more closely related to the earlier part, and it might flow better this way, but feel free to revert -->

- **Model fine-tuning.** Various pre-training/fine-tuning strategies have been proposed for improving the accuracy and robustness of MLIPs. 
<!-- cite MLST 2024, one other? --> 
Model fine-tuning in `graph-pes` can be performed using the `graph-pes-train` CLI tool, where users can optionally specify which of the model's parameters should be frozen during training, and easily account for changes in the level of theory used to label the training data.
<!-- also mention FT is critically important for foundational models? -->


- **Universal or foundational MLIPs.** A topical and recent area of research is the development of universal force-fields that can be used to describe the potential energy surface of a wide range of systems. `graph-pes` integrates directly with the `mace-torch`, `mattersim` and `orb-models` packages to provide access to, among others, the `MACE-OFF`, `MACE-MP`, `GO-MACE`, `Egret-v1`, `MatterSim`, `orb-v2`, and `orb-v3` families of models. [@Kovacs-25-01, @Batatia-24-03, @El-Machachi-24, @Mann-25-05, @Yang-24-05, @Neumann-24-10, @Rhodes-25-04] Each of these integrations generates `GraphPESModels` that are directly compatible with all `graph-pes` features, including fine-tuning, validation pipelines, and MD simulations.
<!-- this looks quite bulky with their citation style and doesn't directly map names of architectures onto papers. is it worth having a table with model names and references (if easyt to do)? -->


- **Beyond numerical validation.** While it is common to benchmark ML-PES models using numerical validation metrics (such as energy and force RMSEs), more extensive and physically motivated validation routines are important before an MLIP model can be confidently used in practice. 
<!-- cite JCP tutorial -->
The *unified*, *varied*, 
<!-- unclear what is meant by varied here -->
and *architecture-agnostic* functionalities that `graph-pes` provides allow researchers to define a validation procedure once, and then use it for all their MLIP architectures and specific models, and to share this validation procedure with the wider community.


# Related work

`graph-pes` is beginning to drive a substantial number of projects within our research group, and we hope that it will be useful to many others. In recent preprints, we have described the use of `graph-pes` for fitting NequIP models to datasets created using the `autoplex` software [@Liu-24-12], and for assessing the zero-shot performance of different graph-network MLIP models [@Mahmoud-25-02].

The core functionality of `graph-pes` builds upon several existing, open-source packages. We use the `Tensor` data structure, `nn` module and `autograd` functionality from `PyTorch` [@Paszke-19] for data representation, model definition and automatic differentiation respectively. We use the `PyTorch Lightning` [@Lightning] to implement our core training loop, as well as for advanced features such as learning rate scheduling, stochastic weight averaging and more.
We use the `ase` [@HjorthLarsen-17-06] package for reading serialised atomic structures from disc, and for converting them into graph representations. We use the `LAMMPS` [@Thompson-22-02] framework for creating the `pair style graph_pes` command for driving MD simulations.
We also use the `e3nn` [@Geiger-22-07] package for implementing the `NequIP` [@Batzner-22-05] and `MACE` [@Batatia-23-01] architectures.

Relevant packages that offer training and validation functionaltiy for specific ML-PES architectures include: `nequip` [@Batzner-22-05], `mace-torch` [@Batatia-23-01], `deepmd-kit` [@Wang-18-07; @Zeng-23-08], `schnetpack` [@schutt2019schnetpack; @schutt2023schnetpack] and `torchmd-net` [@TorchMDNet].
<!-- I wonder if the previous two paragraphs should go under a different heading -- e.g. the first paragraph could be a "Future Directions" like section (if JOSS style allows)? The last part in fact I think is the one where we should compare/contrast -->


# Acknowledgements

We thank Krystian Gierczak, Daniel Thomas du Toit, and Zoé Faure Beaulieu for early testing and feedback.
This work was supported by UK Research and Innovation [grant number EP/X016188/1].
<!-- order alphabetically? unless this is chronological? -->
<!-- please add also funding acknowledgements for the Year 1-3 funding if the work on this started prior to MT 2024 -->

# References


