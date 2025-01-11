---
title: 'graph-pes: graph-based machine-learning models for potential energy surfaces'
tags:
  - Python
  - machine-learning
  - potential energy surfaces
  - molecular dynamics
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

`graph-pes` is an open-source toolkit that accelerates the development, training and deployment of machine-learned potential energy surfaces (ML-PESs) that act on graph-based representations of atomic structures. The `graph-pes` toolkit comprises three components:

1. **the `graph_pes` Python package**: a modular Python library containing all the functionality required to build and train graph-based ML-PES models. This includes a mature data pipeline for converting atomic structures into graph representations, a fully featured ML-PES base class with automatic force and stress inferences, and a suite of common data manipulation and model building blocks.

2. **the `graph-pes-train` command-line interface**: a unified tool for training graph-based ML-PES models on datasets of labelled atomic structures. Several popular model architectures are provided out of the box, but the interface is designed so that custom, end-user defined architectures can be used (alongside custom loss functions, optimizers, datasets, etc.)

3. **the `pair style graph_pes` LAMMPS integration**: a pair style for using LAMMPS to perform GPU-accelerated molecular dynamics simulations using any graph-based ML-PES model defined and trained using the `graph_pes` package.


# Statement of need


# Related work

`graph-pes` is driving a significant number of projects within the Deringer group, and has already been cited in @Liu-24-12.

The core functionality of `graph-pes` builds upon

- `PyTorch` [@Paszke-19]

- `ase` [@HjorthLarsen-17-06]

- `LAMMPS` 

`graph-pes` also builds upon the `e3nn` [@Geiger-22-07] package for implementing the `NequIP` [@Batzner-22-05] and `MACE` [@Batatia-23-01] architectures.

Other note-worthy softwares that offer similar functionality to `graph-pes` include:

- `nequip`

- `mace-torch`

- `deepmd-kit`

- `schnetpack`

- `torchmd`


# Acknowledgements

Krystian Gierczak, Daniel Thomas du Toit and Zoé Faure Beaulieu for early testing and feedback.


# References