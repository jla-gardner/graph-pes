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
 - name: Inorganic Chemistry Deparment, University of Oxford
   index: 1
date: 10/01/2025
bibliography: paper.bib
---

# Summary

TODO

# Statement of need

TODO

# Key references

`graph-pes` is driving a significant number of projects within the Deringer group, and has already been cited in `@Liu-24-12`.

The core functionality of `graph-pes` builds upon the
    - `PyTorch` `[@Paszke-19]`
    - `ase` `[@HjorthLarsen-17-06]`
`graph-pes` also builds upon the `e3nn` `[@Geiger-22-07]` package for implementing the `NequIP` `[@Batzner-22-05]` and `MACE` `[@Batatia-23-01]` architectures.

Other note-worthy softwares that offer similar functionality to `graph-pes` include:
- `nequip`
- `mace-torch`
- `deepmd`
- `schnetpack`
- `torchmd`

# Acknowledgements

Krystian Gierczak, Daniel Thomas du Toit and Zoé Faure Beaulieu for early testing and feedback.
