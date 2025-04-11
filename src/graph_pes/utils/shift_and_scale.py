from __future__ import annotations

from typing import Sequence

import ase
import numpy as np
import torch
from ase.data import chemical_symbols
from sklearn.linear_model import Ridge

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_structures,
    sum_per_structure,
    to_batch,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.nn import PerElementParameter

from .logger import logger


def guess_per_element_mean_and_var(
    per_structure_quantity: torch.Tensor,
    batch: AtomicGraph,
    min_variance: float = 0.01,
) -> tuple[dict[int, float], dict[int, float]]:
    r"""
    Guess the per-element mean (:math:`\mu_Z`) and variance (:math:`\sigma_Z^2`)
    of a per-structure quantity using ridge regression under the following assumptions:

    1. the per-structure property, :math:`P`, is a summation over local
       properties of its components atoms: :math:`P = \sum_{i=1}^{N} p_{Z_i}`.
    2. the per-atom properties, :math:`p_{Z_i}`, are independent and identically
       distributed (i.i.d.) for each atom of type :math:`Z_i` according to a
       normal distribution: :math:`p_{Z_i} \sim \mathcal{N}(\mu_{Z_i}, \sigma_{Z_i}^2)`.

    Parameters
    ----------
    per_structure_quantity
        The per-structure quantity to guess the per-element mean and variance of.
    batch
        The batch of graphs to use for guessing the per-element mean and variance.

    Returns
    -------
    means
        A dictionary mapping atomic numbers to per-element means.
    variances
        A dictionary mapping atomic numbers to per-element variances.
    """  # noqa: E501

    # extract the atomic numbers to tensor N such that:
    # N[structure, Z] is the number of atoms of atomic number Z in structure
    unique_Zs = torch.unique(batch.Z)  # (n_Z,)
    N = torch.zeros(number_of_structures(batch), len(unique_Zs))  # (batch, n_Z)
    for i, Z in enumerate(unique_Zs):
        N[:, i] = sum_per_structure((batch.Z == Z).float(), batch)

    if np.linalg.matrix_rank(N.numpy()) < N.shape[1]:
        logger.warning(
            """\
We are attempting to guess the mean per-element
contribution for a per-structure quantity (usually
the total energy). 

However, the composition of the training set is such that 
no unique solution is possible. 

This is probably because you are training on structures
all with the same composition (e.g. all structures are
of the form n H2O). Consider explicitly setting the
per-element contributions if you know them, or
including a variety of structures of different
compositions in the training set.
"""
        )

    # calculate the per-element mean
    # use Ridge rather than LinearRegression to avoid singular matrices
    # when e.g. only one structure contains an atom of a given type...
    ridge = Ridge(fit_intercept=False, alpha=0.00001)
    ridge.fit(N.numpy(), per_structure_quantity)
    mu_Z = torch.tensor(ridge.coef_)
    means = {int(Z): float(mu) for Z, mu in zip(unique_Zs, mu_Z)}

    # calculate the per-element variance
    residuals = per_structure_quantity - N @ mu_Z
    # assuming that the squared residuals are a sum of the independent
    # variances for each atom, we can estimate these variances again
    # using Ridge regression
    ridge.fit(N.numpy(), residuals**2)
    var_Z = ridge.coef_
    # avoid negative variances by clipping to min value
    variances = {
        int(Z): max(float(var), min_variance)
        for Z, var in zip(unique_Zs, var_Z)
    }

    logger.debug(f"Per-element means: {means}")
    logger.debug(f"Per-element variances: {variances}")

    return means, variances


@torch.no_grad()
def add_auto_offset(
    model: GraphPESModel,
    structures: Sequence[AtomicGraph] | Sequence[ase.Atoms],
) -> GraphPESModel:
    from graph_pes.models import AdditionModel, LearnableOffset

    logger.info("""\
Attempting to automatically detect the offset energy for each element.
We do this by first generating predictions for each training structure 
(up to `config.fitting.max_n_pre_fit` if specified). 
This is a slow process! If you already know the reference energies (or the
difference in reference energies if you are fine-tuning an existing model to a
different level of theory), 
we recommend setting `config.fitting.auto_fit_reference_energies` to `False`
and manually specifying a `LearnableOffset` component of your model.

See the "Fine-tuning" tutorial in the docs for more information: 
https://jla-gardner.github.io/graph-pes/quickstart/fine-tuning.html""")

    graphs = [
        AtomicGraph.from_ase(s, model.cutoff.item())
        if isinstance(s, ase.Atoms)
        else s
        for s in structures
    ]

    model.eval()

    predictions = torch.tensor([model.predict_energy(g) for g in graphs])
    references = torch.tensor([g.properties["energy"] for g in graphs])
    diffs = references - predictions
    means, _ = guess_per_element_mean_and_var(diffs, to_batch(graphs))

    if all(abs(mu) < 1e-4 for mu in means.values()):
        logger.info(
            "When attempting to automatically fit the offset energy, "
            "no offset was detected, so skipping."
        )
        return model

    offset_param = PerElementParameter.from_dict(
        **{chemical_symbols[Z]: mu for Z, mu in means.items()},
        requires_grad=True,
    )
    offset_model = LearnableOffset()
    offset_model._offsets = offset_param
    offset_model._has_been_pre_fit.fill_(1)

    if isinstance(model, AdditionModel) and "auto_offset" not in model.models:
        model.models["auto_offset"] = offset_model
    else:
        model = AdditionModel(base=model, auto_offset=offset_model)

    model.train()

    return model
