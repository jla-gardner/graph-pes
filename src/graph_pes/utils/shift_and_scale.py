from __future__ import annotations

import torch
from sklearn.linear_model import Ridge

from graph_pes.atomic_graph import (
    AtomicGraph,
    number_of_structures,
    sum_per_structure,
)

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
