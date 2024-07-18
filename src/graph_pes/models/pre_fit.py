from __future__ import annotations

import torch
from sklearn.linear_model import Ridge

from graph_pes.graphs.graph_typing import AtomicGraphBatch
from graph_pes.graphs.operations import number_of_structures, sum_per_structure
from graph_pes.logger import logger

MIN_VARIANCE = 0.01


def guess_per_element_mean_and_var(
    per_structure_quantity: torch.Tensor,
    batch: AtomicGraphBatch,
) -> tuple[dict[int, float], dict[int, float]]:
    r"""
    Guess the per-element mean (:math:`\mu_Z`) and variance (:math:`\sigma_Z^2`)
    of a per-structure quantity under the following assumptions:

    1. the per-structure property, :math:`P`, is a summation over local
       properties of its components atoms: :math:`P = \sum_{i=1}^{N} p_{Z_i}`.
    2. the per-atom properties, :math:`p_{Z_i}`, are independent and identically
       distributed (i.i.d.) for each atom of type :math:`Z_i` according to a
       normal distribution:
       :math:`p_{Z} \sim \mathcal{N}(\mu_{Z}, \sigma_{Z}^2)`.
    """

    # extract the atomic numbers to tensor N such that:
    # N[structure, Z] is the number of atoms of atomic number Z in structure
    unique_Zs = torch.unique(batch["atomic_numbers"])  # (n_Z,)
    N = torch.zeros(number_of_structures(batch), len(unique_Zs))  # (batch, n_Z)
    for i, Z in enumerate(unique_Zs):
        N[:, i] = sum_per_structure(
            (batch["atomic_numbers"] == Z).float(), batch
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
        int(Z): max(float(var), MIN_VARIANCE)
        for Z, var in zip(unique_Zs, var_Z)
    }

    logger.debug(f"Per-element means: {means}")
    logger.debug(f"Per-element variances: {variances}")

    return means, variances
