from __future__ import annotations

import numpy as np
import torch
from graph_pes.graphs.graph_typing import LabelledBatch, LabelledGraph
from graph_pes.graphs.operations import number_of_structures, to_batch
from graph_pes.models.pre_fit import (
    MIN_VARIANCE,
    guess_per_element_mean_and_var,
)


def _create_batch(
    mu: dict[int, float],
    sigma: dict[int, float],
    weights: list[float] | None = None,
) -> LabelledBatch:
    """
    Create a batch of structures with local energies distributed
    according to the given parameters.

    Parameters
    ----------
    mu
        The per-element mean local energies.
    sigma
        The per-element standard deviations in local energy.
    weights
        The relative likelihood of sampling each element.
    """

    N = 1_000
    graphs: list[LabelledGraph] = []
    rng = np.random.default_rng(0)
    for _ in range(N):
        structure_size = rng.integers(4, 10)
        Zs = rng.choice(list(mu.keys()), size=structure_size, p=weights)
        total_E = 0
        for Z in Zs:
            total_E += rng.normal(mu[Z], sigma[Z])
        graphs.append(
            dict(
                atomic_numbers=torch.LongTensor(Zs),
                _positions=torch.randn(structure_size, 3),
                energy=torch.tensor(total_E),
                _neighbour_cell_offsets=torch.zeros((2, 0)),
                neighbour_index=torch.zeros((2, 0), dtype=torch.long),
                cell=torch.eye(3),
            )  # type: ignore
        )
    return to_batch(graphs)


def test_guess_per_element_mean_and_var():
    mu = {1: -1.0, 2: -2.0}
    sigma = {1: 0.1, 2: 0.2}
    batch = _create_batch(mu=mu, sigma=sigma)

    # quickly check that this batch is as expected
    assert number_of_structures(batch) == 1_000
    assert sorted(torch.unique(batch["atomic_numbers"]).tolist()) == [1, 2]

    # calculate the per-element mean and variance
    per_structure_quantity = batch["energy"]
    means, variances = guess_per_element_mean_and_var(
        per_structure_quantity, batch
    )

    # are means roughly right?
    for Z, actual_mu in mu.items():
        assert np.isclose(means[Z], actual_mu, atol=0.01)

    # are variances roughly right?
    for Z, actual_sigma in sigma.items():
        assert np.isclose(variances[Z], actual_sigma**2, atol=0.01)


def test_clamping():
    # variances can not be negative: ensure that they are clamped
    mu = {1: -1.0, 2: -2.0}
    sigma = {1: 0.0, 2: 1.0}
    batch = _create_batch(mu=mu, sigma=sigma)

    # calculate the per-element mean and variance
    means, variances = guess_per_element_mean_and_var(batch["energy"], batch)

    # ensure no variance is less than the value we choose to clamp to
    for value in variances.values():
        assert value >= MIN_VARIANCE
