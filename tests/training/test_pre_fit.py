from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import (
    number_of_structures,
    to_batch,
)
from graph_pes.models import LennardJonesMixture
from graph_pes.models.addition import AdditionModel
from graph_pes.utils.shift_and_scale import guess_per_element_mean_and_var


def _create_batch(
    mu: dict[int, float],
    sigma: dict[int, float],
    weights: list[float] | None = None,
) -> AtomicGraph:
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
    graphs: list[AtomicGraph] = []
    rng = np.random.default_rng(0)
    for _ in range(N):
        structure_size = rng.integers(4, 10)
        Zs = rng.choice(list(mu.keys()), size=structure_size, p=weights)
        total_E = 0
        for Z in Zs:
            total_E += rng.normal(mu[Z], sigma[Z])
        graphs.append(
            AtomicGraph.create_with_defaults(
                Z=torch.LongTensor(Zs),
                R=torch.randn(structure_size, 3),
                properties={"energy": torch.tensor(total_E)},
            )
        )
    return to_batch(graphs)


def test_guess_per_element_mean_and_var():
    mu = {1: -1.0, 2: -2.0}
    sigma = {1: 0.1, 2: 0.2}
    batch = _create_batch(mu=mu, sigma=sigma)

    # quickly check that this batch is as expected
    assert number_of_structures(batch) == 1_000
    assert sorted(torch.unique(batch.Z).tolist()) == [1, 2]

    # calculate the per-element mean and variance
    per_structure_quantity = batch.properties["energy"]
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
    means, variances = guess_per_element_mean_and_var(
        batch.properties["energy"], batch, min_variance=0.01
    )

    # ensure no variance is less than the value we choose to clamp to
    for value in variances.values():
        assert value >= 0.01


models = [
    LennardJonesMixture(),
    AdditionModel(a=LennardJonesMixture(), b=LennardJonesMixture()),
]
names = ["LennardJonesMixture", "AdditionModel"]


@pytest.mark.parametrize("model", models, ids=names)
def test(
    tmp_path: Path,
    model: GraphPESModel,
    caplog: pytest.LogCaptureFixture,
):
    assert model.elements_seen == []

    # show the model C and H
    methane = molecule("CH4")
    methane.info["energy"] = 1.0
    model.pre_fit_all_components([AtomicGraph.from_ase(methane, cutoff=3.0)])
    assert model.elements_seen == ["H", "C"]

    # check that these are persisted over save and load
    torch.save(model, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt")
    assert loaded.elements_seen == ["H", "C"]

    # show the model C, H, and O
    acetaldehyde = molecule("CH3CHO")
    acetaldehyde.info["energy"] = 2.0
    model.pre_fit_all_components(
        [AtomicGraph.from_ase(acetaldehyde, cutoff=3.0)]
    )
    assert any(
        record.levelname == "WARNING"
        and "has already been pre-fitted" in record.message
        for record in caplog.records
    )
    assert model.elements_seen == ["H", "C", "O"]
