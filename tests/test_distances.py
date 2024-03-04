from __future__ import annotations

import pytest
import torch
from graph_pes.models.distances import (
    Bessel,
    CosineEnvelope,
    DistanceExpansion,
    Envelope,
    ExponentialRBF,
    GaussianSmearing,
    PolynomialEnvelope,
    SinExpansion,
)


@pytest.mark.parametrize(
    "expansion",
    [
        Bessel,
        SinExpansion,
        GaussianSmearing,
        ExponentialRBF,
    ],
)
def test_expansions(expansion: type[DistanceExpansion]):
    n_features = 17
    cutoff = 5.0
    r = torch.linspace(0, cutoff, 10)
    x = expansion(n_features, cutoff)(r)
    assert x.shape == (10, n_features)
    assert x.grad_fn is not None

    x = expansion(n_features, cutoff, trainable=False)(r)
    assert x.grad_fn is None


@pytest.mark.parametrize(
    "envelope",
    [
        Envelope,
        CosineEnvelope,
        PolynomialEnvelope,
    ],
)
def test_envelopes(envelope: type[Envelope]):
    cutoff = 5.0
    env = envelope(cutoff=cutoff)

    r = torch.tensor([4.5, 5, 5.5])
    a, b, c = env(r).tolist()
    assert a > 0, "The envelope should be positive"
    assert b == 0, "The envelope should be zero at the cutoff"
    assert c == 0, "The envelope should be zero beyond the cutoff"
