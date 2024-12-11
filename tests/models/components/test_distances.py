from __future__ import annotations

from pathlib import Path

import pytest
import torch

from graph_pes.models.components.distances import (
    Bessel,
    CosineEnvelope,
    DistanceExpansion,
    Envelope,
    ExponentialRBF,
    GaussianSmearing,
    PolynomialEnvelope,
    SinExpansion,
)

_expansions = [Bessel, GaussianSmearing, SinExpansion, ExponentialRBF]
_names = [expansion.__name__ for expansion in _expansions]
parameterise_expansions = pytest.mark.parametrize(
    "expansion_klass",
    _expansions,
    ids=_names,
)


@parameterise_expansions
def test_torchscript(
    expansion_klass: type[DistanceExpansion],
    tmp_path: Path,
):
    n_features = 17
    cutoff = 5.0
    expansion = expansion_klass(n_features, cutoff, trainable=True)

    scripted = torch.jit.script(expansion)
    assert isinstance(scripted, torch.jit.ScriptModule)
    r = torch.linspace(0, cutoff, 10)
    x = scripted(r)

    torch.jit.save(scripted, tmp_path / "expansion.pt")
    loaded: torch.jit.ScriptModule = torch.jit.load(tmp_path / "expansion.pt")

    assert torch.allclose(x, loaded(r))


@parameterise_expansions
def test_expansions(expansion_klass: type[DistanceExpansion]):
    n_features = 17
    cutoff = 5.0
    r = torch.linspace(0, cutoff, 10)
    x = expansion_klass(n_features, cutoff)(r)
    assert x.shape == (10, n_features)
    assert x.grad_fn is not None

    x = expansion_klass(n_features, cutoff, trainable=False)(r)
    assert x.grad_fn is None


@pytest.mark.parametrize(
    "envelope",
    [
        CosineEnvelope,
        PolynomialEnvelope,
    ],
)
def test_envelopes(envelope: type[Envelope]):
    cutoff = 5.0
    env = envelope(cutoff=cutoff)

    r = torch.tensor([4.5, 5, 5.5])
    x = env(r)
    assert x.shape == (3,)
    a, b, c = env(r).tolist()
    assert a > 0, "The envelope should be positive"
    assert b == 0, "The envelope should be zero at the cutoff"
    assert c == 0, "The envelope should be zero beyond the cutoff"
