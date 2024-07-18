from __future__ import annotations

from pathlib import Path

import pytest
import torch
from graph_pes.models.distances import (
    Bessel,
    DistanceExpansion,
    ExponentialRBF,
    GaussianSmearing,
    SinExpansion,
)

_expansions = [Bessel, GaussianSmearing, SinExpansion, ExponentialRBF]


@pytest.mark.parametrize(
    "expansion_klass",
    _expansions,
    ids=[expansion.__name__ for expansion in _expansions],
)
def test_distance_expansions(
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
