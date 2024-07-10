from __future__ import annotations

from pathlib import Path

import pytest
from graph_pes.core import GraphPESModel
from graph_pes.deploy import deploy_model
from graph_pes.models import ALL_MODELS, NequIP


@pytest.mark.parametrize(
    "model_klass",
    ALL_MODELS,
    ids=[model.__name__ for model in ALL_MODELS],
)
def test_deploy(model_klass: type[GraphPESModel], tmp_path: Path):
    if model_klass is NequIP:
        model: GraphPESModel = NequIP(n_elements=2)  # type: ignore
    else:
        model = model_klass()

    save_path = tmp_path / f"{model.__class__.__name__}.pt"
    deploy_model(model, cutoff=4.0, path=save_path)
