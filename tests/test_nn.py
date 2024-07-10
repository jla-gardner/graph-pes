from __future__ import annotations

import pytest
import torch
from graph_pes.nn import (
    MLP,
    PerElementEmbedding,
    PerElementParameter,
    parse_activation,
)
from graph_pes.util import MAX_Z


def test_per_element_parameter(tmp_path):
    pep = PerElementParameter.of_length(5)
    assert pep._index_dims == 1
    assert pep.data.shape == (MAX_Z + 1, 5)
    assert isinstance(pep, PerElementParameter)
    assert isinstance(pep, torch.Tensor)
    assert isinstance(pep, torch.nn.Parameter)

    # no elements have been registered, so there should (appear to) be no
    # trainable parameters
    assert pep.numel() == 0

    # register the parameter for use with hydrogen
    pep.register_elements([1])
    assert pep.numel() == 5

    # test save and loading
    torch.save(pep, tmp_path / "pep.pt")
    pep_loaded = torch.load(tmp_path / "pep.pt")
    assert pep_loaded.numel() == 5
    assert pep.data.allclose(pep_loaded.data)
    assert pep.requires_grad == pep_loaded.requires_grad
    assert pep._accessed_Zs == pep_loaded._accessed_Zs
    assert pep._index_dims == pep_loaded._index_dims

    # test default value init
    assert PerElementParameter.of_length(1, default_value=1.0).data.allclose(
        torch.ones(MAX_Z + 1)
    )

    # test shape api
    pep = PerElementParameter.of_shape((5, 5), index_dims=2)
    assert pep.data.shape == (MAX_Z + 1, MAX_Z + 1, 5, 5)

    # test errors
    with pytest.raises(ValueError, match="Unknown element: ZZZ"):
        PerElementParameter.from_dict(ZZZ=1)


def test_per_element_embedding():
    embedding = PerElementEmbedding(10)
    embedding._embeddings.register_elements([1, 2, 3, 4, 5])
    Z = torch.tensor([1, 2, 3, 4, 5])
    assert embedding(Z).shape == (5, 10)
    assert embedding.parameters().__next__().numel() == 50


def test_mlp():
    mlp = MLP([10, 20, 1])

    # test behaviour
    assert mlp(torch.zeros(10)).shape == (1,)

    # test properties
    assert mlp.input_size == 10
    assert mlp.output_size == 1

    # test internals
    assert len(mlp.linear_layers) == 2

    # test nice repr
    assert "MLP(10 → 20 → 1" in str(mlp)


def test_activations():
    act = parse_activation("ReLU")
    assert act(torch.tensor([-1.0])).item() == 0.0

    with pytest.raises(
        ValueError, match="Activation function ZZZ not found in `torch.nn`."
    ):
        parse_activation("ZZZ")
