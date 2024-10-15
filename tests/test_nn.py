from __future__ import annotations

import pytest
import torch
from graph_pes.nn import (
    MLP,
    AtomicOneHot,
    PerElementEmbedding,
    PerElementParameter,
    UniformModuleDict,
    UniformModuleList,
    parse_activation,
)
from graph_pes.util import MAX_Z, left_aligned_div, left_aligned_mul


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


def test_one_hot():
    one_hot = AtomicOneHot(["H", "C", "O"])

    Z = torch.tensor([1, 6, 8])
    Z_emb = one_hot(Z)

    assert Z_emb.shape == (3, 3)
    assert Z_emb.allclose(torch.eye(3))

    with pytest.raises(ValueError, match="Unknown element"):
        one_hot(torch.tensor([2]))


def test_module_dict():
    umd = UniformModuleDict(
        a=torch.nn.Linear(10, 10),
        b=torch.nn.Linear(10, 10),
    )

    assert len(umd) == 2

    for k, v in umd.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.nn.Linear)

    assert "a" in umd

    b = umd.pop("b")
    assert b is not None
    assert len(umd) == 1


def test_module_list():
    uml = UniformModuleList(
        [
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        ]
    )

    assert len(uml) == 2
    assert isinstance(uml[0], torch.nn.Linear)

    uml[1] = torch.nn.Linear(100, 100)
    assert isinstance(uml[1], torch.nn.Linear)
    assert uml[1].in_features == 100

    uml.append(torch.nn.Linear(1000, 1000))
    assert len(uml) == 3

    lin = uml.pop(-1)
    assert len(uml) == 2
    assert lin.in_features == 1000

    uml.insert(1, torch.nn.Linear(10000, 10000))
    assert len(uml) == 3
    assert uml[1].in_features == 10000


@pytest.mark.parametrize("x_dim", [0, 1, 2, 3])
def test_left_aligned_ops(x_dim: int):
    N = 10

    y = torch.ones((N,)) * 5

    if x_dim == 0:
        x = torch.randn(N)
    else:
        x = torch.randn(N, *[i + 1 for i in range(x_dim)])

    z = left_aligned_mul(x, y)
    assert z.shape == x.shape
    assert z.allclose(x * 5)

    z = left_aligned_div(x, y)
    assert z.shape == x.shape
    assert z.allclose(x / 5)
