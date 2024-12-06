from __future__ import annotations

import numpy as np
import pytest
import torch
from graph_pes.utils.misc import (
    as_possible_tensor,
    build_single_nested_dict,
    differentiate,
    full_3x3_to_voigt_6,
    nested_merge,
    nested_merge_all,
    random_split,
    voigt_6_to_full_3x3,
)

possible_tensors = [
    (1, True),
    (1.0, True),
    ([1, 2, 3], True),
    (torch.tensor([1, 2, 3]), True),
    (np.array([1, 2, 3]), True),
    (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), True),
    ("hello", False),
]


@pytest.mark.parametrize("obj, can_be_converted", possible_tensors)
def test_as_possible_tensor(obj, can_be_converted):
    if can_be_converted:
        assert isinstance(as_possible_tensor(obj), torch.Tensor)
    else:
        assert as_possible_tensor(obj) is None


def test_differentiate():
    # test that it works
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()
    dy_dx = differentiate(y, x)
    assert torch.allclose(dy_dx, torch.ones_like(x))

    # test that it works with a non-scalar y
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x**2
    dy_dx = differentiate(y, x)
    assert torch.allclose(dy_dx, 2 * x)

    # test that it works if x is not part of the computation graph
    x = torch.tensor([1.0, 2.0, 3.0])
    z = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = z.sum()
    dy_dx = differentiate(y, x)
    assert torch.allclose(dy_dx, torch.zeros_like(x))

    # finally, we want to test that the gradient itself has a gradient
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    dy_dx = differentiate(y, x, keep_graph=True)
    dy_dx2 = differentiate(dy_dx, x)
    assert torch.allclose(dy_dx2, 2 * torch.ones_like(x))


def test_nested_merge():
    a = {"a": 1, "b": {"c": 2}, "d": 3}
    b = {"a": 3, "b": {"c": 4}}
    c = nested_merge(a, b)
    assert c == {"a": 3, "b": {"c": 4}, "d": 3}, "nested_merge failed"
    assert a == {"a": 1, "b": {"c": 2}, "d": 3}, "nested_merge mutated a"
    assert b == {"a": 3, "b": {"c": 4}}, "nested_merge mutated b"


def test_build_single_nested_dict():
    assert build_single_nested_dict(["a", "b", "c"], 4) == {
        "a": {"b": {"c": 4}}
    }


def test_nested_merge_all():
    assert nested_merge_all({"a": 1}, {"a": 2, "b": 1}, {"a": 3}) == {
        "a": 3,
        "b": 1,
    }

    assert nested_merge_all(
        {"a": {"b": {"c": 1}}},
        {"a": {"b": {"d": 2}}},
        {"a": {"b": {"c": 2}}},
    ) == {"a": {"b": {"c": 2, "d": 2}}}


def test_random_split():
    indices = list(range(10))
    split = random_split(indices, lengths=[2, 2], seed=0)
    assert split == [[2, 8], [4, 9]]

    with pytest.raises(ValueError, match="Not enough things to split"):
        random_split(indices, lengths=[20, 20], seed=0)


def test_stress_conversions():
    # non-batched
    stress = torch.rand(3, 3)
    # symmetrize:
    stress = (stress + stress.T) / 2
    voigt = full_3x3_to_voigt_6(stress)
    assert voigt.shape == (6,)
    stress_again = voigt_6_to_full_3x3(voigt)
    torch.testing.assert_close(stress_again, stress)

    # batched
    stress = torch.rand(2, 3, 3)
    # symmetrize:
    stress = (stress + stress.transpose(1, 2)) / 2
    voigt = full_3x3_to_voigt_6(stress)
    assert voigt.shape == (2, 6)
    stress_again = voigt_6_to_full_3x3(voigt)
    torch.testing.assert_close(stress_again, stress)
