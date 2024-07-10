from __future__ import annotations

import pytest
import torch
from graph_pes.training.loss import MAE, RMSE, Loss, TotalLoss


def test_metrics():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(MAE()(a, b), torch.tensor(0.0))
    assert torch.allclose(RMSE()(a, b), torch.tensor(0.0))

    c = torch.tensor([0, 0, 0]).float()
    assert torch.allclose(MAE()(a, c), torch.tensor(2.0))
    assert torch.allclose(RMSE()(a, c), torch.tensor((1 + 4 + 9) / 3).sqrt())


def test_loss_ops():
    l1 = Loss("energy")
    l2 = Loss("forces")

    weighted = 10 * l1 + l2 * 1
    assert isinstance(weighted, TotalLoss)
    assert set(weighted.losses) == {l1, l2}
    assert set(weighted.weights) == {10, 1}

    weighted = l1 / 2 + l2 / 3
    assert isinstance(weighted, TotalLoss)
    assert set(weighted.losses) == {l1, l2}
    assert set(weighted.weights) == {0.5, 1 / 3}

    weighted = l1 + l1 + l1
    assert isinstance(weighted, TotalLoss)
    assert len(weighted.losses) == 3


def test_warnings():
    l1 = Loss("energy")
    l2 = Loss("forces")

    with pytest.raises(TypeError, match="Cannot multiply Loss and"):
        l1 * l2  # type: ignore
    with pytest.raises(TypeError, match="Cannot divide Loss and"):
        l1 / l2  # type: ignore
    with pytest.raises(TypeError, match="Cannot multiply Loss and"):
        l1 * "hello"  # type: ignore
    with pytest.raises(TypeError, match="Cannot add Loss and"):
        l1 + "hello"  # type: ignore

    total = TotalLoss([l1, l2], [1, 1])
    with pytest.raises(TypeError, match="Cannot multiply TotalLoss and"):
        total * l1  # type: ignore
    with pytest.raises(TypeError, match="Cannot add TotalLoss and"):
        total + "hello"  # type: ignore
