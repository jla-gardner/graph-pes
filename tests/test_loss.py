from __future__ import annotations

import torch
from graph_pes.loss import MAE, RMSE, Loss, WeightedLoss


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
    assert isinstance(weighted, WeightedLoss)
    assert set(weighted.losses) == {l1, l2}
    assert set(weighted.weights) == {10, 1}
