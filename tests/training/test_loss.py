from __future__ import annotations

import pytest
import torch

from graph_pes.training.loss import (
    MAE,
    RMSE,
    Huber,
    ScaleFreeHuber,
    WeightedLoss,
)


def test_metrics():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(MAE()(a, b), torch.tensor(0.0))
    assert torch.allclose(RMSE()(a, b), torch.tensor(0.0))

    c = torch.tensor([0, 0, 0]).float()
    assert torch.allclose(MAE()(a, c), torch.tensor(2.0))
    assert torch.allclose(RMSE()(a, c), torch.tensor((1 + 4 + 9) / 3).sqrt())


def test_excpetion():
    with pytest.raises(ImportError):
        WeightedLoss()


def test_hubers():
    for delta in [0.1, 0.5, 1.0, 2.0]:
        a = torch.tensor([delta])
        b = torch.tensor([0.0])
        huber = Huber(delta)
        assert torch.allclose(huber(a, b), torch.tensor(delta**2 / 2))

        scaled_huber = ScaleFreeHuber(delta)
        assert torch.allclose(scaled_huber(a, b), torch.tensor(1.0))
