from __future__ import annotations

import torch

from pjm_forecast.models.quantile_losses import WeightedHuberMQLoss


def test_weighted_huber_mqloss_accepts_per_quantile_deltas() -> None:
    loss = WeightedHuberMQLoss(
        quantiles=[0.1, 0.5, 0.9],
        delta=0.75,
        quantile_deltas=[1.25, 0.75, 1.25],
    )

    assert torch.allclose(loss.quantile_delta_tensor, torch.tensor([1.25, 0.75, 1.25]))


def test_weighted_huber_mqloss_penalizes_crossed_quantiles() -> None:
    base_loss = WeightedHuberMQLoss(
        quantiles=[0.1, 0.5, 0.9],
        delta=0.75,
        monotonicity_penalty=0.0,
    )
    constrained_loss = WeightedHuberMQLoss(
        quantiles=[0.1, 0.5, 0.9],
        delta=0.75,
        monotonicity_penalty=1.0,
    )
    y = torch.zeros((1, 2, 1))
    crossed_y_hat = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]])

    assert constrained_loss(y, crossed_y_hat) > base_loss(y, crossed_y_hat)
