from __future__ import annotations

from typing import Sequence


def _normalize_quantile_weights(quantile_weights: Sequence[float] | None, expected_size: int):
    import torch

    if quantile_weights is None:
        return torch.ones(expected_size, dtype=torch.float32)

    values = [float(value) for value in quantile_weights]
    if len(values) != expected_size:
        raise ValueError(
            f"quantile_weights length must match quantiles length; received {len(values)} values for {expected_size} quantiles."
        )
    if any(value <= 0.0 for value in values):
        raise ValueError("quantile_weights must be strictly positive.")

    weights = torch.tensor(values, dtype=torch.float32)
    return weights * (len(values) / torch.sum(weights))


class WeightedMQLoss:
    def __new__(cls, *, quantiles: Sequence[float], quantile_weights: Sequence[float] | None = None):
        from neuralforecast.losses.pytorch import MQLoss  # type: ignore

        class _WeightedMQLossImpl(MQLoss):
            def __init__(self):
                super().__init__(quantiles=list(quantiles))
                self.register_buffer(
                    "quantile_weight_tensor",
                    _normalize_quantile_weights(quantile_weights, len(list(quantiles))),
                )

            def __call__(self, y, y_hat, y_insample=None, mask=None):
                import torch

                if y_hat.ndim == 3:
                    y_hat = y_hat.unsqueeze(-1)

                y = y.unsqueeze(-1)
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                else:
                    mask = torch.ones_like(y, device=y.device)

                error = y_hat - y
                sq = torch.maximum(-error, torch.zeros_like(error))
                s1_q = torch.maximum(error, torch.zeros_like(error))

                quantiles_tensor = self.quantiles[None, None, None, :].to(y_hat.device)
                quantile_weights_tensor = self.quantile_weight_tensor[None, None, None, :].to(y_hat.device)
                losses = (1 / self.outputsize_multiplier) * quantile_weights_tensor * (
                    quantiles_tensor * sq + (1 - quantiles_tensor) * s1_q
                )
                weights = self._compute_weights(y=losses, mask=mask)
                return torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1e-8)

        return _WeightedMQLossImpl()

class WeightedHuberMQLoss:
    def __new__(
        cls,
        *,
        quantiles: Sequence[float],
        delta: float = 1.0,
        quantile_weights: Sequence[float] | None = None,
    ):
        from neuralforecast.losses.pytorch import HuberMQLoss  # type: ignore

        class _WeightedHuberMQLossImpl(HuberMQLoss):
            def __init__(self):
                super().__init__(quantiles=list(quantiles), delta=float(delta))
                self.register_buffer(
                    "quantile_weight_tensor",
                    _normalize_quantile_weights(quantile_weights, len(list(quantiles))),
                )

            def __call__(self, y, y_hat, y_insample=None, mask=None):
                import torch
                import torch.nn.functional as F

                if y_hat.ndim == 3:
                    y_hat = y_hat.unsqueeze(-1)

                y = y.unsqueeze(-1)
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                else:
                    mask = torch.ones_like(y, device=y.device)

                error = y_hat - y
                zero_error = torch.zeros_like(error)
                sq = torch.maximum(-error, torch.zeros_like(error))
                s1_q = torch.maximum(error, torch.zeros_like(error))

                quantiles_tensor = self.quantiles[None, None, None, :].to(y_hat.device)
                quantile_weights_tensor = self.quantile_weight_tensor[None, None, None, :].to(y_hat.device)
                losses = F.huber_loss(quantiles_tensor * sq, zero_error, reduction="none", delta=self.delta) + F.huber_loss(
                    (1 - quantiles_tensor) * s1_q,
                    zero_error,
                    reduction="none",
                    delta=self.delta,
                )
                losses = (1 / self.outputsize_multiplier) * quantile_weights_tensor * losses
                weights = self._compute_weights(y=losses, mask=mask)
                return torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1e-8)

        return _WeightedHuberMQLossImpl()
