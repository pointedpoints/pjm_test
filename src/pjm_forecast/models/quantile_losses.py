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


def _normalize_quantile_deltas(quantile_deltas: Sequence[float] | None, expected_size: int, fallback_delta: float):
    import torch

    if quantile_deltas is None:
        return torch.full((expected_size,), float(fallback_delta), dtype=torch.float32)

    values = [float(value) for value in quantile_deltas]
    if len(values) != expected_size:
        raise ValueError(
            f"quantile_deltas length must match quantiles length; received {len(values)} values for {expected_size} quantiles."
        )
    if any(value <= 0.0 for value in values):
        raise ValueError("quantile_deltas must be strictly positive.")
    return torch.tensor(values, dtype=torch.float32)


def _huber_loss_to_zero(values, deltas):
    import torch

    abs_values = torch.abs(values)
    return torch.where(abs_values <= deltas, 0.5 * values**2, deltas * (abs_values - 0.5 * deltas))


def _monotonicity_penalty(y_hat, quantile_count: int):
    import torch

    if y_hat.shape[-1] == quantile_count:
        differences = y_hat[..., :-1] - y_hat[..., 1:]
    elif y_hat.ndim >= 2 and y_hat.shape[-2] == quantile_count:
        differences = y_hat[..., :-1, :] - y_hat[..., 1:, :]
    else:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    return torch.mean(torch.relu(differences) ** 2)


class WeightedMQLoss:
    def __new__(
        cls,
        *,
        quantiles: Sequence[float],
        quantile_weights: Sequence[float] | None = None,
        monotonicity_penalty: float = 0.0,
    ):
        from neuralforecast.losses.pytorch import MQLoss  # type: ignore

        class _WeightedMQLossImpl(MQLoss):
            def __init__(self):
                super().__init__(quantiles=list(quantiles))
                self.monotonicity_penalty = float(monotonicity_penalty)
                self.register_buffer(
                    "quantile_weight_tensor",
                    _normalize_quantile_weights(quantile_weights, len(list(quantiles))),
                )

            def __call__(self, y, y_hat, y_insample=None, mask=None):
                import torch

                raw_y_hat = y_hat
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
                loss = torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1e-8)
                if self.monotonicity_penalty > 0.0:
                    loss = loss + self.monotonicity_penalty * _monotonicity_penalty(raw_y_hat, len(list(quantiles)))
                return loss

        return _WeightedMQLossImpl()


class WeightedHuberMQLoss:
    def __new__(
        cls,
        *,
        quantiles: Sequence[float],
        delta: float = 1.0,
        quantile_weights: Sequence[float] | None = None,
        quantile_deltas: Sequence[float] | None = None,
        monotonicity_penalty: float = 0.0,
    ):
        from neuralforecast.losses.pytorch import HuberMQLoss  # type: ignore

        class _WeightedHuberMQLossImpl(HuberMQLoss):
            def __init__(self):
                super().__init__(quantiles=list(quantiles), delta=float(delta))
                self.monotonicity_penalty = float(monotonicity_penalty)
                self.register_buffer(
                    "quantile_weight_tensor",
                    _normalize_quantile_weights(quantile_weights, len(list(quantiles))),
                )
                self.register_buffer(
                    "quantile_delta_tensor",
                    _normalize_quantile_deltas(quantile_deltas, len(list(quantiles)), float(delta)),
                )

            def __call__(self, y, y_hat, y_insample=None, mask=None):
                import torch

                raw_y_hat = y_hat
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
                quantile_delta_tensor = self.quantile_delta_tensor[None, None, None, :].to(y_hat.device)
                losses = _huber_loss_to_zero(quantiles_tensor * sq, quantile_delta_tensor) + _huber_loss_to_zero(
                    (1 - quantiles_tensor) * s1_q,
                    quantile_delta_tensor,
                )
                losses = (1 / self.outputsize_multiplier) * quantile_weights_tensor * losses
                weights = self._compute_weights(y=losses, mask=mask)
                loss = torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1e-8)
                if self.monotonicity_penalty > 0.0:
                    loss = loss + self.monotonicity_penalty * _monotonicity_penalty(raw_y_hat, len(list(quantiles)))
                return loss

        return _WeightedHuberMQLossImpl()
