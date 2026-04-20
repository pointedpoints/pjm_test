from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame


@dataclass(frozen=True)
class QuantileSurface:
    probabilities: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if probabilities.ndim != 1 or values.ndim != 1:
            raise ValueError("QuantileSurface expects one-dimensional probabilities and values.")
        if len(probabilities) != len(values):
            raise ValueError("QuantileSurface probabilities and values must have the same length.")
        if len(probabilities) < 2:
            raise ValueError("QuantileSurface requires at least two quantile knots.")
        if not np.all(np.diff(probabilities) > 0):
            raise ValueError("QuantileSurface probabilities must be strictly increasing.")
        if probabilities[0] <= 0.0 or probabilities[-1] >= 1.0:
            raise ValueError("QuantileSurface probabilities must lie strictly inside (0, 1).")
        if np.any(np.diff(values) < 0):
            raise ValueError("QuantileSurface values must be monotone non-decreasing.")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_quantiles(
        cls,
        probabilities: list[float] | np.ndarray,
        values: list[float] | np.ndarray,
    ) -> QuantileSurface:
        probs = np.asarray(probabilities, dtype=float)
        vals = np.asarray(values, dtype=float)
        order = np.argsort(probs)
        return cls(probabilities=probs[order], values=vals[order])

    def ppf(self, probability: float | np.ndarray) -> float | np.ndarray:
        probs = np.asarray(probability, dtype=float)
        clipped = np.clip(probs, 0.0, 1.0)
        extended_probs = np.concatenate(([0.0], self.probabilities, [1.0]))
        extended_values = np.concatenate(([self.values[0]], self.values, [self.values[-1]]))
        result = np.interp(clipped, extended_probs, extended_values)
        if np.ndim(probability) == 0:
            return float(result)
        return result

    def cdf(self, value: float | np.ndarray) -> float | np.ndarray:
        targets = np.asarray(value, dtype=float)
        x_knots, p_knots = self._cdf_knots()
        result = np.interp(targets, x_knots, p_knots, left=0.0, right=1.0)
        if np.ndim(value) == 0:
            return float(result)
        return result

    def interval(self, coverage: float) -> tuple[float, float]:
        coverage = float(coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1).")
        tail = (1.0 - coverage) / 2.0
        return float(self.ppf(tail)), float(self.ppf(1.0 - tail))

    def sample(
        self,
        size: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        uniforms = rng.uniform(0.0, 1.0, size=size)
        return np.asarray(self.ppf(uniforms), dtype=float)

    def pit(self, observation: float) -> float:
        return float(self.cdf(float(observation)))

    def crps(self, observation: float) -> float:
        grid = np.concatenate(([0.0], self.probabilities, [1.0]))
        quantiles = np.asarray(self.ppf(grid), dtype=float)
        losses = _pinball_loss(np.full_like(grid, float(observation)), quantiles, grid)
        return float(2.0 * np.trapezoid(losses, grid))

    def _cdf_knots(self) -> tuple[np.ndarray, np.ndarray]:
        unique_values: list[float] = []
        unique_probabilities: list[float] = []
        for value, probability in zip(self.values, self.probabilities, strict=True):
            scalar_value = float(value)
            scalar_probability = float(probability)
            if unique_values and np.isclose(unique_values[-1], scalar_value):
                unique_probabilities[-1] = scalar_probability
            else:
                unique_values.append(scalar_value)
                unique_probabilities.append(scalar_probability)
        return np.asarray(unique_values, dtype=float), np.asarray(unique_probabilities, dtype=float)


def quantile_surfaces_from_frame(predictions: pd.DataFrame) -> dict[pd.Timestamp, QuantileSurface]:
    if not is_quantile_prediction_frame(predictions):
        return {}

    quantile_predictions = predictions.copy()
    quantile_predictions["quantile"] = quantile_predictions["quantile"].astype(float)
    surfaces: dict[pd.Timestamp, QuantileSurface] = {}
    for ds, ds_frame in quantile_predictions.groupby("ds", sort=True):
        ordered = ds_frame.sort_values("quantile").copy()
        ordered["y_pred"] = np.maximum.accumulate(ordered["y_pred"].to_numpy(dtype=float))
        surfaces[pd.Timestamp(ds)] = QuantileSurface.from_quantiles(
            ordered["quantile"].to_numpy(dtype=float),
            ordered["y_pred"].to_numpy(dtype=float),
        )
    return surfaces


def pit_values_from_quantile_predictions(predictions: pd.DataFrame) -> np.ndarray:
    if not is_quantile_prediction_frame(predictions):
        return np.asarray([], dtype=float)

    surfaces = quantile_surfaces_from_frame(predictions)
    y_true = predictions.groupby("ds", sort=True)["y"].first()
    return np.asarray(
        [surfaces[ds].pit(y_true.loc[ds]) for ds in y_true.index],
        dtype=float,
    )


def mean_crps_from_quantile_predictions(predictions: pd.DataFrame) -> float:
    if not is_quantile_prediction_frame(predictions):
        return float("nan")

    surfaces = quantile_surfaces_from_frame(predictions)
    y_true = predictions.groupby("ds", sort=True)["y"].first()
    values = [surfaces[ds].crps(y_true.loc[ds]) for ds in y_true.index]
    return float(np.mean(values))


def summarize_pit(predictions: pd.DataFrame) -> dict[str, float]:
    pit_values = pit_values_from_quantile_predictions(predictions)
    if pit_values.size == 0:
        return {"pit_mean": float("nan"), "pit_variance": float("nan")}
    return {
        "pit_mean": float(np.mean(pit_values)),
        "pit_variance": float(np.var(pit_values)),
    }


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    errors = y_true - y_pred
    return np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
