from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame, point_prediction_view
from pjm_forecast.quantile_surface import mean_crps_from_quantile_predictions, summarize_pit


QUANTILE_INTERVALS: dict[str, tuple[float, float]] = {
    "80": (0.10, 0.90),
    "90": (0.05, 0.95),
    "98": (0.01, 0.99),
}


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, np.nan, denominator)
    return float(np.nanmean(200.0 * np.abs(y_true - y_pred) / denominator))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray) -> float:
    errors = y_true - y_pred
    losses = np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return float(np.mean(losses))


def compute_quantile_diagnostics(predictions: pd.DataFrame) -> dict[str, float | bool]:
    diagnostics: dict[str, float | bool] = {"has_quantiles": is_quantile_prediction_frame(predictions)}
    if not diagnostics["has_quantiles"]:
        diagnostics["crossing_rate"] = float("nan")
        diagnostics["pinball"] = float("nan")
        diagnostics["crps"] = float("nan")
        diagnostics["pit_mean"] = float("nan")
        diagnostics["pit_variance"] = float("nan")
        for label in QUANTILE_INTERVALS:
            diagnostics[f"coverage_{label}"] = float("nan")
            diagnostics[f"width_{label}"] = float("nan")
        return diagnostics

    quantile_predictions = predictions.copy()
    quantile_predictions["quantile"] = quantile_predictions["quantile"].astype(float)
    prediction_grid = quantile_predictions.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)
    y_true = quantile_predictions.groupby("ds", sort=True)["y"].first()

    if prediction_grid.shape[1] <= 1:
        diagnostics["crossing_rate"] = 0.0
    else:
        diffs = np.diff(prediction_grid.to_numpy(dtype=float), axis=1)
        diagnostics["crossing_rate"] = float((diffs < 0).any(axis=1).mean())
    diagnostics["pinball"] = pinball_loss(
        quantile_predictions["y"].to_numpy(dtype=float),
        quantile_predictions["y_pred"].to_numpy(dtype=float),
        quantile_predictions["quantile"].to_numpy(dtype=float),
    )
    diagnostics["crps"] = mean_crps_from_quantile_predictions(quantile_predictions)
    diagnostics.update(summarize_pit(quantile_predictions))

    for label, (lower, upper) in QUANTILE_INTERVALS.items():
        lower_column = _resolve_quantile_column(prediction_grid.columns, lower)
        upper_column = _resolve_quantile_column(prediction_grid.columns, upper)
        if lower_column is None or upper_column is None:
            diagnostics[f"coverage_{label}"] = float("nan")
            diagnostics[f"width_{label}"] = float("nan")
            continue
        lower_values = prediction_grid[lower_column].to_numpy(dtype=float)
        upper_values = prediction_grid[upper_column].to_numpy(dtype=float)
        y_values = y_true.reindex(prediction_grid.index).to_numpy(dtype=float)
        diagnostics[f"coverage_{label}"] = float(((y_values >= lower_values) & (y_values <= upper_values)).mean())
        diagnostics[f"width_{label}"] = float(np.mean(upper_values - lower_values))
    return diagnostics


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    point_view = point_prediction_view(predictions)
    y_true = point_view["y"].to_numpy(dtype=float)
    y_pred = point_view["y_pred"].to_numpy(dtype=float)
    metrics = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "smape": smape(y_true, y_pred)}
    if is_quantile_prediction_frame(predictions):
        metrics["pinball"] = pinball_loss(
            predictions["y"].to_numpy(dtype=float),
            predictions["y_pred"].to_numpy(dtype=float),
            predictions["quantile"].to_numpy(dtype=float),
        )
    return metrics


def compute_hourly_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    hourly = point_prediction_view(predictions)
    hourly["hour"] = pd.to_datetime(hourly["ds"]).dt.hour
    rows = []
    for hour, hour_df in hourly.groupby("hour", sort=True):
        rows.append({"hour": int(hour), "mae": mae(hour_df["y"].to_numpy(), hour_df["y_pred"].to_numpy())})
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def _resolve_quantile_column(columns: pd.Index, target: float) -> float | None:
    for column in columns.tolist():
        if np.isclose(float(column), target):
            return float(column)
    return None
