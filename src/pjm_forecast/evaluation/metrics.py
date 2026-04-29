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

UPPER_TAIL_GAPS: dict[str, tuple[float, float]] = {
    "q95_q99": (0.95, 0.99),
    "q99_q995": (0.99, 0.995),
}

UPPER_TAIL_MISS_METRICS = [
    "q99_exceedance_rate",
    "q99_excess_mean",
    "q99_excess_p95",
    "max_y_q99_gap",
    "worst_q99_underprediction",
    "daily_max_q99_gap_mean",
    "daily_max_q99_gap_max",
]

MEDIAN_DIAGNOSTIC_METRICS = [
    "q50_mae",
    "q50_bias_mean",
    "q50_bias_median",
]


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
        for label in UPPER_TAIL_GAPS:
            diagnostics[f"{label}_gap_mean"] = float("nan")
            diagnostics[f"{label}_slope_mean"] = float("nan")
        for metric_name in UPPER_TAIL_MISS_METRICS:
            diagnostics[metric_name] = float("nan")
        for metric_name in MEDIAN_DIAGNOSTIC_METRICS:
            diagnostics[metric_name] = float("nan")
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
    diagnostics.update(_compute_median_diagnostics(prediction_grid, y_true))

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
    diagnostics.update(_compute_upper_tail_diagnostics(prediction_grid, y_true))
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


def _compute_upper_tail_diagnostics(prediction_grid: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    for label, (lower, upper) in UPPER_TAIL_GAPS.items():
        lower_column = _resolve_quantile_column(prediction_grid.columns, lower)
        upper_column = _resolve_quantile_column(prediction_grid.columns, upper)
        if lower_column is None or upper_column is None:
            diagnostics[f"{label}_gap_mean"] = float("nan")
            diagnostics[f"{label}_slope_mean"] = float("nan")
            continue

        gap = prediction_grid[upper_column].to_numpy(dtype=float) - prediction_grid[lower_column].to_numpy(dtype=float)
        diagnostics[f"{label}_gap_mean"] = float(np.mean(gap))
        diagnostics[f"{label}_slope_mean"] = float(np.mean(gap / (upper - lower)))

    q99_column = _resolve_quantile_column(prediction_grid.columns, 0.99)
    if q99_column is None:
        for metric_name in UPPER_TAIL_MISS_METRICS:
            diagnostics[metric_name] = float("nan")
        return diagnostics

    aligned_y = y_true.reindex(prediction_grid.index).astype(float)
    q99 = prediction_grid[q99_column].astype(float)
    excess = (aligned_y - q99).clip(lower=0.0)
    diagnostics["q99_exceedance_rate"] = float((aligned_y > q99).mean())
    diagnostics["q99_excess_mean"] = float(excess.mean())
    diagnostics["q99_excess_p95"] = float(np.quantile(excess.to_numpy(dtype=float), 0.95))

    max_y_timestamp = aligned_y.idxmax()
    diagnostics["max_y_q99_gap"] = float(aligned_y.loc[max_y_timestamp] - q99.loc[max_y_timestamp])
    diagnostics["worst_q99_underprediction"] = float(excess.max())

    daily = pd.DataFrame({"y": aligned_y, "q99": q99})
    daily["day"] = pd.DatetimeIndex(daily.index).floor("D")
    daily_maxima = daily.groupby("day", sort=True)[["y", "q99"]].max()
    daily_max_gap = daily_maxima["y"] - daily_maxima["q99"]
    diagnostics["daily_max_q99_gap_mean"] = float(daily_max_gap.mean())
    diagnostics["daily_max_q99_gap_max"] = float(daily_max_gap.max())
    return diagnostics


def _compute_median_diagnostics(prediction_grid: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    q50_column = _resolve_quantile_column(prediction_grid.columns, 0.5)
    if q50_column is None:
        return {metric_name: float("nan") for metric_name in MEDIAN_DIAGNOSTIC_METRICS}

    aligned_y = y_true.reindex(prediction_grid.index).astype(float)
    q50 = prediction_grid[q50_column].astype(float)
    residual = aligned_y - q50
    diagnostics["q50_mae"] = float(residual.abs().mean())
    diagnostics["q50_bias_mean"] = float(residual.mean())
    diagnostics["q50_bias_median"] = float(residual.median())
    return diagnostics
