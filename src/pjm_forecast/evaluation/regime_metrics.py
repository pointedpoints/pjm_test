from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame, point_prediction_view

from .metrics import QUANTILE_INTERVALS, pinball_loss


REGIME_ORDER = ["all", "normal", "high", "spike", "extreme", "daily_max"]
TAIL_QUANTILES = [0.90, 0.95, 0.99]


def compute_regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    point_view = point_prediction_view(predictions).copy()
    point_view["ds"] = pd.to_datetime(point_view["ds"])
    point_view = point_view.sort_values("ds").reset_index(drop=True)
    point_view["regime"] = _price_regimes(point_view["y"].astype(float))

    if is_quantile_prediction_frame(predictions):
        quantile_grid = _quantile_grid(predictions)
    else:
        quantile_grid = pd.DataFrame(index=pd.DatetimeIndex(point_view["ds"]))

    rows = []
    for regime in REGIME_ORDER:
        if regime == "daily_max":
            mask = _daily_max_mask(point_view)
        elif regime == "all":
            mask = pd.Series(True, index=point_view.index)
        else:
            mask = point_view["regime"].eq(regime)

        subset = point_view.loc[mask].copy()
        rows.append(_metrics_for_subset(regime, subset, quantile_grid))
    return pd.DataFrame(rows)


def _price_regimes(y: pd.Series) -> pd.Series:
    if y.empty:
        return pd.Series(index=y.index, dtype=object)
    p80 = float(y.quantile(0.80))
    p90 = float(y.quantile(0.90))
    p95 = float(y.quantile(0.95))
    regimes = pd.Series("normal", index=y.index, dtype=object)
    regimes.loc[(y >= p80) & (y < p90)] = "high"
    regimes.loc[(y >= p90) & (y < p95)] = "spike"
    regimes.loc[y >= p95] = "extreme"
    return regimes


def _daily_max_mask(point_view: pd.DataFrame) -> pd.Series:
    if point_view.empty:
        return pd.Series(False, index=point_view.index)
    days = point_view["ds"].dt.floor("D")
    daily_idx = point_view.groupby(days, sort=True)["y"].idxmax()
    return point_view.index.to_series().isin(daily_idx)


def _quantile_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    quantile_predictions = predictions.copy()
    quantile_predictions["ds"] = pd.to_datetime(quantile_predictions["ds"])
    quantile_predictions["quantile"] = quantile_predictions["quantile"].astype(float)
    return quantile_predictions.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)


def _metrics_for_subset(regime: str, subset: pd.DataFrame, quantile_grid: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {"regime": regime, "n": int(len(subset))}
    metric_names = [
        "p50_mae",
        "p50_rmse",
        "p50_bias",
        "p50_underprediction_mean",
        "p50_underprediction_p95",
        "p90_pinball",
        "p95_pinball",
        "p99_pinball",
        "coverage_80",
        "coverage_90",
        "coverage_98",
        "width_80",
        "width_90",
        "width_98",
        "q99_exceedance_rate",
        "q99_excess_mean",
        "q99_excess_p95",
        "worst_q99_underprediction",
        "daily_max_q99_gap_mean",
        "daily_max_q99_gap_max",
    ]
    if subset.empty:
        row.update({name: float("nan") for name in metric_names})
        return row

    y = subset["y"].reset_index(drop=True).astype(float)
    p50 = _prediction_series(subset, quantile_grid, 0.50)
    residual = y - p50
    underprediction = residual.clip(lower=0.0)
    row["p50_mae"] = float(residual.abs().mean())
    row["p50_rmse"] = float(np.sqrt(np.mean(np.square(residual.to_numpy(dtype=float)))))
    row["p50_bias"] = float(residual.mean())
    row["p50_underprediction_mean"] = float(underprediction.mean())
    row["p50_underprediction_p95"] = float(np.quantile(underprediction.to_numpy(dtype=float), 0.95))

    for quantile in TAIL_QUANTILES:
        label = f"p{int(quantile * 100):02d}_pinball"
        q_pred = _prediction_series(subset, quantile_grid, quantile)
        if q_pred.isna().all():
            row[label] = float("nan")
            continue
        row[label] = pinball_loss(
            y.to_numpy(dtype=float),
            q_pred.to_numpy(dtype=float),
            np.full(len(subset), quantile, dtype=float),
        )

    for interval_label, (lower, upper) in QUANTILE_INTERVALS.items():
        lower_pred = _prediction_series(subset, quantile_grid, lower)
        upper_pred = _prediction_series(subset, quantile_grid, upper)
        if lower_pred.isna().all() or upper_pred.isna().all():
            row[f"coverage_{interval_label}"] = float("nan")
            row[f"width_{interval_label}"] = float("nan")
            continue
        row[f"coverage_{interval_label}"] = float(((y >= lower_pred) & (y <= upper_pred)).mean())
        row[f"width_{interval_label}"] = float((upper_pred - lower_pred).mean())

    q99 = _prediction_series(subset, quantile_grid, 0.99)
    if q99.isna().all():
        row["q99_exceedance_rate"] = float("nan")
        row["q99_excess_mean"] = float("nan")
        row["q99_excess_p95"] = float("nan")
        row["worst_q99_underprediction"] = float("nan")
        row["daily_max_q99_gap_mean"] = float("nan")
        row["daily_max_q99_gap_max"] = float("nan")
        return row

    q99_excess = (y - q99).clip(lower=0.0)
    daily_max_gap = y - q99
    row["q99_exceedance_rate"] = float((y > q99).mean())
    row["q99_excess_mean"] = float(q99_excess.mean())
    row["q99_excess_p95"] = float(np.quantile(q99_excess.to_numpy(dtype=float), 0.95))
    row["worst_q99_underprediction"] = float(q99_excess.max())
    row["daily_max_q99_gap_mean"] = float(daily_max_gap.mean())
    row["daily_max_q99_gap_max"] = float(daily_max_gap.max())
    return row


def _prediction_series(subset: pd.DataFrame, quantile_grid: pd.DataFrame, quantile: float) -> pd.Series:
    if not quantile_grid.empty:
        column = _resolve_quantile_column(quantile_grid.columns, quantile)
        if column is not None:
            return quantile_grid[column].reindex(pd.DatetimeIndex(subset["ds"])).reset_index(drop=True).astype(float)
    if np.isclose(quantile, 0.50):
        return subset["y_pred"].reset_index(drop=True).astype(float)
    return pd.Series(np.nan, index=range(len(subset)), dtype=float)


def _resolve_quantile_column(columns: pd.Index, target: float) -> float | None:
    for column in columns.tolist():
        if np.isclose(float(column), target):
            return float(column)
    return None
