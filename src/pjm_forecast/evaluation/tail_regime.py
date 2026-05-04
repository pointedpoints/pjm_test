from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame


def compute_tail_regime_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    grid, y_true = _quantile_grid(predictions)
    if grid.empty:
        return pd.DataFrame(columns=_tail_columns())

    q99 = _resolve_quantile(grid, 0.99)
    q995 = _resolve_quantile(grid, 0.995)
    if q99 is None or q995 is None:
        return pd.DataFrame(columns=_tail_columns())

    thresholds = {
        "p50": float(y_true.quantile(0.50)),
        "p80": float(y_true.quantile(0.80)),
        "p90": float(y_true.quantile(0.90)),
        "p95": float(y_true.quantile(0.95)),
        "p99": float(y_true.quantile(0.99)),
    }
    masks = {
        "all": pd.Series(True, index=y_true.index),
        "actual_le_p50": y_true <= thresholds["p50"],
        "actual_p50_p80": (y_true > thresholds["p50"]) & (y_true <= thresholds["p80"]),
        "actual_p80_p90": (y_true > thresholds["p80"]) & (y_true <= thresholds["p90"]),
        "actual_p90_p95": (y_true > thresholds["p90"]) & (y_true <= thresholds["p95"]),
        "actual_p95_p99": (y_true > thresholds["p95"]) & (y_true <= thresholds["p99"]),
        "actual_gt_p99": y_true > thresholds["p99"],
    }
    rows = [_summarize_tail(label, mask, y_true, grid[q99], grid[q995]) for label, mask in masks.items()]
    return pd.DataFrame(rows, columns=_tail_columns())


def compute_daily_peak_tail_gap(predictions: pd.DataFrame) -> pd.DataFrame:
    grid, y_true = _quantile_grid(predictions)
    if grid.empty:
        return pd.DataFrame(columns=_daily_columns())
    q99 = _resolve_quantile(grid, 0.99)
    q995 = _resolve_quantile(grid, 0.995)
    if q99 is None or q995 is None:
        return pd.DataFrame(columns=_daily_columns())

    frame = pd.DataFrame({"y": y_true, "q99": grid[q99], "q995": grid[q995]}, index=grid.index)
    frame["day"] = pd.DatetimeIndex(frame.index).floor("D")
    rows: list[dict[str, object]] = []
    for day, day_frame in frame.groupby("day", sort=True):
        peak_timestamp = day_frame["y"].idxmax()
        peak = day_frame.loc[peak_timestamp]
        rows.append(
            {
                "day": pd.Timestamp(day).date().isoformat(),
                "actual_max": float(peak["y"]),
                "actual_peak_hour": int(pd.Timestamp(peak_timestamp).hour),
                "peak_q99": float(peak["q99"]),
                "peak_q995": float(peak["q995"]),
                "peak_q99_gap": float(peak["y"] - peak["q99"]),
                "peak_q995_gap": float(peak["y"] - peak["q995"]),
                "daily_q99_upper_coverage": float((day_frame["y"] <= day_frame["q99"]).mean()),
                "daily_q995_upper_coverage": float((day_frame["y"] <= day_frame["q995"]).mean()),
            }
        )
    return pd.DataFrame(rows, columns=_daily_columns())


def _quantile_grid(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not is_quantile_prediction_frame(predictions):
        return pd.DataFrame(), pd.Series(dtype=float)
    frame = predictions.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["quantile"] = frame["quantile"].astype(float)
    grid = frame.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)
    y_true = frame.groupby("ds", sort=True)["y"].first().reindex(grid.index).astype(float)
    return grid, y_true


def _resolve_quantile(grid: pd.DataFrame, target: float) -> float | None:
    for column in grid.columns:
        if np.isclose(float(column), target):
            return float(column)
    return None


def _summarize_tail(label: str, mask: pd.Series, y: pd.Series, q99: pd.Series, q995: pd.Series) -> dict[str, object]:
    subset_y = y.loc[mask]
    subset_q99 = q99.loc[mask]
    subset_q995 = q995.loc[mask]
    if subset_y.empty:
        return {
            "regime": label,
            "n_hours": 0,
            "actual_mean": np.nan,
            "actual_max": np.nan,
            "q99_upper_coverage": np.nan,
            "q995_upper_coverage": np.nan,
            "q99_excess_mean": np.nan,
            "q99_excess_p95": np.nan,
            "q99_excess_max": np.nan,
        }
    q99_excess = (subset_y - subset_q99).clip(lower=0.0)
    return {
        "regime": label,
        "n_hours": int(len(subset_y)),
        "actual_mean": float(subset_y.mean()),
        "actual_max": float(subset_y.max()),
        "q99_upper_coverage": float((subset_y <= subset_q99).mean()),
        "q995_upper_coverage": float((subset_y <= subset_q995).mean()),
        "q99_excess_mean": float(q99_excess.mean()),
        "q99_excess_p95": float(q99_excess.quantile(0.95)),
        "q99_excess_max": float(q99_excess.max()),
    }


def _tail_columns() -> list[str]:
    return [
        "regime",
        "n_hours",
        "actual_mean",
        "actual_max",
        "q99_upper_coverage",
        "q995_upper_coverage",
        "q99_excess_mean",
        "q99_excess_p95",
        "q99_excess_max",
    ]


def _daily_columns() -> list[str]:
    return [
        "day",
        "actual_max",
        "actual_peak_hour",
        "peak_q99",
        "peak_q995",
        "peak_q99_gap",
        "peak_q995_gap",
        "daily_q99_upper_coverage",
        "daily_q995_upper_coverage",
    ]
