from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame, point_prediction_view


PRICE_BINS: list[tuple[str, float, float]] = [
    ("<=10", float("-inf"), 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100-200", 100.0, 200.0),
    (">200", 200.0, float("inf")),
]


def compute_relative_error_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    point = _q50_point_view(predictions)
    if point.empty:
        return pd.DataFrame(columns=_columns())

    point["ds"] = pd.to_datetime(point["ds"])
    point["abs_error"] = (point["y"].astype(float) - point["y_pred"].astype(float)).abs()
    denominator = point["y"].astype(float).abs().replace(0.0, np.nan)
    point["ape"] = point["abs_error"] / denominator
    smape_denominator = point["y"].astype(float).abs() + point["y_pred"].astype(float).abs()
    point["smape"] = np.where(smape_denominator == 0.0, np.nan, 2.0 * point["abs_error"] / smape_denominator)
    point["month"] = point["ds"].dt.to_period("M").astype(str)
    point["hour"] = point["ds"].dt.hour.astype(str)

    rows: list[dict[str, object]] = [_summarize("all", "all", point)]
    for label, lower, upper in PRICE_BINS:
        subset = point.loc[_price_bin_mask(point["y"].astype(float), lower, upper)]
        rows.append(_summarize("actual_price_bin", label, subset))

    for month, subset in point.groupby("month", sort=True):
        rows.append(_summarize("month", str(month), subset))
    for hour, subset in point.groupby("hour", sort=True):
        rows.append(_summarize("hour", str(hour), subset))

    return pd.DataFrame(rows, columns=_columns())


def _price_bin_mask(values: pd.Series, lower: float, upper: float) -> pd.Series:
    if np.isneginf(lower):
        return values <= upper
    if np.isposinf(upper):
        return values > lower
    if np.isclose(upper, 200.0):
        return (values >= lower) & (values <= upper)
    return (values >= lower) & (values < upper)


def _q50_point_view(predictions: pd.DataFrame) -> pd.DataFrame:
    if not is_quantile_prediction_frame(predictions):
        return point_prediction_view(predictions).loc[:, ["ds", "y", "y_pred"]].copy()
    frame = predictions.copy()
    frame["quantile"] = frame["quantile"].astype(float)
    q50 = frame.loc[np.isclose(frame["quantile"], 0.50), ["ds", "y", "y_pred"]].copy()
    return q50.sort_values("ds").reset_index(drop=True)


def _summarize(slice_type: str, label: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "slice_type": slice_type,
            "slice": label,
            "n_hours": 0,
            "actual_mean": np.nan,
            "actual_median": np.nan,
            "q50_mae": np.nan,
            "q50_bias_mean": np.nan,
            "q50_bias_median": np.nan,
            "wape": np.nan,
            "smape": np.nan,
            "median_ape": np.nan,
            "p75_ape": np.nan,
            "p90_ape": np.nan,
            "share_ape_le_10pct": np.nan,
            "share_ape_le_25pct": np.nan,
            "share_ape_le_50pct": np.nan,
        }
    y = frame["y"].astype(float)
    y_pred = frame["y_pred"].astype(float)
    error = y - y_pred
    abs_error = frame["abs_error"].astype(float)
    ape = frame["ape"].astype(float)
    return {
        "slice_type": slice_type,
        "slice": label,
        "n_hours": int(len(frame)),
        "actual_mean": float(y.mean()),
        "actual_median": float(y.median()),
        "q50_mae": float(abs_error.mean()),
        "q50_bias_mean": float(error.mean()),
        "q50_bias_median": float(error.median()),
        "wape": float(abs_error.sum() / y.abs().sum()) if float(y.abs().sum()) > 0.0 else np.nan,
        "smape": float(frame["smape"].mean()),
        "median_ape": float(ape.median()),
        "p75_ape": float(ape.quantile(0.75)),
        "p90_ape": float(ape.quantile(0.90)),
        "share_ape_le_10pct": float((ape <= 0.10).mean()),
        "share_ape_le_25pct": float((ape <= 0.25).mean()),
        "share_ape_le_50pct": float((ape <= 0.50).mean()),
    }


def _columns() -> list[str]:
    return [
        "slice_type",
        "slice",
        "n_hours",
        "actual_mean",
        "actual_median",
        "q50_mae",
        "q50_bias_mean",
        "q50_bias_median",
        "wape",
        "smape",
        "median_ape",
        "p75_ape",
        "p90_ape",
        "share_ape_le_10pct",
        "share_ape_le_25pct",
        "share_ape_le_50pct",
    ]
