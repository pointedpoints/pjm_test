from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, np.nan, denominator)
    return float(np.nanmean(200.0 * np.abs(y_true - y_pred) / denominator))


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["y"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "smape": smape(y_true, y_pred)}


def compute_hourly_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    hourly = predictions.copy()
    hourly["hour"] = pd.to_datetime(hourly["ds"]).dt.hour
    rows = []
    for hour, hour_df in hourly.groupby("hour", sort=True):
        rows.append({"hour": int(hour), "mae": mae(hour_df["y"].to_numpy(), hour_df["y_pred"].to_numpy())})
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)

