from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from pjm_forecast.config import ProjectConfig


def _window_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, window_days: int) -> pd.DataFrame:
    history_end = forecast_day - pd.Timedelta(hours=1)
    window_start = forecast_day - pd.Timedelta(days=window_days)
    mask = (feature_df["ds"] >= window_start) & (feature_df["ds"] <= history_end)
    return feature_df.loc[mask].copy()


def _future_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future_end = forecast_day + pd.Timedelta(hours=horizon - 1)
    mask = (feature_df["ds"] >= forecast_day) & (feature_df["ds"] <= future_end)
    future_df = feature_df.loc[mask].copy()
    if len(future_df) != horizon:
        raise ValueError(f"Expected {horizon} rows for future horizon, got {len(future_df)} on {forecast_day}.")
    return future_df


def _retrain_due(forecast_day: pd.Timestamp, retrain_weekday: int, existing_model: object | None) -> bool:
    return existing_model is None or forecast_day.weekday() == retrain_weekday


def run_rolling_backtest(
    config: ProjectConfig,
    feature_df: pd.DataFrame,
    split_name: str,
    forecast_days: list[pd.Timestamp],
    model_builder: Callable[[], object],
    model_name: str,
    seed: int,
    output_path: Path | None = None,
) -> pd.DataFrame:
    horizon = config.backtest["horizon"]
    window_days = config.backtest["rolling_window_days"]
    retrain_weekday = config.backtest["retrain_weekday"]
    model = None
    predictions = []

    for forecast_day in forecast_days:
        history_df = _window_slice(feature_df, forecast_day=forecast_day, window_days=window_days)
        future_df = _future_slice(feature_df, forecast_day=forecast_day, horizon=horizon)

        if history_df.empty:
            raise ValueError(f"Empty history window for forecast day {forecast_day}.")

        if _retrain_due(forecast_day, retrain_weekday=retrain_weekday, existing_model=model):
            model = model_builder()
            model.fit(history_df)

        prediction_df = model.predict(history_df=history_df, future_df=future_df)
        merged = future_df.loc[:, ["ds", "y"]].merge(prediction_df, on="ds", how="left")
        merged["model"] = model_name
        merged["split"] = split_name
        merged["seed"] = seed
        merged["quantile"] = pd.NA
        merged["metadata"] = json.dumps({"forecast_day": forecast_day.isoformat()})
        predictions.append(merged)

    result = pd.concat(predictions, axis=0, ignore_index=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
    return result

