from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pjm_forecast.retrieval import RetrievalConfig, RetrievalParams, apply_residual_retrieval, tune_retrieval_params


def _make_feature_frame() -> pd.DataFrame:
    rows = []
    horizon_curve = np.linspace(100.0, 200.0, 24)
    phases = {day: 0.2 * day for day in range(20)}
    phases[15] = phases[12]
    for day in range(20):
        day_start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)
        for hour in range(24):
            ts = day_start + pd.Timedelta(hours=hour)
            shape = 12.0 * np.sin(2 * np.pi * hour / 24.0 + phases[day])
            rows.append(
                {
                    "unique_id": "PJM_COMED",
                    "ds": ts,
                    "y": 30.0 + hour * 0.5 + (day % 7),
                    "system_load_forecast": horizon_curve[hour] + shape,
                    "zonal_load_forecast": horizon_curve[hour] * 0.4 + 0.5 * shape,
                    "is_weekend": float(ts.weekday() >= 5),
                    "is_holiday": 0.0,
                    "day_of_week_sin": np.sin(2 * np.pi * ts.weekday() / 7.0),
                    "day_of_week_cos": np.cos(2 * np.pi * ts.weekday() / 7.0),
                    "day_of_year_sin": np.sin(2 * np.pi * ts.dayofyear / 366.0),
                    "day_of_year_cos": np.cos(2 * np.pi * ts.dayofyear / 366.0),
                    "month_sin": np.sin(2 * np.pi * ts.month / 12.0),
                    "month_cos": np.cos(2 * np.pi * ts.month / 12.0),
                }
            )
    return pd.DataFrame(rows)


def _make_prediction_day(feature_df: pd.DataFrame, day: int, residual_value: float, split: str) -> pd.DataFrame:
    forecast_day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)
    day_df = feature_df.loc[feature_df["ds"].dt.normalize() == forecast_day].copy()
    day_df = day_df.loc[:, ["ds", "y"]]
    day_df["y_pred"] = day_df["y"] - residual_value
    day_df["model"] = "nbeatsx"
    day_df["split"] = split
    day_df["seed"] = 7
    day_df["quantile"] = pd.NA
    day_df["metadata"] = json.dumps({"forecast_day": forecast_day.isoformat()})
    return day_df


def test_apply_residual_retrieval_corrects_with_matching_neighbor() -> None:
    feature_df = _make_feature_frame()
    warmup_predictions = pd.concat(
        [
            _make_prediction_day(feature_df, day=8, residual_value=0.5, split="warmup"),
            _make_prediction_day(feature_df, day=10, residual_value=1.0, split="warmup"),
            _make_prediction_day(feature_df, day=12, residual_value=2.0, split="warmup"),
        ],
        ignore_index=True,
    )
    validation_predictions = _make_prediction_day(feature_df, day=15, residual_value=2.0, split="validation")
    config = RetrievalConfig(
        history_days=7,
        price_weight=0.5,
        load_weight=0.5,
        calendar_weight=0.0,
        top_k=1,
        min_gap_days=1,
        residual_clip_quantile=0.975,
    )
    corrected = apply_residual_retrieval(
        feature_df=feature_df,
        base_predictions=validation_predictions,
        initial_memory_predictions=warmup_predictions,
        config=config,
        params=RetrievalParams(alpha=1.0, tau=0.05),
    )
    base_mae = np.mean(np.abs(validation_predictions["y"].to_numpy() - validation_predictions["y_pred"].to_numpy()))
    corrected_mae = np.mean(np.abs(corrected["y"].to_numpy() - corrected["y_pred"].to_numpy()))
    assert corrected_mae < base_mae
    assert corrected["model"].iloc[0] == "nbeatsx_rag"


def test_tune_retrieval_params_prefers_nonzero_alpha_when_neighbors_help() -> None:
    feature_df = _make_feature_frame()
    warmup_predictions = pd.concat(
        [
            _make_prediction_day(feature_df, day=8, residual_value=0.5, split="warmup"),
            _make_prediction_day(feature_df, day=10, residual_value=1.0, split="warmup"),
            _make_prediction_day(feature_df, day=12, residual_value=2.0, split="warmup"),
        ],
        ignore_index=True,
    )
    validation_predictions = _make_prediction_day(feature_df, day=15, residual_value=2.0, split="validation")
    config = RetrievalConfig(
        history_days=7,
        price_weight=0.5,
        load_weight=0.5,
        calendar_weight=0.0,
        top_k=1,
        min_gap_days=1,
        residual_clip_quantile=0.975,
    )
    best_params, scores = tune_retrieval_params(
        feature_df=feature_df,
        validation_predictions=validation_predictions,
        initial_memory_predictions=warmup_predictions,
        config=config,
        alpha_grid=[0.0, 0.5, 1.0],
        tau_grid=[0.05, 0.2],
        volatility_quantile_grid=[None],
    )
    assert best_params.alpha == 1.0
    assert scores["alpha=0.0000,tau=0.0500,vol_q=none"] > scores["alpha=1.0000,tau=0.0500,vol_q=none"]
