from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import point_prediction_view


SEGMENTS = [
    "all",
    "actual_normal_day",
    "actual_spike_day",
    "forecast_low_risk_day",
    "forecast_high_risk_day",
]


def compute_normal_day_diagnostics(
    predictions: pd.DataFrame,
    *,
    actual_daily_max_quantile: float = 0.95,
    low_risk_score_column: str = "spike_score",
    low_risk_threshold: float = 0.50,
    low_risk_aggregation: str = "mean",
) -> pd.DataFrame:
    if not 0.0 < actual_daily_max_quantile < 1.0:
        raise ValueError("actual_daily_max_quantile must be between 0 and 1.")
    if low_risk_aggregation not in {"mean", "max"}:
        raise ValueError("low_risk_aggregation must be either 'mean' or 'max'.")

    point = point_prediction_view(predictions).copy()
    point["ds"] = pd.to_datetime(point["ds"])
    point = point.sort_values("ds").reset_index(drop=True)
    point["day"] = point["ds"].dt.floor("D")

    daily_max = point.groupby("day", sort=True)["y"].max().astype(float)
    actual_threshold = float(daily_max.quantile(actual_daily_max_quantile)) if not daily_max.empty else np.nan
    normal_days = daily_max.loc[daily_max <= actual_threshold].index
    spike_days = daily_max.loc[daily_max > actual_threshold].index

    rows = [
        _summarize(
            "all",
            point,
            actual_daily_max_threshold=actual_threshold,
            low_risk_score_column=low_risk_score_column,
            low_risk_score_threshold=low_risk_threshold,
        ),
        _summarize(
            "actual_normal_day",
            point.loc[point["day"].isin(normal_days)],
            actual_daily_max_threshold=actual_threshold,
            low_risk_score_column=low_risk_score_column,
            low_risk_score_threshold=low_risk_threshold,
        ),
        _summarize(
            "actual_spike_day",
            point.loc[point["day"].isin(spike_days)],
            actual_daily_max_threshold=actual_threshold,
            low_risk_score_column=low_risk_score_column,
            low_risk_score_threshold=low_risk_threshold,
        ),
    ]
    rows.extend(
        _forecast_risk_rows(
            point,
            low_risk_score_column,
            low_risk_threshold,
            low_risk_aggregation,
            actual_threshold=actual_threshold,
        )
    )
    return pd.DataFrame(rows, columns=_columns())


def _forecast_risk_rows(
    point: pd.DataFrame,
    score_column: str,
    threshold: float,
    aggregation: str,
    *,
    actual_threshold: float,
) -> list[dict[str, object]]:
    if score_column not in point.columns:
        return [
            _summarize_empty(
                "forecast_low_risk_day",
                actual_daily_max_threshold=actual_threshold,
                low_risk_score_column=score_column,
                low_risk_score_threshold=threshold,
            ),
            _summarize_empty(
                "forecast_high_risk_day",
                actual_daily_max_threshold=actual_threshold,
                low_risk_score_column=score_column,
                low_risk_score_threshold=threshold,
            ),
        ]

    scores = point.loc[:, ["day", score_column]].copy()
    scores[score_column] = scores[score_column].astype(float)
    daily_scores = getattr(scores.groupby("day", sort=True)[score_column], aggregation)()
    low_risk_days = daily_scores.loc[daily_scores <= float(threshold)].index
    high_risk_days = daily_scores.loc[daily_scores > float(threshold)].index
    return [
        _summarize(
            "forecast_low_risk_day",
            point.loc[point["day"].isin(low_risk_days)],
            actual_daily_max_threshold=actual_threshold,
            low_risk_score_column=score_column,
            low_risk_score_threshold=threshold,
        ),
        _summarize(
            "forecast_high_risk_day",
            point.loc[point["day"].isin(high_risk_days)],
            actual_daily_max_threshold=actual_threshold,
            low_risk_score_column=score_column,
            low_risk_score_threshold=threshold,
        ),
    ]


def _summarize(
    segment: str,
    frame: pd.DataFrame,
    *,
    actual_daily_max_threshold: float,
    low_risk_score_column: str,
    low_risk_score_threshold: float,
) -> dict[str, object]:
    if frame.empty:
        return _summarize_empty(
            segment,
            actual_daily_max_threshold=actual_daily_max_threshold,
            low_risk_score_column=low_risk_score_column,
            low_risk_score_threshold=low_risk_score_threshold,
        )

    y = frame["y"].astype(float)
    y_pred = frame["y_pred"].astype(float)
    error = y - y_pred
    abs_error = error.abs()
    y_abs = y.abs()
    ape = abs_error / y_abs.replace(0.0, np.nan)
    smape_denominator = y_abs + y_pred.abs()
    smape_values = np.where(smape_denominator == 0.0, np.nan, 2.0 * abs_error / smape_denominator)

    return {
        "segment": segment,
        "n_days": int(frame["day"].nunique()),
        "n_hours": int(len(frame)),
        "actual_daily_max_threshold": float(actual_daily_max_threshold),
        "low_risk_score_column": low_risk_score_column,
        "low_risk_score_threshold": float(low_risk_score_threshold),
        "q50_wape": float(abs_error.sum() / y_abs.sum()) if float(y_abs.sum()) > 0.0 else np.nan,
        "median_ape": float(ape.median()),
        "p75_ape": float(ape.quantile(0.75)),
        "p90_ape": float(ape.quantile(0.90)),
        "smape": float(np.nanmean(smape_values)),
        "q50_mae": float(abs_error.mean()),
        "q50_bias_mean": float(error.mean()),
        "actual_mean": float(y.mean()),
        "actual_median": float(y.median()),
    }


def _summarize_empty(
    segment: str,
    *,
    actual_daily_max_threshold: float,
    low_risk_score_column: str,
    low_risk_score_threshold: float,
) -> dict[str, object]:
    return {
        "segment": segment,
        "n_days": 0,
        "n_hours": 0,
        "actual_daily_max_threshold": float(actual_daily_max_threshold),
        "low_risk_score_column": low_risk_score_column,
        "low_risk_score_threshold": float(low_risk_score_threshold),
        "q50_wape": np.nan,
        "median_ape": np.nan,
        "p75_ape": np.nan,
        "p90_ape": np.nan,
        "smape": np.nan,
        "q50_mae": np.nan,
        "q50_bias_mean": np.nan,
        "actual_mean": np.nan,
        "actual_median": np.nan,
    }


def _columns() -> list[str]:
    return [
        "segment",
        "n_days",
        "n_hours",
        "actual_daily_max_threshold",
        "low_risk_score_column",
        "low_risk_score_threshold",
        "actual_mean",
        "actual_median",
        "q50_mae",
        "q50_bias_mean",
        "q50_wape",
        "smape",
        "median_ape",
        "p75_ape",
        "p90_ape",
    ]
