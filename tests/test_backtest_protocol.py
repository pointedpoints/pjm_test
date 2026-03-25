from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.backtest.engine import run_rolling_backtest
from pjm_forecast.config import load_config
from pjm_forecast.models.seasonal_naive import SeasonalNaiveModel


def _feature_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2017-01-01 00:00:00")
    for offset in range(24 * 30):
        ts = start + pd.Timedelta(hours=offset)
        rows.append(
            {
                "unique_id": "PJM_COMED",
                "ds": ts,
                "y": float(offset),
                "system_load_forecast": float(offset + 100),
                "zonal_load_forecast": float(offset + 200),
                "is_weekend": int(ts.weekday() >= 5),
                "is_holiday": 0,
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "day_of_week_sin": 0.0,
                "day_of_week_cos": 1.0,
                "day_of_year_sin": 0.0,
                "day_of_year_cos": 1.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "price_lag_24": float(offset - 24) if offset >= 24 else float("nan"),
                "price_lag_48": float(offset - 48) if offset >= 48 else float("nan"),
                "price_lag_72": float(offset - 72) if offset >= 72 else float("nan"),
                "price_lag_168": float(offset - 168) if offset >= 168 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_backtest_generates_prediction_contract() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00"), pd.Timestamp("2017-01-10 00:00:00")]
    result = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
    )
    assert len(result) == 48
    assert set(["ds", "y", "y_pred", "model", "split", "seed", "quantile", "metadata"]).issubset(result.columns)


def test_backtest_resumes_from_existing_chunks(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00"), pd.Timestamp("2017-01-10 00:00:00")]
    output_path = tmp_path / "seasonal_naive_validation_seed7.parquet"

    first = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=[forecast_days[0]],
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
        output_path=output_path,
    )
    assert len(first) == 24

    resumed = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
        output_path=output_path,
    )
    assert len(resumed) == 48
    assert output_path.exists()
