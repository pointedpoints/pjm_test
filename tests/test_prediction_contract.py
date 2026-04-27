from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.backtest.engine import run_rolling_backtest
from pjm_forecast.config import load_config
from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.model_io import validate_model_prediction_output
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.prediction_contract import enforce_monotonic_quantiles
from pjm_forecast.prepared_data import FeatureSchema


def _future_frame() -> pd.DataFrame:
    return pd.DataFrame({"ds": pd.date_range("2026-01-01 00:00:00", periods=4, freq="h")})


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


class QuantileDummyModel(ForecastModel):
    def fit(self, train_df: pd.DataFrame) -> None:
        del train_df

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        rows = []
        for index, ds in enumerate(future_df["ds"]):
            base = 10.0 + index
            for quantile, offset in [(0.1, -2.0), (0.5, 0.0), (0.9, 2.0)]:
                rows.append({"ds": ds, "quantile": quantile, "y_pred": base + offset})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QuantileDummyModel":
        del path
        return cls()


def test_validate_model_prediction_output_normalizes_point_predictions() -> None:
    future_df = _future_frame()
    prediction_df = pd.DataFrame({"ds": future_df["ds"], "y_pred": [1.0, 2.0, 3.0, 4.0]})
    normalized = validate_model_prediction_output(prediction_df, future_df=future_df, model_name="dummy")

    assert list(normalized.columns) == ["ds", "quantile", "y_pred"]
    assert normalized["quantile"].isna().all()


def test_validate_model_prediction_output_normalizes_quantile_predictions() -> None:
    future_df = _future_frame()
    prediction_df = QuantileDummyModel().predict(pd.DataFrame(), future_df)
    normalized = validate_model_prediction_output(prediction_df, future_df=future_df, model_name="dummy")

    assert list(normalized.columns) == ["ds", "quantile", "y_pred"]
    assert sorted(normalized["quantile"].dropna().unique().tolist()) == [0.1, 0.5, 0.9]
    assert len(normalized) == 12


def test_validate_model_prediction_output_rejects_missing_expected_quantile() -> None:
    future_df = _future_frame()
    prediction_df = QuantileDummyModel().predict(pd.DataFrame(), future_df)
    prediction_df = prediction_df.loc[~np.isclose(prediction_df["quantile"], 0.5)].reset_index(drop=True)

    with pytest.raises(ValueError, match="quantile grid does not match expected quantiles"):
        validate_model_prediction_output(
            prediction_df,
            future_df=future_df,
            model_name="dummy",
            expected_quantiles=[0.1, 0.5, 0.9],
        )


def test_validate_model_prediction_output_rejects_extra_expected_quantile() -> None:
    future_df = _future_frame()
    prediction_df = QuantileDummyModel().predict(pd.DataFrame(), future_df)
    prediction_df = pd.concat(
        [
            prediction_df,
            pd.DataFrame(
                {
                    "ds": future_df["ds"],
                    "quantile": [0.7] * len(future_df),
                    "y_pred": [99.0] * len(future_df),
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="quantile grid does not match expected quantiles"):
        validate_model_prediction_output(
            prediction_df,
            future_df=future_df,
            model_name="dummy",
            expected_quantiles=[0.1, 0.5, 0.9],
        )


def test_compute_metrics_uses_p50_for_point_metrics_and_pinball_for_quantiles() -> None:
    prediction_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                ]
            ),
            "y": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "y_pred": [8.0, 10.0, 12.0, 18.0, 20.0, 22.0],
            "model": ["nbeatsx"] * 6,
            "split": ["validation"] * 6,
            "seed": [7] * 6,
            "quantile": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
            "metadata": ["{}"] * 6,
        }
    )
    metrics = compute_metrics(prediction_df)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert "pinball" in metrics
    assert metrics["pinball"] > 0.0


def test_compute_quantile_diagnostics_reports_crossing_coverage_and_width() -> None:
    prediction_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                ]
            ),
            "y": [10.0] * 8 + [20.0] * 8,
            "y_pred": [
                6.0,
                7.0,
                8.0,
                10.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                20.0,
                22.0,
                23.0,
                24.0,
                25.0,
            ],
            "model": ["nbeatsx"] * 16,
            "split": ["validation"] * 16,
            "seed": [7] * 16,
            "quantile": [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995] * 2,
            "metadata": ["{}"] * 16,
        }
    )

    diagnostics = compute_quantile_diagnostics(prediction_df)

    assert diagnostics["has_quantiles"] is True
    assert diagnostics["crossing_rate"] == 0.0
    assert diagnostics["coverage_80"] == 1.0
    assert diagnostics["coverage_90"] == 1.0
    assert diagnostics["coverage_98"] == 1.0
    assert diagnostics["width_80"] == 4.0
    assert diagnostics["width_90"] == 6.0
    assert diagnostics["width_98"] == 8.0
    assert diagnostics["pinball"] > 0.0
    assert diagnostics["crps"] > 0.0
    assert 0.0 <= diagnostics["pit_mean"] <= 1.0
    assert diagnostics["pit_variance"] >= 0.0
    assert diagnostics["q50_mae"] == 0.0
    assert diagnostics["q50_bias_mean"] == 0.0
    assert diagnostics["q50_bias_median"] == 0.0
    assert diagnostics["q95_q99_gap_mean"] == 1.0
    assert diagnostics["q95_q99_slope_mean"] == pytest.approx(25.0)
    assert diagnostics["q99_q995_gap_mean"] == 1.0
    assert diagnostics["q99_q995_slope_mean"] == pytest.approx(200.0)
    assert diagnostics["q99_exceedance_rate"] == 0.0
    assert diagnostics["q99_excess_mean"] == 0.0
    assert diagnostics["worst_q99_underprediction"] == 0.0


def test_compute_quantile_diagnostics_returns_na_for_point_predictions() -> None:
    prediction_df = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "y": [1.0, 2.0],
            "y_pred": [1.5, 2.5],
            "model": ["seasonal_naive"] * 2,
            "split": ["validation"] * 2,
            "seed": [7] * 2,
            "quantile": [pd.NA] * 2,
            "metadata": ["{}"] * 2,
        }
    )

    diagnostics = compute_quantile_diagnostics(prediction_df)

    assert diagnostics["has_quantiles"] is False
    assert np.isnan(diagnostics["crossing_rate"])
    assert np.isnan(diagnostics["coverage_80"])
    assert np.isnan(diagnostics["width_98"])
    assert np.isnan(diagnostics["crps"])
    assert np.isnan(diagnostics["pit_mean"])
    assert np.isnan(diagnostics["q50_mae"])
    assert np.isnan(diagnostics["q50_bias_mean"])
    assert np.isnan(diagnostics["q95_q99_gap_mean"])
    assert np.isnan(diagnostics["q99_exceedance_rate"])


def test_enforce_monotonic_quantiles_removes_crossings() -> None:
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                ]
            ),
            "y": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "y_pred": [9.0, 8.0, 11.0, 17.0, 21.0, 20.0],
            "model": ["nbeatsx"] * 6,
            "split": ["validation"] * 6,
            "seed": [7] * 6,
            "quantile": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
            "metadata": ["{}"] * 6,
        }
    )

    corrected = enforce_monotonic_quantiles(frame)
    grouped = corrected.sort_values(["ds", "quantile"]).groupby("ds")["y_pred"].apply(list)
    assert grouped.iloc[0] == [9.0, 9.0, 11.0]
    assert grouped.iloc[1] == [17.0, 21.0, 21.0]


def test_backtest_accepts_quantile_prediction_contract() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00")]
    result = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=QuantileDummyModel,
        model_name="quantile_dummy",
        seed=7,
    )

    assert len(result) == 24 * 3
    assert sorted(result["quantile"].dropna().unique().tolist()) == [0.1, 0.5, 0.9]
    FeatureSchema(config).validate_prediction_frame(result, require_metadata=True)
