from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.backtest.engine import run_rolling_backtest
from pjm_forecast.config import load_config
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.prepared_data import FeatureSchema
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


class FitSignatureModel(ForecastModel):
    def __init__(self, fit_markers: list[pd.Timestamp]) -> None:
        self.fit_markers = fit_markers
        self.current_marker: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        marker = pd.Timestamp(train_df["ds"].max())
        self.fit_markers.append(marker)
        self.current_marker = float(marker.day)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self.current_marker is None:
            raise RuntimeError("Model must be fit before predict.")
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": np.full(len(future_df), self.current_marker)})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"current_marker": self.current_marker}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FitSignatureModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(fit_markers=[])
        model.current_marker = payload["current_marker"]
        return model


class MisalignedPredictionModel(ForecastModel):
    def fit(self, train_df: pd.DataFrame) -> None:
        del train_df

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        shifted_ds = future_df["ds"] + pd.Timedelta(hours=1)
        return pd.DataFrame({"ds": shifted_ds.to_numpy(), "y_pred": np.ones(len(future_df), dtype=float)})

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MisalignedPredictionModel":
        del path
        return cls()


class QuantileResumeModel(ForecastModel):
    def fit(self, train_df: pd.DataFrame) -> None:
        del train_df

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        rows = []
        for index, ds in enumerate(future_df["ds"]):
            base = 10.0 + index
            for quantile, offset in [(0.1, -1.0), (0.5, 0.0), (0.9, 1.0)]:
                rows.append({"ds": ds, "quantile": quantile, "y_pred": base + offset})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QuantileResumeModel":
        del path
        return cls()


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
    FeatureSchema(config).validate_prediction_frame(result, require_metadata=True)


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


def test_backtest_resume_matches_clean_run_without_extra_retrain(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    config.raw["backtest"]["retrain_weekday"] = 0
    feature_df = _feature_frame()
    forecast_days = [
        pd.Timestamp("2017-01-09 00:00:00"),
        pd.Timestamp("2017-01-10 00:00:00"),
        pd.Timestamp("2017-01-11 00:00:00"),
    ]
    output_path = tmp_path / "fit_signature_validation_seed7.parquet"

    full_fit_markers: list[pd.Timestamp] = []
    full_result = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=lambda: FitSignatureModel(full_fit_markers),
        model_name="fit_signature",
        seed=7,
    )

    resumed_fit_markers: list[pd.Timestamp] = []
    run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=[forecast_days[0]],
        model_builder=lambda: FitSignatureModel(resumed_fit_markers),
        model_name="fit_signature",
        seed=7,
        output_path=output_path,
    )
    resumed_result = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=lambda: FitSignatureModel(resumed_fit_markers),
        model_name="fit_signature",
        seed=7,
        output_path=output_path,
    )

    comparable_columns = ["ds", "y", "y_pred", "model", "split", "seed", "metadata"]
    pd.testing.assert_frame_equal(
        full_result.loc[:, comparable_columns].reset_index(drop=True),
        resumed_result.loc[:, comparable_columns].reset_index(drop=True),
    )


def test_backtest_rejects_non_contiguous_resumed_chunks(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [
        pd.Timestamp("2017-01-09 00:00:00"),
        pd.Timestamp("2017-01-10 00:00:00"),
        pd.Timestamp("2017-01-11 00:00:00"),
    ]
    output_path = tmp_path / "seasonal_naive_validation_seed7.parquet"

    run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=[forecast_days[0], forecast_days[2]],
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
        output_path=output_path,
    )

    with pytest.raises(ValueError, match="contiguous prefix"):
        run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=forecast_days,
            model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
            model_name="seasonal_naive",
            seed=7,
            output_path=output_path,
        )


def test_backtest_rejects_stale_chunk_contract_mismatch(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00"), pd.Timestamp("2017-01-10 00:00:00")]
    output_path = tmp_path / "seasonal_naive_validation_seed7.parquet"

    run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=[forecast_days[0]],
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
        output_path=output_path,
    )
    chunk_path = output_path.parent / "chunks" / output_path.stem / "2017-01-09.parquet"
    chunk_df = pd.read_parquet(chunk_path)
    chunk_df["seed"] = 99
    chunk_df.to_parquet(chunk_path, index=False)

    with pytest.raises(ValueError, match="seed"):
        run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=forecast_days,
            model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
            model_name="seasonal_naive",
            seed=7,
            output_path=output_path,
        )


def test_backtest_rejects_misaligned_model_prediction_output() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00")]

    with pytest.raises(ValueError, match="do not align"):
        run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=forecast_days,
            model_builder=MisalignedPredictionModel,
            model_name="misaligned_prediction",
            seed=7,
        )


def test_backtest_rejects_hour_regime_calibration_without_spike_context() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    config.raw.setdefault("report", {})["quantile_postprocess"] = {
        "monotonic": True,
        "calibration": {
            "enabled": True,
            "source_split": "validation",
            "method": "cqr_asymmetric",
            "group_by": "hour_x_regime",
            "regime_score_column": "spike_score",
            "regime_threshold": 0.5,
            "min_group_size": 24,
        },
    }
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00")]

    with pytest.raises(ValueError, match="requires future feature column 'spike_score'"):
        run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=forecast_days,
            model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
            model_name="seasonal_naive",
            seed=7,
        )


def test_backtest_writes_spike_context_for_hour_cqr_diagnostics(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    config.raw.setdefault("report", {})["quantile_postprocess"] = {
        "monotonic": True,
        "calibration": {
            "enabled": True,
            "source_split": "validation",
            "method": "cqr_asymmetric",
            "group_by": "hour",
            "regime_score_column": "spike_score",
            "regime_threshold": 0.5,
            "min_group_size": 24,
        },
    }
    feature_df = _feature_frame()
    feature_df["spike_score"] = 0.75
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00")]
    output_path = tmp_path / "seasonal_naive_validation_seed7.parquet"

    run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=forecast_days,
        model_builder=lambda: SeasonalNaiveModel(seasonal_lag_hours=24),
        model_name="seasonal_naive",
        seed=7,
        output_path=output_path,
    )

    prediction_df = pd.read_parquet(output_path)
    assert "spike_score" in prediction_df.columns
    assert prediction_df["spike_score"].eq(0.75).all()


def test_backtest_rejects_resumed_chunk_with_missing_expected_quantile(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["backtest"]["rolling_window_days"] = 8
    config.raw["models"]["quantile_resume"] = {
        "loss_name": "mqloss",
        "quantiles": [0.1, 0.5, 0.9],
    }
    feature_df = _feature_frame()
    forecast_days = [pd.Timestamp("2017-01-09 00:00:00"), pd.Timestamp("2017-01-10 00:00:00")]
    output_path = tmp_path / "quantile_resume_validation_seed7.parquet"

    run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name="validation",
        forecast_days=[forecast_days[0]],
        model_builder=QuantileResumeModel,
        model_name="quantile_resume",
        seed=7,
        output_path=output_path,
    )
    chunk_path = output_path.parent / "chunks" / output_path.stem / "2017-01-09.parquet"
    chunk_df = pd.read_parquet(chunk_path)
    chunk_df = chunk_df.loc[~np.isclose(chunk_df["quantile"].astype(float), 0.9)].reset_index(drop=True)
    chunk_df.to_parquet(chunk_path, index=False)

    with pytest.raises(ValueError, match="quantile grid does not match expected quantiles"):
        run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=forecast_days,
            model_builder=QuantileResumeModel,
            model_name="quantile_resume",
            seed=7,
            output_path=output_path,
        )
