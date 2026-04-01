from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.config import load_config
from pjm_forecast.prepared_data import FeatureSchema, PreparedDataset, prediction_metadata
from pjm_forecast.spike_correction import (
    SpikeCorrectorRunner,
    build_spike_backend,
    build_spike_training_rows,
)
from pjm_forecast.workspace import ArtifactStore


def _make_panel_frame(days: int = 24) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2020-01-01 00:00:00")
    for offset in range(days * 24):
        ts = start + pd.Timedelta(hours=offset)
        day = offset // 24
        hour = ts.hour
        spike_bump = 24.0 if hour in {17, 18} and day >= 10 else 0.0
        rows.append(
            {
                "unique_id": "PJM_COMED",
                "ds": ts,
                "y": 28.0 + day * 0.25 + hour * 0.5 + spike_bump,
                "system_load_forecast": 10000.0 + day * 10.0 + hour * 8.0,
                "zonal_load_forecast": 2000.0 + day * 4.0 + hour * 3.0,
            }
        )
    return pd.DataFrame(rows)


def _make_prediction_day(feature_df: pd.DataFrame, day_index: int, split: str) -> pd.DataFrame:
    forecast_day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_index)
    day_df = feature_df.loc[feature_df["ds"].dt.normalize() == forecast_day].copy()
    day_df = day_df.loc[:, ["ds", "y"]]
    rank_desc = day_df["y"].rank(method="first", ascending=False)
    residual = np.where(rank_desc <= 2, 8.0, 1.0)
    day_df["y_pred"] = day_df["y"] - residual
    day_df["model"] = "nbeatsx"
    day_df["split"] = split
    day_df["seed"] = 7
    day_df["quantile"] = pd.NA
    day_df["metadata"] = prediction_metadata(forecast_day)
    return day_df


def _prepared_dataset() -> PreparedDataset:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["spike_correction"]["warmup_days"] = 7
    schema = FeatureSchema(config)
    panel_df = _make_panel_frame(days=24)
    feature_df = schema.build_feature_frame(panel_df)
    return PreparedDataset(
        config=config,
        schema=schema,
        panel_df=panel_df,
        feature_df=feature_df,
        split_boundaries={
            "train_end": pd.Timestamp("2020-01-14"),
            "validation_start": pd.Timestamp("2020-01-16"),
            "validation_end": pd.Timestamp("2020-01-16"),
            "test_start": pd.Timestamp("2020-01-17"),
            "test_end": pd.Timestamp("2020-01-17"),
        },
    )


def _artifact_store(tmp_path: Path) -> ArtifactStore:
    directories = {
        "raw_data_dir": tmp_path / "data" / "raw",
        "processed_data_dir": tmp_path / "data" / "processed",
        "artifact_dir": tmp_path / "artifacts",
        "hyperparameter_dir": tmp_path / "artifacts" / "hyperparameters",
        "prediction_dir": tmp_path / "artifacts" / "predictions",
        "metrics_dir": tmp_path / "artifacts" / "metrics",
        "plots_dir": tmp_path / "artifacts" / "plots",
        "report_dir": tmp_path / "artifacts" / "report",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactStore(directories)


class _StubClassifier:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probs = np.where(features["base_hour_rank_desc"].to_numpy(dtype=float) <= 2.0, 0.9, 0.1)
        return np.column_stack([1.0 - probs, probs])


class _StubRegressor:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.where(features["base_hour_rank_desc"].to_numpy(dtype=float) <= 2.0, 8.0, 1.0)


class _StubBackend:
    def fit_classifier(self, features: pd.DataFrame, labels: pd.Series, params: dict[str, object]):
        del params
        assert "y" not in features.columns
        assert "residual" not in features.columns
        assert labels.isin([0, 1]).all()
        return _StubClassifier()

    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]):
        del params
        assert "y" not in features.columns
        assert "residual" not in features.columns
        assert len(targets) > 0
        return _StubRegressor()


def test_build_spike_training_rows_uses_only_forecast_time_features() -> None:
    prepared_dataset = _prepared_dataset()
    prediction_df = _make_prediction_day(prepared_dataset.feature_df, day_index=15, split="validation")

    rows = build_spike_training_rows(
        feature_df=prepared_dataset.feature_df,
        base_predictions=prediction_df,
        schema=prepared_dataset.schema,
    )

    assert "y" not in rows.feature_columns
    assert "residual" not in rows.feature_columns
    assert "spike_label" not in rows.feature_columns
    assert "y_pred_base" in rows.feature_columns
    assert "system_load_forecast" in rows.feature_columns
    assert "price_lag_24" in rows.feature_columns


def test_spike_corrector_runner_tunes_and_applies_with_structured_outputs(tmp_path: Path) -> None:
    prepared_dataset = _prepared_dataset()
    artifacts = _artifact_store(tmp_path)
    config = prepared_dataset.config
    config.raw["spike_correction"]["enabled"] = True
    requested_splits: list[str] = []

    warmup_predictions = pd.concat(
        [
            _make_prediction_day(prepared_dataset.feature_df, day_index=8, split="spike_warmup"),
            _make_prediction_day(prepared_dataset.feature_df, day_index=10, split="spike_warmup"),
            _make_prediction_day(prepared_dataset.feature_df, day_index=12, split="spike_warmup"),
        ],
        ignore_index=True,
    )
    validation_predictions = _make_prediction_day(prepared_dataset.feature_df, day_index=15, split="validation")
    test_predictions = _make_prediction_day(prepared_dataset.feature_df, day_index=16, split="test")

    def _prediction_loader(
        feature_df: pd.DataFrame,
        forecast_days: list[pd.Timestamp],
        split_name: str,
        seed: int,
        model_name: str,
        variant: str | None,
    ) -> pd.DataFrame:
        del forecast_days, variant
        requested_splits.append(split_name)
        assert seed == int(config.project["benchmark_seed"])
        assert model_name == config.spike_base_model_name
        assert feature_df is prepared_dataset.feature_df
        lookup = {
            "spike_warmup": warmup_predictions,
            "validation": validation_predictions,
            "test": test_predictions,
        }
        return lookup[split_name].copy()

    runner = SpikeCorrectorRunner(
        config=config,
        prepared_dataset=prepared_dataset,
        artifacts=artifacts,
        prediction_loader=_prediction_loader,
        backend_factory=lambda family, seed: _StubBackend(),
    )

    tuning_result = runner.tune(base_model=config.spike_base_model_name, split="validation")
    validation_output = runner.apply(base_model=config.spike_base_model_name, split="validation")
    test_output = runner.apply(base_model=config.spike_base_model_name, split="test")

    params_payload = json.loads(artifacts.spike_params(config.spike_output_model_name).read_text(encoding="utf-8"))
    validation_df = pd.read_parquet(validation_output)
    diagnostics_df = pd.read_csv(artifacts.spike_diagnostics("test", config.spike_output_model_name))

    assert tuning_result.path == artifacts.spike_params(config.spike_output_model_name)
    assert params_payload["output_model_name"] == config.spike_output_model_name
    assert params_payload["model_family"] == "lightgbm"
    assert set(params_payload["selected_params"]) == {"spike_quantile", "gate_threshold", "delta_clip_quantile"}
    assert params_payload["score_grid"]
    assert validation_output.exists()
    assert test_output.exists()
    assert {"ds", "y", "y_pred", "model", "split", "seed", "quantile", "metadata"}.issubset(validation_df.columns)
    assert {"y_pred_base", "spike_prob", "spike_flag", "spike_delta"}.issubset(validation_df.columns)
    assert validation_df["model"].iloc[0] == config.spike_output_model_name
    assert diagnostics_df["overall_mae_delta"].notna().all()
    assert diagnostics_df["spike_precision"].notna().all()
    assert requested_splits.count("spike_warmup") >= 1
    assert requested_splits.count("validation") >= 1
    assert requested_splits.count("test") >= 1


def test_build_spike_backend_raises_clear_import_error_when_lightgbm_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("missing lightgbm")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "lightgbm", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="lightgbm"):
        build_spike_backend("lightgbm", seed=7)


def test_build_spike_backend_rejects_unimplemented_xgboost() -> None:
    with pytest.raises(ValueError, match="xgboost"):
        build_spike_backend("xgboost", seed=7)
