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
from pjm_forecast.stacking import (
    StackingRunner,
    build_stacking_backend,
    build_stacking_training_rows,
)
from pjm_forecast.workspace import ArtifactStore


def _make_panel_frame(days: int = 24) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2020-01-01 00:00:00")
    for offset in range(days * 24):
        ts = start + pd.Timedelta(hours=offset)
        day = offset // 24
        hour = ts.hour
        spike_bump = 18.0 if hour in {17, 18} and day >= 10 else 0.0
        rows.append(
            {
                "unique_id": "PJM_COMED",
                "ds": ts,
                "y": 30.0 + day * 0.25 + hour * 0.4 + spike_bump,
                "system_load_forecast": 10000.0 + day * 12.0 + hour * 8.0,
                "zonal_load_forecast": 2100.0 + day * 4.0 + hour * 2.0,
            }
        )
    return pd.DataFrame(rows)


def _make_prediction_day(
    feature_df: pd.DataFrame,
    *,
    day_index: int,
    split: str,
    model_name: str,
    bias: float,
    slope: float,
) -> pd.DataFrame:
    forecast_day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_index)
    day_df = feature_df.loc[feature_df["ds"].dt.normalize() == forecast_day].copy()
    day_df = day_df.loc[:, ["ds", "y"]]
    hours = day_df["ds"].dt.hour.to_numpy(dtype=float)
    day_df["y_pred"] = day_df["y"] + bias + slope * (hours - 12.0)
    day_df["model"] = model_name
    day_df["split"] = split
    day_df["seed"] = 7
    day_df["quantile"] = pd.NA
    day_df["metadata"] = prediction_metadata(forecast_day)
    return day_df


def _prepared_dataset() -> PreparedDataset:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["stacking"]["warmup_days"] = 7
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


class _StubRegressor:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (
            0.55 * features["pred_nbeatsx"].to_numpy(dtype=float)
            + 0.30 * features["pred_lear"].to_numpy(dtype=float)
            + 0.15 * features["pred_seasonal_naive"].to_numpy(dtype=float)
        )


class _StubBackend:
    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]):
        del params
        assert "y" not in features.columns
        assert len(targets) > 0
        return _StubRegressor()


def test_build_stacking_training_rows_uses_prediction_aggregates_and_schema_features() -> None:
    prepared_dataset = _prepared_dataset()
    prediction_frames = {
        "seasonal_naive": _make_prediction_day(
            prepared_dataset.feature_df,
            day_index=15,
            split="validation",
            model_name="seasonal_naive",
            bias=3.0,
            slope=0.05,
        ),
        "lear": _make_prediction_day(
            prepared_dataset.feature_df,
            day_index=15,
            split="validation",
            model_name="lear",
            bias=1.5,
            slope=0.02,
        ),
        "dnn": _make_prediction_day(
            prepared_dataset.feature_df,
            day_index=15,
            split="validation",
            model_name="dnn",
            bias=1.0,
            slope=-0.01,
        ),
        "nbeatsx": _make_prediction_day(
            prepared_dataset.feature_df,
            day_index=15,
            split="validation",
            model_name="nbeatsx",
            bias=0.5,
            slope=0.01,
        ),
    }

    rows = build_stacking_training_rows(
        feature_df=prepared_dataset.feature_df,
        prediction_frames=prediction_frames,
        schema=prepared_dataset.schema,
        base_model_names=["seasonal_naive", "lear", "dnn", "nbeatsx"],
    )

    assert "y" not in rows.feature_columns
    assert "pred_seasonal_naive" in rows.feature_columns
    assert "pred_nbeatsx" in rows.feature_columns
    assert "pred_ensemble_mean" in rows.feature_columns
    assert "pred_gap_to_mean_nbeatsx" in rows.feature_columns
    assert "system_load_forecast" in rows.feature_columns
    assert "price_lag_24" in rows.feature_columns


def test_build_stacking_training_rows_rejects_misaligned_prediction_timestamps() -> None:
    prepared_dataset = _prepared_dataset()
    seasonal_naive = _make_prediction_day(
        prepared_dataset.feature_df,
        day_index=15,
        split="validation",
        model_name="seasonal_naive",
        bias=3.0,
        slope=0.05,
    )
    nbeatsx = _make_prediction_day(
        prepared_dataset.feature_df,
        day_index=15,
        split="validation",
        model_name="nbeatsx",
        bias=0.5,
        slope=0.01,
    ).iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="aligned prediction timestamps"):
        build_stacking_training_rows(
            feature_df=prepared_dataset.feature_df,
            prediction_frames={"seasonal_naive": seasonal_naive, "nbeatsx": nbeatsx},
            schema=prepared_dataset.schema,
            base_model_names=["seasonal_naive", "nbeatsx"],
        )


def test_stacking_runner_tunes_and_applies_with_structured_outputs(tmp_path: Path) -> None:
    prepared_dataset = _prepared_dataset()
    artifacts = _artifact_store(tmp_path)
    config = prepared_dataset.config
    config.raw["stacking"]["enabled"] = True
    requested_keys: list[tuple[str, str]] = []

    model_bias = {
        "seasonal_naive": (3.0, 0.05),
        "lear": (1.5, 0.02),
        "dnn": (1.0, -0.01),
        "nbeatsx": (0.5, 0.01),
    }
    split_days = {
        "stacking_warmup": [8, 10, 12],
        "validation": [15],
        "test": [16],
    }

    def _prediction_loader(
        feature_df: pd.DataFrame,
        forecast_days: list[pd.Timestamp],
        split_name: str,
        seed: int,
        model_name: str,
        variant: str | None,
    ) -> pd.DataFrame:
        del forecast_days, variant
        requested_keys.append((split_name, model_name))
        assert seed == int(config.project["benchmark_seed"])
        assert feature_df is prepared_dataset.feature_df
        bias, slope = model_bias[model_name]
        frames = [
            _make_prediction_day(
                prepared_dataset.feature_df,
                day_index=day_index,
                split=split_name,
                model_name=model_name,
                bias=bias,
                slope=slope,
            )
            for day_index in split_days[split_name]
        ]
        return pd.concat(frames, axis=0, ignore_index=True)

    runner = StackingRunner(
        config=config,
        prepared_dataset=prepared_dataset,
        artifacts=artifacts,
        prediction_loader=_prediction_loader,
        backend_factory=lambda family, seed: _StubBackend(),
    )

    tuning_result = runner.tune(split="validation")
    validation_output = runner.apply(split="validation")
    test_output = runner.apply(split="test")

    params_payload = json.loads(artifacts.stacking_params(config.stacking_output_model_name).read_text(encoding="utf-8"))
    validation_df = pd.read_parquet(validation_output)
    diagnostics_df = pd.read_csv(artifacts.stacking_diagnostics("test", config.stacking_output_model_name))

    assert tuning_result.path == artifacts.stacking_params(config.stacking_output_model_name)
    assert params_payload["output_model_name"] == config.stacking_output_model_name
    assert params_payload["model_family"] == "lightgbm"
    assert params_payload["base_model_names"] == config.stacking_base_model_names
    assert set(params_payload["selected_params"]) == {"num_leaves", "learning_rate", "min_child_samples"}
    assert params_payload["score_grid"]
    assert validation_output.exists()
    assert test_output.exists()
    assert {"ds", "y", "y_pred", "model", "split", "seed", "quantile", "metadata"}.issubset(validation_df.columns)
    assert {"pred_seasonal_naive", "pred_lear", "pred_dnn", "pred_nbeatsx", "pred_ensemble_mean"}.issubset(validation_df.columns)
    assert validation_df["model"].iloc[0] == config.stacking_output_model_name
    assert diagnostics_df["overall_mae_stacker"].notna().all()
    assert diagnostics_df["best_base_model"].notna().all()
    assert ("stacking_warmup", "seasonal_naive") in requested_keys
    assert ("validation", "nbeatsx") in requested_keys
    assert ("test", "lear") in requested_keys


def test_build_stacking_backend_raises_clear_import_error_when_lightgbm_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("missing lightgbm")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "lightgbm", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="lightgbm"):
        build_stacking_backend("lightgbm", seed=7)


def test_build_stacking_backend_rejects_unimplemented_xgboost() -> None:
    with pytest.raises(ValueError, match="xgboost"):
        build_stacking_backend("xgboost", seed=7)
