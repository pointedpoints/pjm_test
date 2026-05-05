from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pjm_forecast.data.ingress import PreparedDataResult
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.pipeline import STAGE_ORDER, _run_audit_event_risk_overlay, _run_export_model_snapshot
from pjm_forecast.prepared_data import PreparedDataset
from pjm_forecast.workspace import Workspace, resolve_mlp_unit_search_options


class SnapshotStubModel(ForecastModel):
    supports_fitted_snapshot = True

    def __init__(self) -> None:
        self.fit_rows = 0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.fit_rows = len(train_df)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": [42.0] * len(future_df)})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps({"fit_rows": self.fit_rows}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SnapshotStubModel":
        del path
        return cls()


def _write_csv(tmp_path: Path, hours: int = 24 * 420) -> Path:
    rows = []
    start = pd.Timestamp("2013-01-01 00:00:00")
    for offset in range(hours):
        rows.append(
            {
                "Date": start + pd.Timedelta(hours=offset),
                "Zonal COMED price": float(offset),
                "System load forecast": float(10_000 + offset),
                "Zonal COMED load foecast": float(2_000 + offset),
            }
        )
    csv_path = tmp_path / "PJM.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_temp_config(tmp_path: Path, csv_path: Path, *, with_weather: bool = False) -> Path:
    base_config = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    base_config["project"]["root_override"] = str(tmp_path / "run")
    base_config["dataset"]["local_csv_path"] = str(csv_path)
    base_config["backtest"]["years_test"] = 1
    base_config["backtest"]["validation_days"] = 28
    base_config["backtest"]["benchmark_models"] = ["seasonal_naive"]
    base_config["backtest"]["rolling_window_days"] = 8
    if with_weather:
        base_config["features"]["future_exog"].extend(
            [
                "weather_temp_mean",
                "weather_cloud_cover_mean",
            ]
        )
        base_config["features"]["lag_sources"] = [
            "system_load_forecast",
            "zonal_load_forecast",
            "weather_temp_mean",
            "weather_cloud_cover_mean",
        ]
        base_config["features"]["source_lags"] = [24]
        base_config["weather"] = {
            "enabled": True,
            "provider": "open_meteo_historical_forecast",
            "model": "hrrr",
            "timezone": "America/Chicago",
            "output_columns": [
                "weather_temp_mean",
                "weather_cloud_cover_mean",
            ],
            "points": [
                {"name": "north", "latitude": 41.9, "longitude": -87.7, "weight": 0.6},
                {"name": "south", "latitude": 41.7, "longitude": -87.6, "weight": 0.4},
            ],
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return config_path


def _quantile_prediction_frame(split: str) -> pd.DataFrame:
    rows = []
    for day, spike_score in [("2026-01-01", 0.2), ("2026-01-02", 0.95)]:
        for hour in [0, 19]:
            ds = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            for quantile, y_pred in [(0.5, 95.0), (0.95, 100.0), (0.99, 105.0), (0.995, 110.0)]:
                rows.append(
                    {
                        "ds": ds,
                        "y": 120.0 if spike_score > 0.9 else 100.0,
                        "y_pred": y_pred,
                        "quantile": quantile,
                        "model": "nhits_tail_grid_weighted_main",
                        "split": split,
                        "seed": 7,
                        "metadata": "{}",
                        "spike_score": spike_score,
                    }
                )
    return pd.DataFrame(rows)


def test_workspace_open_respects_root_override_and_artifact_contract(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    assert workspace.artifacts.feature_store() == (tmp_path / "run" / "data" / "processed" / "feature_store.parquet").resolve()
    assert workspace.artifacts.prediction("seasonal_naive", "test", 7).name == "seasonal_naive_test_seed7.parquet"
    assert workspace.artifacts.prediction_chunk_dir("seasonal_naive", "test", 7).name == "seasonal_naive_test_seed7"
    assert workspace.artifacts.report_asset("test_metrics.csv") == (tmp_path / "run" / "artifacts" / "report" / "test_metrics.csv").resolve()
    assert workspace.artifacts.quantile_diagnostics("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_quantile_diagnostics.csv"
    ).resolve()
    assert workspace.artifacts.spike_score_diagnostics("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_spike_score_diagnostics.csv"
    ).resolve()
    assert workspace.artifacts.scenario_diagnostics("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_scenario_diagnostics.csv"
    ).resolve()
    assert workspace.artifacts.event_risk_audit_dir("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_event_risk_tail_overlay"
    ).resolve()
    assert workspace.artifacts.quality_gate_summary("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_quality_gate_summary.csv"
    ).resolve()
    assert workspace.artifacts.run_manifest("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_run_manifest.json"
    ).resolve()
    assert workspace.artifacts.snapshot_manifest("nbeatsx_snapshot") == (
        tmp_path / "run" / "artifacts" / "models" / "nbeatsx_snapshot" / "manifest.json"
    ).resolve()


def test_workspace_audit_event_risk_overlay_writes_expected_files(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["backtest"]["benchmark_models"] = ["nhits_tail_grid_weighted_main"]
    workspace.config.raw["models"]["nhits_tail_grid_weighted_main"] = {"type": "nhits"}
    workspace.config.raw["report"]["quantile_postprocess"] = {
        "monotonic": True,
        "calibration": {
            "enabled": True,
            "source_split": "validation",
            "method": "cqr_asymmetric",
            "group_by": "hour",
            "min_group_size": 1,
        },
        "event_risk_tail_overlay": {
            "enabled": True,
            "source_split": "validation",
            "risk_score_column": "spike_score",
            "risk_aggregation": "mean",
            "risk_threshold_quantile": 0.50,
            "residual_quantile": 1.0,
            "max_uplift": 25.0,
            "target_quantiles": [0.99, 0.995],
            "validation_holdout_days": 1,
        },
    }

    for split in ["validation", "test"]:
        path = workspace.artifacts.prediction("nhits_tail_grid_weighted_main", split, 7)
        path.parent.mkdir(parents=True, exist_ok=True)
        _quantile_prediction_frame(split).to_parquet(path, index=False)

    output_dir = workspace.audit_event_risk_overlay("test")

    assert (output_dir / "overlay_implementation_audit.json").exists()
    assert (output_dir / "spike_score_audit.json").exists()
    assert (output_dir / "width_by_regime.csv").exists()


def test_workspace_finalize_quality_flow_writes_summary_and_manifest(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    workspace.artifacts.metrics("test").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run": "seasonal_naive_seed_7",
                "model": "seasonal_naive",
                "seed": 7,
                "mae": 18.5,
                "pinball": 2.25,
            }
        ]
    ).to_csv(workspace.artifacts.metrics("test"), index=False)
    pd.DataFrame(
        [
            {
                "run": "seasonal_naive_seed_7",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.025,
                "post_q99_excess_mean": 1.5,
                "post_worst_q99_underprediction": 12.0,
                "post_width_98": 105.2,
            }
        ]
    ).to_csv(workspace.artifacts.quantile_diagnostics("test"), index=False)
    event_audit_dir = workspace.artifacts.event_risk_audit_dir("test")
    event_audit_dir.mkdir(parents=True, exist_ok=True)
    (event_audit_dir / "overlay_implementation_audit.json").write_text(
        json.dumps({"selected_variant": "hour_cqr"}),
        encoding="utf-8",
    )
    (event_audit_dir / "spike_score_audit.json").write_text(
        json.dumps({"availability_status": "PASS"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"regime": "all", "before_width_98": 100.0, "after_width_98": 105.2},
            {"regime": "normal", "before_width_98": 100.0, "after_width_98": 105.2},
        ]
    ).to_csv(event_audit_dir / "width_by_regime.csv", index=False)

    written = workspace.finalize_quality_flow("test")

    assert written == [
        workspace.artifacts.quality_gate_summary("test"),
        workspace.artifacts.run_manifest("test"),
    ]
    assert workspace.artifacts.quality_gate_summary("test").exists()
    assert workspace.artifacts.run_manifest("test").exists()
    manifest = json.loads(workspace.artifacts.run_manifest("test").read_text(encoding="utf-8"))
    artifact_paths = [item["path"] for item in manifest["artifacts"]]
    assert str(event_audit_dir / "overlay_implementation_audit.json") in artifact_paths
    assert str(event_audit_dir / "spike_score_audit.json") in artifact_paths
    assert str(event_audit_dir / "width_by_regime.csv") in artifact_paths
    copied = workspace.export_report("test")
    assert workspace.artifacts.report_asset("test_quality_gate_summary.csv") in copied
    assert workspace.artifacts.report_asset("test_run_manifest.json") in copied


def test_workspace_finalize_quality_flow_writes_manifest_when_required_artifact_is_missing(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    workspace.artifacts.metrics("test").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run": "seasonal_naive_seed_7",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.025,
            }
        ]
    ).to_csv(workspace.artifacts.quantile_diagnostics("test"), index=False)

    with pytest.raises(FileNotFoundError):
        workspace.finalize_quality_flow("test")

    assert workspace.artifacts.run_manifest("test").exists()
    manifest = json.loads(workspace.artifacts.run_manifest("test").read_text(encoding="utf-8"))
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    assert artifacts[str(workspace.artifacts.metrics("test"))]["exists"] is False
    assert artifacts[str(workspace.artifacts.quality_gate_summary("test"))]["exists"] is False


def test_workspace_audit_event_risk_overlay_rejects_non_validation_source_split(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["report"]["quantile_postprocess"] = {
        "event_risk_tail_overlay": {
            "enabled": True,
            "source_split": "test",
        },
    }

    with pytest.raises(ValueError, match="event-risk audit must use validation source_split"):
        workspace.audit_event_risk_overlay("test")


@pytest.mark.parametrize(
    "postprocess_config",
    [
        ["not-a-mapping"],
        {"event_risk_tail_overlay": ["not-a-mapping"]},
    ],
)
def test_workspace_audit_event_risk_overlay_rejects_malformed_event_config(
    tmp_path: Path,
    postprocess_config: object,
) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["report"]["quantile_postprocess"] = postprocess_config

    with pytest.raises(ValueError, match="report.quantile_postprocess"):
        workspace.audit_event_risk_overlay("test")


def test_workspace_audit_event_risk_overlay_rejects_malformed_calibration_config(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["report"]["quantile_postprocess"] = {
        "calibration": [],
        "event_risk_tail_overlay": {
            "enabled": True,
            "source_split": "validation",
        },
    }

    with pytest.raises(ValueError, match="report.quantile_postprocess.calibration"):
        workspace.audit_event_risk_overlay("test")


def test_pipeline_stage_order_includes_quality_closure() -> None:
    assert STAGE_ORDER == [
        "prepare_data",
        "tune_model",
        "backtest_all_models",
        "evaluate_and_plot",
        "audit_event_risk_overlay",
        "finalize_quality_flow",
        "export_report_assets",
        "export_model_snapshot",
    ]


def test_pipeline_audit_event_risk_overlay_skips_when_overlay_disabled() -> None:
    class Config:
        raw = {"report": {"quantile_postprocess": {"event_risk_tail_overlay": {"enabled": False}}}}

    class WorkspaceStub:
        config = Config()
        audit_calls = 0

        def audit_event_risk_overlay(self, split: str) -> None:
            del split
            self.audit_calls += 1

    workspace = WorkspaceStub()

    _run_audit_event_risk_overlay(workspace, "test")

    assert workspace.audit_calls == 0


def test_pipeline_audit_event_risk_overlay_runs_when_overlay_enabled() -> None:
    class Config:
        raw = {"report": {"quantile_postprocess": {"event_risk_tail_overlay": {"enabled": True}}}}

    class WorkspaceStub:
        config = Config()
        audit_calls: list[str]

        def __init__(self) -> None:
            self.audit_calls = []

        def audit_event_risk_overlay(self, split: str) -> None:
            self.audit_calls.append(split)

    workspace = WorkspaceStub()

    _run_audit_event_risk_overlay(workspace, "test")

    assert workspace.audit_calls == ["test"]


def test_pipeline_export_model_snapshot_uses_tuning_model_and_skips_existing_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nhits_candidate_snapshot" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    class Config:
        tuning = {"model_name": "nhits_candidate"}
        backtest = {"benchmark_models": ["seasonal_naive"]}
        models = {
            "nhits_candidate": {"type": "nhits"},
            "seasonal_naive": {"type": "seasonal_naive"},
        }

    class Artifacts:
        def snapshot_manifest(self, snapshot_name: str) -> Path:
            assert snapshot_name == "nhits_candidate_snapshot"
            return manifest_path

    class WorkspaceStub:
        config = Config()
        artifacts = Artifacts()
        exports: list[tuple[str, str]]

        def __init__(self) -> None:
            self.exports = []

        def export_model_snapshot(self, *, model_name: str, snapshot_name: str) -> None:
            self.exports.append((model_name, snapshot_name))

    workspace = WorkspaceStub()

    _run_export_model_snapshot(workspace, "test")

    assert workspace.exports == []


def test_pipeline_export_model_snapshot_ignores_non_neural_model(tmp_path: Path) -> None:
    class Config:
        tuning = {}
        backtest = {"benchmark_models": ["seasonal_naive"]}
        models = {"seasonal_naive": {"type": "seasonal_naive"}}

    class Artifacts:
        def snapshot_manifest(self, snapshot_name: str) -> Path:
            return tmp_path / snapshot_name / "manifest.json"

    class WorkspaceStub:
        config = Config()
        artifacts = Artifacts()
        exports = 0

        def export_model_snapshot(self, *, model_name: str, snapshot_name: str) -> None:
            del model_name, snapshot_name
            self.exports += 1

    workspace = WorkspaceStub()

    _run_export_model_snapshot(workspace, "test")

    assert workspace.exports == 0


def test_resolve_mlp_unit_search_options_prefers_configured_values() -> None:
    tuning_cfg = {
        "search_space": {
            "mlp_units": [[256, 256], "384x384"],
        }
    }
    assert resolve_mlp_unit_search_options(tuning_cfg) == ["256x256", "384x384"]


def test_workspace_build_model_applies_named_nhits_best_params(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["models"]["nhits_candidate"] = {
        "type": "nhits",
        "h": 24,
        "input_size": 168,
        "max_steps": 10,
        "learning_rate": 0.001,
        "batch_size": 16,
        "dropout_prob_theta": 0.0,
        "scaler_type": "identity",
        "stack_types": ["identity", "identity", "identity"],
        "mlp_units": [[256, 256], [256, 256], [256, 256]],
        "loss_name": "huber_mqloss",
        "loss_delta": 0.75,
        "quantiles": [0.1, 0.5, 0.9],
    }
    workspace.artifacts.best_params("nhits_candidate").parent.mkdir(parents=True, exist_ok=True)
    workspace.artifacts.best_params("nhits_candidate").write_text(
        json.dumps({"input_size": 336, "mlp_units": "768x768"}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_raw_build_model(model_name: str, *, seed=None, disable_ensemble: bool = False):
        del seed, disable_ensemble
        captured["model_name"] = model_name
        captured["model_cfg"] = dict(workspace.config.models[model_name])
        return SnapshotStubModel()

    monkeypatch.setattr(workspace, "_raw_build_model", _fake_raw_build_model)

    workspace.build_model("nhits_candidate")

    assert captured["model_name"] == "nhits_candidate"
    assert captured["model_cfg"]["input_size"] == 336
    assert captured["model_cfg"]["mlp_units"] == [[768, 768], [768, 768], [768, 768]]


def test_workspace_main_flow_writes_predictions_metrics_and_report(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.prepare()

    feature_df = workspace.feature_frame()
    days = pd.Index(feature_df["ds"].dt.normalize().drop_duplicates().sort_values())
    split_boundaries = {
        "train_end": days[-5].isoformat(),
        "validation_start": days[-4].isoformat(),
        "validation_end": days[-3].isoformat(),
        "test_start": days[-2].isoformat(),
        "test_end": days[-1].isoformat(),
    }
    workspace.artifacts.split_boundaries().write_text(json.dumps(split_boundaries, indent=2), encoding="utf-8")

    workspace.backtest("test")
    workspace.evaluate("test")
    copied = workspace.export_report("test")
    report_dir = workspace.directories["report_dir"]
    for copied_path in copied:
        assert copied_path.exists()
    assert report_dir.exists()
    for child in report_dir.iterdir():
        child.unlink()
    report_dir.rmdir()
    rebuilt = workspace.export_report("test")

    assert workspace.artifacts.prediction("seasonal_naive", "test", 7).exists()
    assert workspace.artifacts.metrics("test").exists()
    assert workspace.artifacts.quantile_diagnostics("test").exists()
    assert workspace.artifacts.spike_score_diagnostics("test").exists()
    assert workspace.artifacts.scenario_diagnostics("test").exists()
    assert workspace.artifacts.dm("test").exists()
    assert workspace.artifacts.hourly_mae_plot("test").exists()
    assert workspace.artifacts.high_vol_week_plot("test").exists()
    assert workspace.artifacts.report_asset("test_metrics.csv") in copied
    assert workspace.artifacts.report_asset("test_quantile_diagnostics.csv") in copied
    assert workspace.artifacts.report_asset("test_spike_score_diagnostics.csv") in copied
    assert workspace.artifacts.report_asset("test_scenario_diagnostics.csv") in copied
    assert workspace.artifacts.report_asset("test_metrics.csv") in rebuilt
    assert workspace.artifacts.report_asset("test_quantile_diagnostics.csv") in rebuilt
    assert workspace.artifacts.report_asset("test_spike_score_diagnostics.csv") in rebuilt
    assert workspace.artifacts.report_asset("test_scenario_diagnostics.csv") in rebuilt


def test_workspace_evaluate_passes_normal_day_diagnostics_to_scorecard(monkeypatch) -> None:
    captured: dict[str, object] = {}
    bundle = object()
    metrics_df = pd.DataFrame([{"run": "nhits_test_seed7", "pinball": 1.0}])
    normal_day_df = pd.DataFrame(
        [
            {
                "run": "nhits_test_seed7",
                "segment": "actual_normal_day",
                "q50_wape": 0.25,
            }
        ]
    )
    relative_error_df = pd.DataFrame([{"run": "nhits_test_seed7", "slice_type": "all", "slice": "all"}])
    tail_regime_df = pd.DataFrame([{"run": "nhits_test_seed7", "regime": "all"}])

    class EvaluatorStub:
        def __init__(self, schema, artifacts) -> None:
            captured["schema"] = schema
            captured["artifacts"] = artifacts

        def load_runs(self, split: str):
            captured["split"] = split
            return bundle

        def compute_metrics(self, loaded_bundle):
            assert loaded_bundle is bundle
            return metrics_df

        def compute_quantile_diagnostics(self, loaded_bundle) -> None:
            assert loaded_bundle is bundle

        def compute_regime_metrics(self, loaded_bundle) -> None:
            assert loaded_bundle is bundle

        def compute_spike_score_diagnostics(self, loaded_bundle) -> None:
            assert loaded_bundle is bundle

        def compute_normal_day_diagnostics(self, loaded_bundle):
            assert loaded_bundle is bundle
            return normal_day_df

        def compute_relative_error(self, loaded_bundle):
            assert loaded_bundle is bundle
            return relative_error_df

        def compute_tail_regime_diagnostics(self, loaded_bundle):
            assert loaded_bundle is bundle
            return tail_regime_df

        def compute_experiment_scorecard(
            self,
            loaded_bundle,
            metrics,
            relative_error,
            tail_regime,
            normal_day=None,
        ) -> None:
            captured["scorecard_args"] = (loaded_bundle, metrics, relative_error, tail_regime, normal_day)

        def compute_scenario_diagnostics(self, loaded_bundle) -> None:
            assert loaded_bundle is bundle

        def compute_dm(self, loaded_bundle) -> None:
            assert loaded_bundle is bundle

        def render_plots(self, loaded_bundle, metrics, split: str) -> None:
            captured["render_args"] = (loaded_bundle, metrics, split)

    workspace = Workspace(config=object(), directories={}, artifacts=object(), models=object())
    monkeypatch.setattr("pjm_forecast.workspace.Evaluator", EvaluatorStub)
    monkeypatch.setattr(workspace, "schema", lambda: "schema")

    workspace.evaluate("test")

    scorecard_args = captured["scorecard_args"]
    assert scorecard_args[0] is bundle
    assert scorecard_args[1] is metrics_df
    assert scorecard_args[2] is relative_error_df
    assert scorecard_args[3] is tail_regime_df
    assert scorecard_args[4] is normal_day_df


def test_workspace_prepare_merges_optional_weather_features(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path, with_weather=True)
    workspace = Workspace.open(config_path)

    source_config = workspace.config.without_weather_feature_contracts()
    base_prepared = PreparedDataset.from_source(source_config, csv_path)
    weather_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(base_prepared.panel_df["ds"], utc=False),
            "weather_temp_mean": [5.0] * len(base_prepared.panel_df),
            "weather_cloud_cover_mean": [70.0] * len(base_prepared.panel_df),
        }
    )
    enriched_panel = base_prepared.panel_df.merge(weather_df, on="ds", how="left")
    enriched_prepared = PreparedDataset.from_panel_frame(workspace.config, enriched_panel)

    monkeypatch.setattr(
        "pjm_forecast.workspace.prepare_dataset",
        lambda config, raw_dir: PreparedDataResult(prepared=enriched_prepared, weather_df=weather_df),
    )

    workspace.prepare()

    panel_df = pd.read_parquet(workspace.artifacts.panel())
    feature_df = workspace.feature_frame()
    assert "weather_temp_mean" in panel_df.columns
    assert "weather_cloud_cover_mean" in panel_df.columns
    assert "weather_temp_mean_lag_24" in feature_df.columns
    assert "weather_cloud_cover_mean_lag_24" in feature_df.columns
    assert workspace.artifacts.weather_features().exists()


def test_artifact_store_prediction_runs_returns_metadata_without_filename_contract(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    prediction_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=24, freq="h"),
            "y": [1.0] * 24,
            "y_pred": [1.5] * 24,
            "model": ["nbeatsx"] * 24,
            "split": ["test"] * 24,
            "seed": [7] * 24,
            "quantile": [pd.NA] * 24,
            "metadata": ["{}"] * 24,
        }
    )
    custom_path = workspace.directories["prediction_dir"] / "custom_name_without_contract.parquet"
    prediction_df.to_parquet(custom_path, index=False)

    runs = workspace.artifacts.prediction_runs("test")

    assert len(runs) == 1
    assert runs[0].name == "custom_name_without_contract"
    assert runs[0].path == custom_path
    assert runs[0].model == "nbeatsx"
    assert runs[0].seed == 7
    assert runs[0].variant == "custom_name_without_contract"


def test_model_store_resolves_named_snapshot_path(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    captured: dict[str, Path] = {}

    def _fake_load(path: Path):
        captured["path"] = path
        return SnapshotStubModel()

    monkeypatch.setattr("pjm_forecast.workspace.load_model_snapshot", _fake_load)
    workspace.models.load_snapshot("nbeatsx_snapshot")

    assert captured["path"] == workspace.artifacts.model_snapshot("nbeatsx_snapshot")


def test_workspace_export_nbeatsx_snapshot_fits_and_saves_model(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.prepare()

    stub_model = SnapshotStubModel()
    monkeypatch.setattr(workspace, "build_model", lambda *args, **kwargs: stub_model)

    output_dir = workspace.export_nbeatsx_snapshot()

    assert output_dir == workspace.artifacts.model_snapshot("nbeatsx_snapshot")
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "payload" / "metadata.json").exists()
    assert stub_model.fit_rows == workspace.config.backtest["rolling_window_days"] * 24


def test_workspace_predict_model_snapshot_writes_prediction_file(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)

    history_path = tmp_path / "history.parquet"
    future_path = tmp_path / "future.parquet"
    output_path = tmp_path / "predictions" / "snapshot.parquet"
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": list(range(48)),
        }
    )
    future_df = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    history_df.to_parquet(history_path, index=False)
    future_df.to_parquet(future_path, index=False)

    stub_model = SnapshotStubModel()
    monkeypatch.setattr(type(workspace.models), "load_snapshot", lambda self, snapshot_name_or_path: stub_model)

    written_path = workspace.predict_model_snapshot(
        snapshot_name_or_path=tmp_path / "snapshot",
        history_path=history_path,
        future_path=future_path,
        output_path=output_path,
    )

    written_df = pd.read_parquet(written_path)
    assert written_path == output_path
    assert list(written_df.columns) == ["ds", "quantile", "y_pred"]
    assert written_df["quantile"].isna().all()
    assert set(written_df["y_pred"]) == {42.0}


def test_workspace_retrieve_nbeatsx_only_orchestrates_runner(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.prepare()

    captured: dict[str, object] = {"tune": [], "apply": []}

    class StubRunner:
        def __init__(self, config, prepared_dataset, artifacts, *, prediction_loader=None) -> None:
            captured["config"] = config
            captured["prepared_dataset"] = prepared_dataset
            captured["artifacts"] = artifacts
            captured["prediction_loader"] = prediction_loader

        def tune(self, base_model: str = "nbeatsx", split: str = "validation"):
            captured["tune"].append((base_model, split))
            return None

        def apply(self, base_model: str = "nbeatsx", split: str = "test"):
            captured["apply"].append((base_model, split))
            return Path("ignored.parquet")

    monkeypatch.setattr("pjm_forecast.workspace.RetrievalRunner", StubRunner)

    workspace.retrieve_nbeatsx("test")

    assert captured["config"] is workspace.config
    assert captured["artifacts"] is workspace.artifacts
    assert callable(captured["prediction_loader"])
    assert captured["tune"] == [(workspace.config.retrieval_base_model_name, "validation")]
    assert captured["apply"] == [
        (workspace.config.retrieval_base_model_name, "validation"),
        (workspace.config.retrieval_base_model_name, "test"),
    ]
