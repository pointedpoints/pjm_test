from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pjm_forecast.data.ingress import PreparedDataResult
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.pipeline import STAGE_ORDER
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
    assert workspace.artifacts.scenario_diagnostics("test") == (
        tmp_path / "run" / "artifacts" / "metrics" / "test_scenario_diagnostics.csv"
    ).resolve()
    assert workspace.artifacts.snapshot_manifest("nbeatsx_snapshot") == (
        tmp_path / "run" / "artifacts" / "models" / "nbeatsx_snapshot" / "manifest.json"
    ).resolve()


def test_pipeline_stage_order_excludes_retrieval() -> None:
    assert STAGE_ORDER == [
        "prepare_data",
        "tune_nbeatsx",
        "backtest_all_models",
        "evaluate_and_plot",
        "export_report_assets",
    ]


def test_resolve_mlp_unit_search_options_prefers_configured_values() -> None:
    tuning_cfg = {
        "search_space": {
            "mlp_units": [[256, 256], "384x384"],
        }
    }
    assert resolve_mlp_unit_search_options(tuning_cfg) == ["256x256", "384x384"]


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
    assert workspace.artifacts.scenario_diagnostics("test").exists()
    assert workspace.artifacts.dm("test").exists()
    assert workspace.artifacts.hourly_mae_plot("test").exists()
    assert workspace.artifacts.high_vol_week_plot("test").exists()
    assert workspace.artifacts.report_asset("test_metrics.csv") in copied
    assert workspace.artifacts.report_asset("test_quantile_diagnostics.csv") in copied
    assert workspace.artifacts.report_asset("test_scenario_diagnostics.csv") in copied
    assert workspace.artifacts.report_asset("test_metrics.csv") in rebuilt
    assert workspace.artifacts.report_asset("test_quantile_diagnostics.csv") in rebuilt
    assert workspace.artifacts.report_asset("test_scenario_diagnostics.csv") in rebuilt


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


def test_workspace_predict_nbeatsx_snapshot_writes_prediction_file(tmp_path: Path, monkeypatch) -> None:
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

    written_path = workspace.predict_nbeatsx_snapshot(
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
