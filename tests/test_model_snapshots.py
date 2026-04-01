from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pjm_forecast.model_io import MODEL_LOADERS, load_model_snapshot, save_model_snapshot
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.ops import predict_model_snapshot
from pjm_forecast.workspace import Workspace


class DummySnapshotModel(ForecastModel):
    name = "dummy"
    supports_fitted_snapshot = True

    def __init__(self, fitted_value: float | None = None) -> None:
        self.fitted_value = fitted_value

    def fit(self, train_df: pd.DataFrame) -> None:
        self.fitted_value = float(train_df["y"].iloc[-1])

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        if self.fitted_value is None:
            raise RuntimeError("DummySnapshotModel must be fit before predict.")
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": [self.fitted_value] * len(future_df)})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"fitted_value": self.fitted_value}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DummySnapshotModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(fitted_value=payload["fitted_value"])


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


def _write_temp_config(tmp_path: Path, csv_path: Path) -> Path:
    base_config = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    base_config["project"]["root_override"] = str(tmp_path / "run")
    base_config["dataset"]["local_csv_path"] = str(csv_path)
    base_config["backtest"]["years_test"] = 1
    base_config["backtest"]["validation_days"] = 28
    base_config["backtest"]["benchmark_models"] = ["seasonal_naive"]
    base_config["backtest"]["rolling_window_days"] = 8
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return config_path


def test_model_snapshot_manifest_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(MODEL_LOADERS, "dummy", DummySnapshotModel)
    model = DummySnapshotModel(fitted_value=42.0)
    snapshot_dir = save_model_snapshot(model, model_name="dummy", snapshot_path=tmp_path / "dummy_snapshot")

    loaded = load_model_snapshot(snapshot_dir)

    assert isinstance(loaded, DummySnapshotModel)
    assert loaded.fitted_value == 42.0
    assert (snapshot_dir / "manifest.json").exists()


def test_predict_model_snapshot_uses_manifest_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(MODEL_LOADERS, "dummy", DummySnapshotModel)
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": list(range(48)),
        }
    )
    future_df = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    model = DummySnapshotModel()
    model.fit(history_df)

    snapshot_dir = save_model_snapshot(model, model_name="dummy", snapshot_path=tmp_path / "dummy_snapshot")
    history_path = tmp_path / "history.parquet"
    future_path = tmp_path / "future.parquet"
    output_path = tmp_path / "predictions.parquet"
    history_df.to_parquet(history_path, index=False)
    future_df.to_parquet(future_path, index=False)

    predict_model_snapshot(str(snapshot_dir), str(history_path), str(future_path), str(output_path))
    prediction_df = pd.read_parquet(output_path)

    assert prediction_df["y_pred"].nunique() == 1
    assert prediction_df["y_pred"].iloc[0] == 47.0


def test_workspace_rejects_snapshot_export_for_model_without_fitted_snapshot(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.prepare()

    with pytest.raises(ValueError, match="does not support fitted snapshot"):
        workspace.export_model_snapshot(model_name="seasonal_naive", snapshot_name="seasonal_naive_snapshot")
