from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.model_io import (
    MODEL_LOADERS,
    load_model_snapshot,
    load_snapshot_manifest,
    predict_with_snapshot,
    save_model_snapshot_bundle,
)
from pjm_forecast.models.base import ForecastModel


class StubSnapshotModel(ForecastModel):
    name = "stub"
    supports_fitted_snapshot = True

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = float(bias)

    def fit(self, train_df: pd.DataFrame) -> None:
        self.bias = float(train_df["y"].iloc[-1])

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": np.full(len(future_df), self.bias)})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"bias": self.bias}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StubSnapshotModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(bias=float(payload["bias"]))


class MisalignedSnapshotModel(StubSnapshotModel):
    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        shifted_ds = future_df["ds"] + pd.Timedelta(hours=1)
        return pd.DataFrame({"ds": shifted_ds.to_numpy(), "y_pred": np.full(len(future_df), self.bias)})


def test_save_model_snapshot_bundle_writes_manifest_and_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(MODEL_LOADERS, "stub", StubSnapshotModel)
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": np.arange(48, dtype=float),
        }
    )
    future_df = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    model = StubSnapshotModel()
    model.fit(history_df)
    snapshot_dir = tmp_path / "snapshot"

    save_model_snapshot_bundle(
        model,
        model_name="stub",
        snapshot_path=snapshot_dir,
        history_df=history_df,
        prediction_horizon=24,
        prediction_freq="h",
    )

    manifest = load_snapshot_manifest(snapshot_dir)
    assert manifest["model_name"] == "stub"
    assert manifest["payload_path"] == "payload"
    assert manifest["history_rows"] == 48
    assert manifest["prediction_horizon"] == 24
    assert manifest["prediction_freq"] == "h"
    assert (snapshot_dir / "payload").exists()

    loaded_model = load_model_snapshot(snapshot_dir)
    predictions = predict_with_snapshot(snapshot_dir, history_df=history_df, future_df=future_df)
    assert isinstance(loaded_model, StubSnapshotModel)
    assert predictions["y_pred"].eq(float(history_df["y"].iloc[-1])).all()


def test_predict_with_snapshot_rejects_misaligned_prediction_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(MODEL_LOADERS, "misaligned_stub", MisalignedSnapshotModel)
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": np.arange(48, dtype=float),
        }
    )
    future_df = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    model = MisalignedSnapshotModel()
    model.fit(history_df)
    snapshot_dir = tmp_path / "snapshot"

    save_model_snapshot_bundle(model, model_name="misaligned_stub", snapshot_path=snapshot_dir)

    with pytest.raises(ValueError, match="must exactly match"):
        predict_with_snapshot(snapshot_dir, history_df=history_df, future_df=future_df)
