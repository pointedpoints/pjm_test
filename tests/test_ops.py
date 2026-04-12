from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast import ops


def test_export_nbeatsx_snapshot_delegates_to_workspace(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected = tmp_path / "artifacts" / "models" / "nbeatsx_snapshot"

    class StubWorkspace:
        def export_model_snapshot(self, *, model_name: str, snapshot_name: str | None = None) -> Path:
            captured["called"] = True
            captured["model_name"] = model_name
            captured["snapshot_name"] = snapshot_name
            return expected

    def _fake_open(config_path: str) -> StubWorkspace:
        captured["config_path"] = config_path
        return StubWorkspace()

    monkeypatch.setattr("pjm_forecast.ops.Workspace.open", _fake_open)

    output = ops.export_nbeatsx_snapshot("configs/pjm_day_ahead_v1.yaml")

    assert captured["config_path"] == "configs/pjm_day_ahead_v1.yaml"
    assert captured["called"] is True
    assert captured["model_name"] == "nbeatsx"
    assert captured["snapshot_name"] == "nbeatsx_snapshot"
    assert output == expected


def test_predict_model_snapshot_writes_predictions_from_manifest_loader(monkeypatch, tmp_path: Path) -> None:
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": list(range(48)),
        }
    )
    future_df = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    history_path = tmp_path / "history.parquet"
    future_path = tmp_path / "future.parquet"
    output_path = tmp_path / "predictions.parquet"
    history_df.to_parquet(history_path, index=False)
    future_df.to_parquet(future_path, index=False)

    captured: dict[str, object] = {}

    def _fake_write_snapshot_predictions(
        snapshot_path: Path,
        *,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        captured["snapshot_path"] = snapshot_path
        captured["history_rows"] = len(history_df)
        captured["future_rows"] = len(future_df)
        result = pd.DataFrame({"ds": future_df["ds"], "y_pred": [42.0] * len(future_df)})
        result.to_parquet(output_path, index=False)
        return output_path

    monkeypatch.setattr("pjm_forecast.ops.write_snapshot_predictions", _fake_write_snapshot_predictions)

    ops.predict_model_snapshot(
        snapshot_path=str(tmp_path / "snapshot"),
        history_path=str(history_path),
        future_path=str(future_path),
        output_path=str(output_path),
    )

    prediction_df = pd.read_parquet(output_path)
    assert captured["snapshot_path"] == tmp_path / "snapshot"
    assert captured["history_rows"] == 48
    assert captured["future_rows"] == 24
    assert prediction_df["y_pred"].eq(42.0).all()
