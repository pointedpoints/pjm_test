from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model_io import write_snapshot_predictions
from .workspace import Workspace


def export_model_snapshot(config_path: str, model_name: str = "nbeatsx", snapshot_name: str | None = None) -> Path:
    workspace = Workspace.open(config_path)
    return workspace.export_model_snapshot(model_name=model_name, snapshot_name=snapshot_name)


def export_configured_model_snapshot(
    config_path: str,
    model_name: str | None = None,
    snapshot_name: str | None = None,
) -> Path:
    workspace = Workspace.open(config_path)
    raw_cfg = getattr(workspace.config, "raw", None)
    if raw_cfg is not None:
        tuning_cfg = raw_cfg.get("tuning", {})
    else:
        try:
            tuning_cfg = workspace.config.tuning
        except KeyError:
            tuning_cfg = {}
    resolved_model_name = model_name or tuning_cfg.get("model_name") or workspace.config.backtest["benchmark_models"][0]
    resolved_snapshot_name = snapshot_name or f"{resolved_model_name}_snapshot"
    return workspace.export_model_snapshot(model_name=resolved_model_name, snapshot_name=resolved_snapshot_name)


def export_nbeatsx_snapshot(config_path: str) -> Path:
    return export_model_snapshot(config_path, model_name="nbeatsx", snapshot_name="nbeatsx_snapshot")


def predict_model_snapshot(snapshot_path: str, history_path: str, future_path: str, output_path: str) -> None:
    history_df = pd.read_parquet(history_path)
    future_df = pd.read_parquet(future_path)
    write_snapshot_predictions(
        Path(snapshot_path),
        history_df=history_df,
        future_df=future_df,
        output_path=Path(output_path),
    )
