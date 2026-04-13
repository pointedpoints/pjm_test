from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models.base import ForecastModel
from .models.epftoolbox_wrappers import DNNModel, LEARModel
from .models.nbeatsx import NBEATSxModel
from .models.seasonal_naive import SeasonalNaiveModel


SNAPSHOT_MANIFEST = "manifest.json"
SNAPSHOT_PAYLOAD = "payload"
SNAPSHOT_FORMAT_VERSION = 1

MODEL_LOADERS: dict[str, type[ForecastModel]] = {
    "seasonal_naive": SeasonalNaiveModel,
    "lear": LEARModel,
    "dnn": DNNModel,
    "nbeatsx": NBEATSxModel,
}


def snapshot_manifest_path(snapshot_path: Path) -> Path:
    return snapshot_path / SNAPSHOT_MANIFEST


def snapshot_payload_path(snapshot_path: Path, payload_name: str = SNAPSHOT_PAYLOAD) -> Path:
    return snapshot_path / payload_name


def validate_model_prediction_output(
    prediction_df: pd.DataFrame,
    *,
    future_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    required_columns = ["ds", "y_pred"]
    missing_columns = [column for column in required_columns if column not in prediction_df.columns]
    if missing_columns:
        raise ValueError(f"Model '{model_name}' prediction output is missing required columns: {missing_columns}")

    normalized = prediction_df.loc[:, required_columns].copy()
    normalized["ds"] = pd.to_datetime(normalized["ds"], utc=False)
    if normalized["ds"].duplicated().any():
        raise ValueError(f"Model '{model_name}' prediction output contains duplicate ds timestamps.")
    if normalized["y_pred"].isna().any():
        raise ValueError(f"Model '{model_name}' prediction output contains missing y_pred values.")

    expected_ds = future_df["ds"].reset_index(drop=True)
    actual_ds = normalized["ds"].reset_index(drop=True)
    if len(actual_ds) != len(expected_ds) or not actual_ds.equals(expected_ds):
        raise ValueError(
            f"Model '{model_name}' prediction timestamps do not align and must exactly match the requested future horizon."
        )
    return normalized


def save_model_snapshot(model: ForecastModel, *, model_name: str, snapshot_path: Path) -> Path:
    return save_model_snapshot_bundle(model, model_name=model_name, snapshot_path=snapshot_path)


def save_model_snapshot_bundle(
    model: ForecastModel,
    *,
    model_name: str,
    snapshot_path: Path,
    history_df: pd.DataFrame | None = None,
    prediction_horizon: int | None = None,
    prediction_freq: str | None = None,
) -> Path:
    if not getattr(model, "supports_fitted_snapshot", False):
        raise ValueError(f"Model '{model_name}' does not support fitted snapshot export.")

    snapshot_path.mkdir(parents=True, exist_ok=True)
    payload_path = snapshot_payload_path(snapshot_path)
    model.save(payload_path)
    manifest = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "model_name": model_name,
        "payload_path": SNAPSHOT_PAYLOAD,
    }
    if history_df is not None and not history_df.empty:
        history_ds = pd.to_datetime(history_df["ds"], utc=False)
        manifest.update(
            {
                "history_rows": len(history_df),
                "history_start": pd.Timestamp(history_ds.min()).isoformat(),
                "history_end": pd.Timestamp(history_ds.max()).isoformat(),
            }
        )
    if prediction_horizon is not None:
        manifest["prediction_horizon"] = int(prediction_horizon)
    if prediction_freq is not None:
        manifest["prediction_freq"] = str(prediction_freq)
    (snapshot_path / SNAPSHOT_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return snapshot_path


def load_snapshot_manifest(snapshot_path: Path) -> dict[str, object]:
    manifest_path = snapshot_manifest_path(snapshot_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(f"Unsupported snapshot format_version: {manifest.get('format_version')!r}")
    if "model_name" not in manifest:
        raise ValueError("Snapshot manifest is missing model_name.")
    if "payload_path" not in manifest:
        raise ValueError("Snapshot manifest is missing payload_path.")
    return manifest


def load_model_snapshot(snapshot_path: Path) -> ForecastModel:
    manifest_path = snapshot_manifest_path(snapshot_path)
    if manifest_path.exists():
        manifest = load_snapshot_manifest(snapshot_path)
        model_name = str(manifest["model_name"])
        loader = MODEL_LOADERS.get(model_name)
        if loader is None:
            raise ValueError(f"Unsupported snapshot model_name: {model_name!r}")
        payload_path = snapshot_payload_path(snapshot_path, payload_name=str(manifest["payload_path"]))
        return loader.load(payload_path)

    legacy_nbeatsx_metadata = snapshot_path / "metadata.json"
    if legacy_nbeatsx_metadata.exists():
        return NBEATSxModel.load(snapshot_path)

    raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")


def predict_with_snapshot(snapshot_path: Path, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    manifest = load_snapshot_manifest(snapshot_path) if snapshot_manifest_path(snapshot_path).exists() else {"model_name": "nbeatsx"}
    model_name = str(manifest["model_name"])
    model = load_model_snapshot(snapshot_path)
    predictions = model.predict(history_df=history_df, future_df=future_df)
    return validate_model_prediction_output(predictions, future_df=future_df, model_name=model_name)


def write_snapshot_predictions(
    snapshot_path: Path,
    *,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    predictions = predict_with_snapshot(snapshot_path, history_df=history_df, future_df=future_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    return output_path
