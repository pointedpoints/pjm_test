from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StackingRows:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    base_model_names: tuple[str, ...]


def build_stacking_training_rows(
    *,
    feature_df: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    schema,
    base_model_names: list[str] | tuple[str, ...],
) -> StackingRows:
    schema.validate_feature_frame(feature_df)
    ordered_model_names = tuple(base_model_names)
    if not ordered_model_names:
        raise ValueError("Stacking requires at least one base model.")

    aligned = _align_prediction_frames(prediction_frames, ordered_model_names, schema)
    feature_columns = (
        schema.future_exog_columns()
        + schema.calendar_columns()
        + schema.price_lag_columns()
        + schema.load_lag_columns()
    )
    feature_columns = tuple(dict.fromkeys(feature_columns))

    feature_slice = feature_df.loc[:, ["ds", *feature_columns]].copy()
    merged = aligned.merge(feature_slice, on="ds", how="left", validate="one_to_one")
    if merged[list(feature_columns)].isna().any().any():
        missing_columns = [column for column in feature_columns if merged[column].isna().any()]
        raise ValueError(f"Stacking rows contain missing feature values: {missing_columns}")

    prediction_feature_columns = tuple(f"pred_{model_name}" for model_name in ordered_model_names)
    merged["pred_ensemble_mean"] = merged.loc[:, list(prediction_feature_columns)].mean(axis=1)
    merged["pred_ensemble_std"] = merged.loc[:, list(prediction_feature_columns)].std(axis=1, ddof=0)
    merged["pred_ensemble_min"] = merged.loc[:, list(prediction_feature_columns)].min(axis=1)
    merged["pred_ensemble_max"] = merged.loc[:, list(prediction_feature_columns)].max(axis=1)
    merged["pred_ensemble_spread"] = merged["pred_ensemble_max"] - merged["pred_ensemble_min"]

    gap_columns: list[str] = []
    for model_name in ordered_model_names:
        column_name = f"pred_gap_to_mean_{model_name}"
        merged[column_name] = merged[f"pred_{model_name}"] - merged["pred_ensemble_mean"]
        gap_columns.append(column_name)

    model_feature_columns = (
        *prediction_feature_columns,
        "pred_ensemble_mean",
        "pred_ensemble_std",
        "pred_ensemble_min",
        "pred_ensemble_max",
        "pred_ensemble_spread",
        *gap_columns,
        *feature_columns,
    )
    return StackingRows(
        frame=merged,
        feature_columns=tuple(model_feature_columns),
        base_model_names=ordered_model_names,
    )


def _align_prediction_frames(
    prediction_frames: dict[str, pd.DataFrame],
    ordered_model_names: tuple[str, ...],
    schema,
) -> pd.DataFrame:
    reference_model = ordered_model_names[0]
    missing_models = [model_name for model_name in ordered_model_names if model_name not in prediction_frames]
    if missing_models:
        raise ValueError(f"Stacking prediction frames are missing models: {missing_models}")

    reference = prediction_frames[reference_model].copy().sort_values("ds").reset_index(drop=True)
    schema.validate_prediction_frame(reference, require_metadata=False)
    aligned = reference.loc[:, ["ds", "y"]].copy()
    aligned[f"pred_{reference_model}"] = reference["y_pred"].to_numpy(dtype=float)

    for model_name in ordered_model_names[1:]:
        candidate = prediction_frames[model_name].copy().sort_values("ds").reset_index(drop=True)
        schema.validate_prediction_frame(candidate, require_metadata=False)
        if not reference["ds"].equals(candidate["ds"]):
            raise ValueError(f"Stacking requires aligned prediction timestamps for model={model_name!r}.")
        if not np.allclose(
            reference["y"].to_numpy(dtype=float),
            candidate["y"].to_numpy(dtype=float),
            equal_nan=False,
        ):
            raise ValueError(f"Stacking requires aligned prediction targets for model={model_name!r}.")
        aligned[f"pred_{model_name}"] = candidate["y_pred"].to_numpy(dtype=float)

    return aligned
