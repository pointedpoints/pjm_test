#!/usr/bin/env python3
"""Build validation quantile ensembles for the current PJM artifacts.

Outputs:
- artifacts_current/predictions/ensemble_family_balanced_validation_seed7.parquet
- artifacts_current/predictions/ensemble_equal_validation_seed7.parquet

Layer 1:
- inverse-pinball weighted NHITS sub-ensemble over nhits_168/nhits_336/nhits_720
- equal-weight NHITS baseline over the same three variants

Layer 2:
- family-balanced ensemble: lightgbm_q=0.40, xgboost_q=0.30, nhits_sub=0.30
- equal ensemble baseline: lightgbm_q=xgboost_q=nhits_equal_sub=1/3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pjm_forecast.prediction_contract import (  # noqa: E402
    enforce_monotonic_quantiles,
    quantile_values,
    validate_canonical_prediction_frame,
)

ROWS_PER_SPLIT = {"validation": 65_520, "test": 131_040}
SEED = 7
SPLIT = "validation"  # mutable via --split
NHITS_MODELS = ["nhits_168", "nhits_336", "nhits_720"]
FAMILY_MODELS = ["lightgbm_q", "xgboost_q"]
REFERENCE_ONLY_MODELS = ["seasonal_naive", "nhits_tail_grid_weighted_main", "nbeatsx"]
FAMILY_BALANCED_WEIGHTS = {"lightgbm_q": 0.40, "xgboost_q": 0.30, "nhits_sub": 0.30}
EQUAL_FAMILY_WEIGHTS = {"lightgbm_q": 1.0 / 3.0, "xgboost_q": 1.0 / 3.0, "nhits_sub": 1.0 / 3.0}
OUTPUT_COLUMNS = ["ds", "y", "quantile", "y_pred", "model", "split", "seed", "metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quantile ensemble predictions.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts_current" / "predictions",
        help="Directory containing prediction parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which split to ensemble (default: validation).",
    )
    return parser.parse_args()


def pinball_loss(frame: pd.DataFrame) -> float:
    y_true = frame["y"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    q = frame["quantile"].astype(float).to_numpy()
    errors = y_true - y_pred
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    mean_loss = float(np.mean(loss))
    if not np.isfinite(mean_loss) or mean_loss <= 0.0:
        raise ValueError(f"Invalid pinball loss for {frame['model'].iloc[0]}: {mean_loss}")
    return mean_loss


def normalize_prediction_frame(path: Path, *, require_quantiles: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    frame = frame.loc[:, OUTPUT_COLUMNS].copy()
    frame["ds"] = pd.to_datetime(frame["ds"], utc=False)
    frame["y"] = frame["y"].astype(float)
    frame["y_pred"] = frame["y_pred"].astype(float)
    frame["model"] = frame["model"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["seed"] = frame["seed"].astype(int)

    if require_quantiles:
        frame["quantile"] = frame["quantile"].astype(float)
        if frame["quantile"].isna().any():
            raise ValueError(f"{path.name} contains missing quantile values.")
        frame = frame.sort_values(["ds", "quantile"]).reset_index(drop=True)
        if len(frame) != ROWS_PER_SPLIT[SPLIT]:
            raise ValueError(f"{path.name} expected {ROWS_PER_SPLIT[SPLIT]} rows, found {len(frame)}.")
        validate_canonical_prediction_frame(frame, require_metadata=True)
        if frame[["ds", "y", "quantile", "y_pred"]].isna().any().any():
            raise ValueError(f"{path.name} contains NaN in ensemble-critical columns.")
    else:
        frame = frame.sort_values("ds").reset_index(drop=True)

    return frame


def load_predictions(predictions_dir: Path) -> dict[str, pd.DataFrame]:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {predictions_dir}")

    loaded: dict[str, pd.DataFrame] = {}
    required_quantile_models = set(NHITS_MODELS + FAMILY_MODELS)
    for path in sorted(predictions_dir.glob(f"*_{SPLIT}_seed{SEED}.parquet")):
        stem_suffix = f"_{SPLIT}_seed{SEED}"
        model_name = path.stem[: -len(stem_suffix)]
        if model_name.startswith("ensemble_"):
            continue
        # Only ensemble inputs are required to share the full 65,520-row quantile grid.
        # seasonal_naive is point-median reference only; nbeatsx / tail-grid variants are
        # informative side artifacts and must not block this ensemble build.
        loaded[model_name] = normalize_prediction_frame(path, require_quantiles=model_name in required_quantile_models)

    required = NHITS_MODELS + FAMILY_MODELS
    missing = [model for model in required if model not in loaded]
    if missing:
        raise FileNotFoundError(f"Missing required ensemble prediction files for models: {missing}")

    return loaded


def assert_same_grid(frames: Mapping[str, pd.DataFrame]) -> list[float]:
    base_name = next(iter(frames))
    base = frames[base_name].loc[:, ["ds", "y", "quantile"]].reset_index(drop=True)
    expected_quantiles = quantile_values(frames[base_name])
    for name, frame in frames.items():
        current = frame.loc[:, ["ds", "y", "quantile"]].reset_index(drop=True)
        if not current.equals(base):
            raise ValueError(f"Prediction grid for {name} does not match {base_name}.")
        current_quantiles = quantile_values(frame)
        if current_quantiles != expected_quantiles:
            raise ValueError(
                f"Quantile grid mismatch for {name}: expected={expected_quantiles}, actual={current_quantiles}"
            )
    return expected_quantiles


def weighted_average(
    frames: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    model_name: str,
    metadata: dict[str, object],
) -> pd.DataFrame:
    missing = [name for name in weights if name not in frames]
    if missing:
        raise ValueError(f"Cannot build {model_name}; missing input frames: {missing}")

    selected = {name: frames[name] for name in weights}
    assert_same_grid(selected)
    total_weight = float(sum(weights.values()))
    if not np.isclose(total_weight, 1.0):
        weights = {name: float(weight) / total_weight for name, weight in weights.items()}

    first = next(iter(selected.values()))
    result = first.loc[:, ["ds", "y", "quantile"]].copy().reset_index(drop=True)
    y_pred = np.zeros(len(result), dtype=float)
    normalized_weights = {name: float(weight) for name, weight in weights.items()}
    for name, weight in normalized_weights.items():
        y_pred += weight * selected[name]["y_pred"].astype(float).to_numpy()

    result["y_pred"] = y_pred
    result["model"] = model_name
    result["split"] = SPLIT
    result["seed"] = SEED
    result["metadata"] = json.dumps({**metadata, "weights": normalized_weights}, sort_keys=True)
    result = result.loc[:, OUTPUT_COLUMNS].sort_values(["ds", "quantile"]).reset_index(drop=True)
    result = enforce_monotonic_quantiles(result)
    result["model"] = model_name
    result["metadata"] = json.dumps({**metadata, "weights": normalized_weights}, sort_keys=True)
    return result.loc[:, OUTPUT_COLUMNS]


def validate_ensemble(frame: pd.DataFrame, *, expected_quantiles: list[float], output_path: Path) -> None:
    if len(frame) != ROWS_PER_SPLIT[SPLIT]:
        raise ValueError(f"{output_path.name} expected {ROWS_PER_SPLIT[SPLIT]} rows, found {len(frame)}.")
    validate_canonical_prediction_frame(frame, require_metadata=True, expected_quantiles=expected_quantiles)
    if frame[OUTPUT_COLUMNS].isna().any().any():
        counts = frame[OUTPUT_COLUMNS].isna().sum()
        raise ValueError(f"{output_path.name} contains NaN values: {counts[counts > 0].to_dict()}")


def main() -> None:
    global SPLIT
    args = parse_args()
    SPLIT = args.split
    predictions_dir = args.predictions_dir.resolve()
    frames = load_predictions(predictions_dir)

    ensemble_inputs = {name: frames[name] for name in NHITS_MODELS + FAMILY_MODELS}
    expected_quantiles = assert_same_grid(ensemble_inputs)

    nhits_losses = {name: pinball_loss(frames[name]) for name in NHITS_MODELS}
    inverse_sum = sum(1.0 / loss for loss in nhits_losses.values())
    nhits_inverse_weights = {name: (1.0 / loss) / inverse_sum for name, loss in nhits_losses.items()}
    nhits_equal_weights = {name: 1.0 / len(NHITS_MODELS) for name in NHITS_MODELS}

    nhits_inverse_sub = weighted_average(
        {name: frames[name] for name in NHITS_MODELS},
        nhits_inverse_weights,
        model_name="nhits_inverse_pinball_subensemble",
        metadata={"layer": 1, "method": "inverse_pinball", "pinball_losses": nhits_losses},
    )
    nhits_equal_sub = weighted_average(
        {name: frames[name] for name in NHITS_MODELS},
        nhits_equal_weights,
        model_name="nhits_equal_subensemble",
        metadata={"layer": 1, "method": "equal_weight_nhits_baseline"},
    )

    family_frames = {
        "lightgbm_q": frames["lightgbm_q"],
        "xgboost_q": frames["xgboost_q"],
        "nhits_sub": nhits_inverse_sub,
    }
    equal_family_frames = {
        "lightgbm_q": frames["lightgbm_q"],
        "xgboost_q": frames["xgboost_q"],
        "nhits_sub": nhits_equal_sub,
    }

    family_balanced = weighted_average(
        family_frames,
        FAMILY_BALANCED_WEIGHTS,
        model_name="ensemble_family_balanced",
        metadata={
            "layer": 2,
            "method": "fixed_family_weights_with_inverse_pinball_nhits_subensemble",
            "nhits_pinball_losses": nhits_losses,
            "nhits_weights": nhits_inverse_weights,
            "reference_only_models_loaded": [name for name in REFERENCE_ONLY_MODELS if name in frames],
        },
    )
    equal = weighted_average(
        equal_family_frames,
        EQUAL_FAMILY_WEIGHTS,
        model_name="ensemble_equal",
        metadata={
            "layer": 2,
            "method": "equal_family_weights_with_equal_nhits_subensemble",
            "nhits_weights": nhits_equal_weights,
            "reference_only_models_loaded": [name for name in REFERENCE_ONLY_MODELS if name in frames],
        },
    )

    outputs = {
        f"ensemble_family_balanced_{SPLIT}_seed{SEED}.parquet": family_balanced,
        f"ensemble_equal_{SPLIT}_seed{SEED}.parquet": equal,
    }
    for filename, frame in outputs.items():
        output_path = predictions_dir / filename
        validate_ensemble(frame, expected_quantiles=expected_quantiles, output_path=output_path)
        frame.to_parquet(output_path, index=False)
        reloaded = normalize_prediction_frame(output_path, require_quantiles=True)
        validate_ensemble(reloaded, expected_quantiles=expected_quantiles, output_path=output_path)

    loaded_files = sorted(path.name for path in predictions_dir.glob(f"*_{SPLIT}_seed{SEED}.parquet"))
    print(f"Loaded {SPLIT} files ({len(loaded_files)}): {loaded_files}")
    print(f"Ensemble inputs: {sorted(ensemble_inputs)}")
    print(f"Reference-only loaded, excluded from ensemble: {[name for name in REFERENCE_ONLY_MODELS if name in frames]}")
    print(f"NHITS pinball losses: {nhits_losses}")
    print(f"NHITS inverse-pinball weights: {nhits_inverse_weights}")
    for filename in outputs:
        print(f"Wrote and verified: {predictions_dir / filename} rows={ROWS_PER_SPLIT[SPLIT]} quantiles={expected_quantiles}")


if __name__ == "__main__":
    main()
