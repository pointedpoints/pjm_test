from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import (
    enforce_monotonic_quantiles,
    is_quantile_prediction_frame,
    quantile_values,
)


DEFAULT_QUANTILE_CALIBRATION_METHOD = "cqr"
SUPPORTED_QUANTILE_CALIBRATION_METHODS = {"cqr", "cqr_asymmetric"}
SUPPORTED_QUANTILE_CALIBRATION_GROUPS = {None, "hour", "hour_x_regime"}
DEFAULT_REGIME_SCORE_COLUMN = "spike_score"
DEFAULT_REGIME_THRESHOLD = 0.67
ALL_GROUP = "__all__"


@dataclass(frozen=True)
class QuantilePairCalibration:
    lower_adjustment: float
    upper_adjustment: float


@dataclass(frozen=True)
class ConformalQuantileCalibration:
    method: str
    group_by: str | None
    adjustments: dict[tuple[object, float, float], QuantilePairCalibration]
    regime_score_column: str = DEFAULT_REGIME_SCORE_COLUMN
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD


def symmetric_quantile_pairs(quantiles: list[float]) -> list[tuple[float, float]]:
    normalized = sorted(float(value) for value in quantiles)
    pairs: list[tuple[float, float]] = []
    for quantile in normalized:
        if quantile >= 0.5:
            continue
        upper = _resolve_quantile(normalized, 1.0 - quantile)
        if upper is None:
            raise ValueError(
                "CQR requires symmetric lower/upper quantile pairs around 0.5. "
                f"Missing partner for quantile={quantile}."
            )
        pairs.append((quantile, upper))
    return pairs


def parse_interval_coverage_floors(
    interval_coverage_floors: dict[str, float] | None,
) -> dict[tuple[float, float], float]:
    if not interval_coverage_floors:
        return {}

    parsed: dict[tuple[float, float], float] = {}
    for key, value in interval_coverage_floors.items():
        lower_text, upper_text = str(key).split("-", maxsplit=1)
        lower = float(lower_text)
        upper = float(upper_text)
        parsed[(lower, upper)] = float(value)
    return parsed


def fit_conformal_quantile_calibration(
    predictions: pd.DataFrame,
    *,
    calibration_method: str = DEFAULT_QUANTILE_CALIBRATION_METHOD,
    group_by: str | None = None,
    interval_coverage_floors: dict[tuple[float, float], float] | dict[str, float] | None = None,
    min_group_size: int = 1,
    regime_score_column: str = DEFAULT_REGIME_SCORE_COLUMN,
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
) -> ConformalQuantileCalibration:
    method = _normalize_calibration_method(calibration_method)
    group_by = _normalize_group_by(group_by)
    min_group_size = int(min_group_size)
    regime_threshold = _normalize_regime_threshold(regime_threshold)
    if min_group_size <= 0:
        raise ValueError("min_group_size must be >= 1.")

    if not is_quantile_prediction_frame(predictions):
        return ConformalQuantileCalibration(
            method=method,
            group_by=group_by,
            adjustments={},
            regime_score_column=regime_score_column,
            regime_threshold=regime_threshold,
        )

    calibrated_source = enforce_monotonic_quantiles(predictions)
    pairs = symmetric_quantile_pairs(quantile_values(calibrated_source))
    if not pairs:
        return ConformalQuantileCalibration(
            method=method,
            group_by=group_by,
            adjustments={},
            regime_score_column=regime_score_column,
            regime_threshold=regime_threshold,
        )

    floors = _normalize_interval_coverage_floors(interval_coverage_floors)
    prediction_grid = calibrated_source.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)
    y_true = calibrated_source.groupby("ds", sort=True)["y"].first().reindex(prediction_grid.index).to_numpy(dtype=float)
    groups = _group_labels(
        calibrated_source,
        prediction_grid.index,
        group_by,
        regime_score_column=regime_score_column,
        regime_threshold=regime_threshold,
    )

    adjustments: dict[tuple[object, float, float], QuantilePairCalibration] = {}
    for lower, upper in pairs:
        target_coverage = _pair_target_coverage(lower, upper, floors)
        global_adjustment = _fit_pair_adjustment(
            lower_values=prediction_grid[lower].to_numpy(dtype=float),
            upper_values=prediction_grid[upper].to_numpy(dtype=float),
            y_true=y_true,
            target_coverage=target_coverage,
            method=method,
        )
        adjustments[(ALL_GROUP, lower, upper)] = global_adjustment

        if group_by is None:
            continue

        for group_label in pd.Index(groups).unique():
            mask = _group_mask(groups, group_label)
            if int(mask.sum()) < min_group_size:
                continue
            adjustments[(group_label, lower, upper)] = _fit_pair_adjustment(
                lower_values=prediction_grid.loc[mask, lower].to_numpy(dtype=float),
                upper_values=prediction_grid.loc[mask, upper].to_numpy(dtype=float),
                y_true=y_true[mask],
                target_coverage=target_coverage,
                method=method,
            )
    return ConformalQuantileCalibration(
        method=method,
        group_by=group_by,
        adjustments=adjustments,
        regime_score_column=regime_score_column,
        regime_threshold=regime_threshold,
    )


def apply_conformal_quantile_calibration(
    predictions: pd.DataFrame,
    calibration: ConformalQuantileCalibration,
) -> pd.DataFrame:
    if not is_quantile_prediction_frame(predictions) or not calibration.adjustments:
        return predictions.copy()

    corrected = predictions.copy()
    quantile_series = corrected["quantile"].astype(float)
    prediction_grid = corrected.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)
    group_labels = _group_labels(
        corrected,
        prediction_grid.index,
        calibration.group_by,
        regime_score_column=calibration.regime_score_column,
        regime_threshold=calibration.regime_threshold,
    )
    labels_by_ds = pd.Series(group_labels, index=prediction_grid.index)
    groups = pd.to_datetime(corrected["ds"]).map(labels_by_ds)
    for lower, upper in symmetric_quantile_pairs(quantile_values(corrected)):
        for group_label in labels_by_ds.unique():
            adjustment = calibration.adjustments.get((group_label, lower, upper))
            if adjustment is None:
                adjustment = calibration.adjustments.get((ALL_GROUP, lower, upper))
            if adjustment is None:
                continue

            group_mask = _group_mask(groups.to_numpy(dtype=object), group_label)
            lower_mask = group_mask & np.isclose(quantile_series, lower)
            upper_mask = group_mask & np.isclose(quantile_series, upper)
            corrected.loc[lower_mask, "y_pred"] = (
                corrected.loc[lower_mask, "y_pred"].astype(float) - adjustment.lower_adjustment
            )
            corrected.loc[upper_mask, "y_pred"] = (
                corrected.loc[upper_mask, "y_pred"].astype(float) + adjustment.upper_adjustment
            )
    return enforce_monotonic_quantiles(corrected)


def postprocess_quantile_predictions(
    predictions: pd.DataFrame,
    *,
    monotonic: bool = True,
    calibration_frame: pd.DataFrame | None = None,
    calibration_method: str = DEFAULT_QUANTILE_CALIBRATION_METHOD,
    calibration_group_by: str | None = None,
    calibration_interval_coverage_floors: dict[tuple[float, float], float] | dict[str, float] | None = None,
    calibration_min_group_size: int = 1,
    calibration_regime_score_column: str = DEFAULT_REGIME_SCORE_COLUMN,
    calibration_regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
) -> pd.DataFrame:
    if not is_quantile_prediction_frame(predictions):
        return predictions.copy()

    corrected = predictions.copy()
    if monotonic:
        corrected = enforce_monotonic_quantiles(corrected)
    if calibration_frame is not None and is_quantile_prediction_frame(calibration_frame):
        calibration = fit_conformal_quantile_calibration(
            calibration_frame,
            calibration_method=calibration_method,
            group_by=calibration_group_by,
            interval_coverage_floors=calibration_interval_coverage_floors,
            min_group_size=calibration_min_group_size,
            regime_score_column=calibration_regime_score_column,
            regime_threshold=calibration_regime_threshold,
        )
        corrected = apply_conformal_quantile_calibration(corrected, calibration)
    return corrected


def _normalize_calibration_method(calibration_method: str) -> str:
    method = str(calibration_method).lower()
    if method not in SUPPORTED_QUANTILE_CALIBRATION_METHODS:
        raise ValueError(f"Unsupported quantile calibration method: {calibration_method!r}")
    return method


def _normalize_group_by(group_by: str | None) -> str | None:
    normalized = None if group_by in {None, "", "none"} else str(group_by).lower()
    if normalized not in SUPPORTED_QUANTILE_CALIBRATION_GROUPS:
        raise ValueError(f"Unsupported quantile calibration group_by: {group_by!r}")
    return normalized


def _normalize_regime_threshold(regime_threshold: float) -> float:
    threshold = float(regime_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("regime_threshold must be in (0, 1).")
    return threshold


def _normalize_interval_coverage_floors(
    interval_coverage_floors: dict[tuple[float, float], float] | dict[str, float] | None,
) -> dict[tuple[float, float], float]:
    if not interval_coverage_floors:
        return {}
    if any(isinstance(key, str) for key in interval_coverage_floors):
        return parse_interval_coverage_floors(interval_coverage_floors)  # type: ignore[arg-type]
    return {(float(lower), float(upper)): float(value) for (lower, upper), value in interval_coverage_floors.items()}  # type: ignore[union-attr]


def _fit_pair_adjustment(
    *,
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    y_true: np.ndarray,
    target_coverage: float,
    method: str,
) -> QuantilePairCalibration:
    alpha = 1.0 - float(target_coverage)
    if not 0.0 < target_coverage < 1.0:
        raise ValueError(f"target_coverage must be in (0, 1); received {target_coverage!r}")

    if method == "cqr":
        scores = np.maximum(lower_values - y_true, y_true - upper_values)
        adjustment = _conformal_quantile(scores, 1.0 - alpha)
        return QuantilePairCalibration(lower_adjustment=adjustment, upper_adjustment=adjustment)

    if method == "cqr_asymmetric":
        tail_target = 1.0 - alpha / 2.0
        lower_adjustment = _conformal_quantile(lower_values - y_true, tail_target)
        upper_adjustment = _conformal_quantile(y_true - upper_values, tail_target)
        return QuantilePairCalibration(
            lower_adjustment=lower_adjustment,
            upper_adjustment=upper_adjustment,
        )

    raise ValueError(f"Unsupported quantile calibration method: {method!r}")


def _conformal_quantile(scores: np.ndarray, target_level: float) -> float:
    values = np.asarray(scores, dtype=float)
    level = min(1.0, np.ceil((len(values) + 1) * float(target_level)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def _pair_target_coverage(
    lower: float,
    upper: float,
    interval_coverage_floors: dict[tuple[float, float], float],
) -> float:
    nominal = upper - lower
    override = interval_coverage_floors.get((lower, upper))
    if override is None:
        return nominal
    return float(override)


def _group_labels(
    frame: pd.DataFrame,
    index: pd.Index,
    group_by: str | None,
    *,
    regime_score_column: str = DEFAULT_REGIME_SCORE_COLUMN,
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
) -> np.ndarray:
    timestamps = pd.to_datetime(index)
    if group_by is None:
        return np.repeat(ALL_GROUP, len(timestamps))
    hour_labels = timestamps.hour.to_numpy(dtype=int)
    if group_by == "hour":
        return hour_labels
    if group_by == "hour_x_regime":
        if regime_score_column not in frame.columns:
            raise ValueError(f"hour_x_regime calibration requires prediction column {regime_score_column!r}.")
        context = frame.loc[:, ["ds", regime_score_column]].drop_duplicates("ds").copy()
        context["ds"] = pd.to_datetime(context["ds"], utc=False)
        scores = context.set_index("ds").reindex(index)[regime_score_column]
        if scores.isna().any():
            raise ValueError(f"hour_x_regime calibration found missing {regime_score_column!r} values.")
        regime_labels = (scores.astype(float).to_numpy() >= float(regime_threshold)).astype(int)
        labels = np.empty(len(hour_labels), dtype=object)
        for idx, (hour, regime) in enumerate(zip(hour_labels, regime_labels, strict=True)):
            labels[idx] = (int(hour), int(regime))
        return labels
    raise ValueError(f"Unsupported quantile calibration group_by: {group_by!r}")


def _group_mask(groups: np.ndarray, target: object) -> np.ndarray:
    return np.fromiter((label == target for label in groups), dtype=bool, count=len(groups))


def _resolve_quantile(quantiles: list[float], target: float) -> float | None:
    for quantile in quantiles:
        if np.isclose(quantile, target):
            return float(quantile)
    return None
