from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.evaluation.spike_score_diagnostics import compute_spike_score_diagnostics
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions


@dataclass(frozen=True)
class HourXRegimeGridResult:
    validation_summary: pd.DataFrame
    test_summary: pd.DataFrame


def evaluate_hour_x_regime_threshold_grid(
    validation_frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame | None = None,
    thresholds: Iterable[float] = (0.50, 0.67),
    validation_holdout_days: int = 91,
    min_group_size: int = 24,
    regime_score_column: str = "spike_score",
    calibration_method: str = "cqr_asymmetric",
) -> HourXRegimeGridResult:
    validation_calibration, validation_eval = split_validation_holdout(
        validation_frame,
        holdout_days=validation_holdout_days,
    )
    validation_summary = evaluate_hour_x_regime_variants(
        eval_frame=validation_eval,
        calibration_frame=validation_calibration,
        mode="validation_holdout",
        thresholds=thresholds,
        min_group_size=min_group_size,
        regime_score_column=regime_score_column,
        calibration_method=calibration_method,
    )
    test_summary = pd.DataFrame()
    if test_frame is not None:
        test_summary = evaluate_hour_x_regime_variants(
            eval_frame=test_frame,
            calibration_frame=validation_frame,
            mode="test",
            thresholds=thresholds,
            min_group_size=min_group_size,
            regime_score_column=regime_score_column,
            calibration_method=calibration_method,
        )
    return HourXRegimeGridResult(validation_summary=validation_summary, test_summary=test_summary)


def split_validation_holdout(frame: pd.DataFrame, *, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_day_series = pd.to_datetime(frame["ds"]).dt.floor("D")
    forecast_days = pd.Index(forecast_day_series.unique()).sort_values()
    if holdout_days <= 0 or holdout_days >= len(forecast_days):
        raise ValueError(f"holdout_days must be in [1, {len(forecast_days) - 1}]")

    calibration_days = set(forecast_days[:-holdout_days])
    evaluation_days = set(forecast_days[-holdout_days:])
    calibration_frame = frame.loc[forecast_day_series.isin(calibration_days)].copy()
    evaluation_frame = frame.loc[forecast_day_series.isin(evaluation_days)].copy()
    return calibration_frame, evaluation_frame


def evaluate_hour_x_regime_variants(
    *,
    eval_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    mode: str,
    thresholds: Iterable[float],
    min_group_size: int = 24,
    regime_score_column: str = "spike_score",
    calibration_method: str = "cqr_asymmetric",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, group_by, threshold in _variant_specs(thresholds):
        params: dict[str, object] = {"monotonic": True}
        if group_by is not None:
            params.update(
                {
                    "calibration_frame": calibration_frame,
                    "calibration_method": calibration_method,
                    "calibration_group_by": group_by,
                    "calibration_min_group_size": int(min_group_size),
                }
            )
        if group_by == "hour_x_regime":
            params.update(
                {
                    "calibration_regime_score_column": regime_score_column,
                    "calibration_regime_threshold": float(threshold),
                }
            )

        processed = postprocess_quantile_predictions(eval_frame, **params)
        row: dict[str, object] = {
            "mode": mode,
            "variant": variant_name,
            "group_by": group_by or "none",
            "regime_threshold": threshold,
        }
        row.update(compute_metrics(processed))
        row.update(compute_quantile_diagnostics(processed))
        row.update(compute_spike_score_diagnostics(processed, score_column=regime_score_column, threshold=threshold or 0.67))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)


def _variant_specs(thresholds: Iterable[float]) -> list[tuple[str, str | None, float | None]]:
    specs: list[tuple[str, str | None, float | None]] = [
        ("raw_monotonic", None, None),
        ("hour_cqr", "hour", None),
    ]
    for threshold in thresholds:
        threshold_value = float(threshold)
        label = int(round(threshold_value * 100))
        specs.append((f"hour_regime_cqr_t{label}", "hour_x_regime", threshold_value))
    return specs
