from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from pjm_forecast.evaluation.hour_x_regime_grid import split_validation_holdout
from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.evaluation.spike_score_diagnostics import compute_spike_score_diagnostics
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions


@dataclass(frozen=True)
class MedianBiasGridResult:
    validation_summary: pd.DataFrame
    test_summary: pd.DataFrame


def evaluate_median_bias_grid(
    validation_frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame | None = None,
    max_abs_adjustments: Iterable[float] = (5.0, 10.0, 20.0),
    validation_holdout_days: int = 91,
    min_group_size: int = 24,
    group_by: str = "hour",
    regime_score_column: str = "spike_score",
    regime_threshold: float = 0.50,
    calibration_method: str = "cqr_asymmetric",
    interval_coverage_floors: dict[str, float] | None = None,
) -> MedianBiasGridResult:
    validation_calibration, validation_eval = split_validation_holdout(
        validation_frame,
        holdout_days=validation_holdout_days,
    )
    validation_summary = evaluate_median_bias_variants(
        eval_frame=validation_eval,
        calibration_frame=validation_calibration,
        mode="validation_holdout",
        max_abs_adjustments=max_abs_adjustments,
        min_group_size=min_group_size,
        group_by=group_by,
        regime_score_column=regime_score_column,
        regime_threshold=regime_threshold,
        calibration_method=calibration_method,
        interval_coverage_floors=interval_coverage_floors,
    )
    test_summary = pd.DataFrame()
    if test_frame is not None:
        test_summary = evaluate_median_bias_variants(
            eval_frame=test_frame,
            calibration_frame=validation_frame,
            mode="test",
            max_abs_adjustments=max_abs_adjustments,
            min_group_size=min_group_size,
            group_by=group_by,
            regime_score_column=regime_score_column,
            regime_threshold=regime_threshold,
            calibration_method=calibration_method,
            interval_coverage_floors=interval_coverage_floors,
        )
    return MedianBiasGridResult(validation_summary=validation_summary, test_summary=test_summary)


def evaluate_median_bias_variants(
    *,
    eval_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    mode: str,
    max_abs_adjustments: Iterable[float],
    min_group_size: int = 24,
    group_by: str = "hour",
    regime_score_column: str = "spike_score",
    regime_threshold: float = 0.50,
    calibration_method: str = "cqr_asymmetric",
    interval_coverage_floors: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, max_abs_adjustment in _variant_specs(max_abs_adjustments):
        params: dict[str, object] = {
            "monotonic": True,
            "calibration_frame": calibration_frame if variant_name != "raw_monotonic" else None,
            "calibration_method": calibration_method,
            "calibration_group_by": group_by if variant_name != "raw_monotonic" else None,
            "calibration_interval_coverage_floors": interval_coverage_floors,
            "calibration_min_group_size": int(min_group_size),
            "calibration_regime_score_column": regime_score_column,
            "calibration_regime_threshold": float(regime_threshold),
        }
        if max_abs_adjustment is not None:
            params.update(
                {
                    "median_bias_enabled": True,
                    "median_bias_group_by": group_by,
                    "median_bias_min_group_size": int(min_group_size),
                    "median_bias_regime_score_column": regime_score_column,
                    "median_bias_regime_threshold": float(regime_threshold),
                    "median_bias_max_abs_adjustment": float(max_abs_adjustment),
                }
            )

        processed = postprocess_quantile_predictions(eval_frame, **params)
        row: dict[str, object] = {
            "mode": mode,
            "variant": variant_name,
            "group_by": group_by if variant_name != "raw_monotonic" else "none",
            "max_abs_adjustment": max_abs_adjustment,
        }
        row.update(compute_metrics(processed))
        row.update(compute_quantile_diagnostics(processed))
        row.update(
            compute_spike_score_diagnostics(
                processed,
                score_column=regime_score_column,
                threshold=regime_threshold,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)


def _variant_specs(max_abs_adjustments: Iterable[float]) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = [
        ("raw_monotonic", None),
        ("hour_cqr", None),
    ]
    for value in max_abs_adjustments:
        adjustment = float(value)
        label = int(adjustment) if adjustment.is_integer() else str(adjustment).replace(".", "p")
        specs.append((f"hour_median_bias_cap{label}", adjustment))
    return specs
