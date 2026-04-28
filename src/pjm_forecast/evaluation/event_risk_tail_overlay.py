from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pjm_forecast.evaluation.hour_x_regime_grid import split_validation_holdout
from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.evaluation.spike_score_diagnostics import compute_spike_score_diagnostics
from pjm_forecast.prediction_contract import enforce_monotonic_quantiles, is_quantile_prediction_frame
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions


@dataclass(frozen=True)
class EventRiskTailOverlay:
    risk_score_column: str
    risk_threshold: float
    risk_aggregation: str
    residual_quantile: float
    uplift: float
    target_quantiles: tuple[float, ...]


@dataclass(frozen=True)
class EventRiskTailOverlayGridResult:
    validation_summary: pd.DataFrame
    test_summary: pd.DataFrame


@dataclass(frozen=True)
class EventRiskTailOverlayAuditArtifacts:
    implementation_audit: dict[str, object]
    spike_score_audit: dict[str, object]
    active_day_diagnostics: pd.DataFrame
    active_days_by_month: pd.DataFrame
    width_by_regime: pd.DataFrame
    pinball_by_quantile: pd.DataFrame
    conservative_variant_grid: pd.DataFrame
    daily_max_gap_detail: pd.DataFrame
    event_day_before_after: pd.DataFrame


def evaluate_event_risk_tail_overlay_grid(
    validation_frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame | None = None,
    validation_holdout_days: int = 91,
    risk_score_column: str = "spike_score",
    risk_aggregations: Iterable[str] = ("mean",),
    risk_threshold_quantiles: Iterable[float] = (0.85, 0.90, 0.95),
    residual_quantiles: Iterable[float] = (0.50, 0.75, 0.90),
    max_uplifts: Iterable[float] = (25.0, 50.0, 100.0),
    target_quantiles: Iterable[float] = (0.99, 0.995),
    calibration_method: str = "cqr_asymmetric",
    calibration_group_by: str | None = "hour",
    calibration_min_group_size: int = 24,
    interval_coverage_floors: dict[str, float] | None = None,
    regime_threshold: float = 0.50,
) -> EventRiskTailOverlayGridResult:
    validation_calibration, validation_eval = split_validation_holdout(
        validation_frame,
        holdout_days=validation_holdout_days,
    )
    validation_summary = evaluate_event_risk_tail_overlay_variants(
        eval_frame=validation_eval,
        calibration_frame=validation_calibration,
        mode="validation_holdout",
        risk_score_column=risk_score_column,
        risk_aggregations=risk_aggregations,
        risk_threshold_quantiles=risk_threshold_quantiles,
        residual_quantiles=residual_quantiles,
        max_uplifts=max_uplifts,
        target_quantiles=target_quantiles,
        calibration_method=calibration_method,
        calibration_group_by=calibration_group_by,
        calibration_min_group_size=calibration_min_group_size,
        interval_coverage_floors=interval_coverage_floors,
        regime_threshold=regime_threshold,
    )
    test_summary = pd.DataFrame()
    if test_frame is not None:
        test_summary = evaluate_event_risk_tail_overlay_variants(
            eval_frame=test_frame,
            calibration_frame=validation_frame,
            mode="test",
            risk_score_column=risk_score_column,
            risk_aggregations=risk_aggregations,
            risk_threshold_quantiles=risk_threshold_quantiles,
            residual_quantiles=residual_quantiles,
            max_uplifts=max_uplifts,
            target_quantiles=target_quantiles,
            calibration_method=calibration_method,
            calibration_group_by=calibration_group_by,
            calibration_min_group_size=calibration_min_group_size,
            interval_coverage_floors=interval_coverage_floors,
            regime_threshold=regime_threshold,
        )
    return EventRiskTailOverlayGridResult(validation_summary=validation_summary, test_summary=test_summary)


def build_event_risk_tail_overlay_audit_artifacts(
    validation_frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame | None = None,
    validation_holdout_days: int = 91,
    risk_score_column: str = "spike_score",
    risk_threshold_quantile: float = 0.90,
    risk_aggregation: str = "mean",
    residual_quantile: float = 1.0,
    max_uplift: float = 50.0,
    target_quantiles: Iterable[float] = (0.99, 0.995),
    calibration_method: str = "cqr_asymmetric",
    calibration_group_by: str | None = "hour",
    calibration_min_group_size: int = 24,
    interval_coverage_floors: dict[str, float] | None = None,
    regime_threshold: float = 0.50,
    risk_score_input_columns: Iterable[str] | None = None,
) -> EventRiskTailOverlayAuditArtifacts:
    validation_calibration, validation_eval = split_validation_holdout(
        validation_frame,
        holdout_days=validation_holdout_days,
    )
    validation_before, validation_after, validation_overlay = _before_after_frames(
        eval_frame=validation_eval,
        calibration_frame=validation_calibration,
        risk_score_column=risk_score_column,
        risk_threshold_quantile=risk_threshold_quantile,
        risk_aggregation=risk_aggregation,
        residual_quantile=residual_quantile,
        max_uplift=max_uplift,
        target_quantiles=target_quantiles,
        calibration_method=calibration_method,
        calibration_group_by=calibration_group_by,
        calibration_min_group_size=calibration_min_group_size,
        interval_coverage_floors=interval_coverage_floors,
        regime_threshold=regime_threshold,
    )
    before_after: dict[str, tuple[pd.DataFrame, pd.DataFrame, EventRiskTailOverlay]] = {
        "validation_holdout": (validation_before, validation_after, validation_overlay)
    }
    if test_frame is not None:
        test_before, test_after, test_overlay = _before_after_frames(
            eval_frame=test_frame,
            calibration_frame=validation_frame,
            risk_score_column=risk_score_column,
            risk_threshold_quantile=risk_threshold_quantile,
            risk_aggregation=risk_aggregation,
            residual_quantile=residual_quantile,
            max_uplift=max_uplift,
            target_quantiles=target_quantiles,
            calibration_method=calibration_method,
            calibration_group_by=calibration_group_by,
            calibration_min_group_size=calibration_min_group_size,
            interval_coverage_floors=interval_coverage_floors,
            regime_threshold=regime_threshold,
        )
        before_after["test"] = (test_before, test_after, test_overlay)

    reference_before, reference_after, reference_overlay = before_after.get("test", before_after["validation_holdout"])
    implementation_audit = {
        "validation_holdout_days": int(validation_holdout_days),
        "calibration_source": "validation_early_segment",
        "selection_source": "validation_holdout",
        "test_used_for_selection": False,
        "residual_source": "raw_monotonic",
        "overlay_application_order": "after_hourly_cqr",
        "target_quantiles": list(reference_overlay.target_quantiles),
        "q50_changed": _quantile_changed(reference_before, reference_after, 0.50),
        "crossing_after_overlay": compute_quantile_diagnostics(reference_after)["crossing_rate"],
        "selected_variant": _variant_name(risk_aggregation, risk_threshold_quantile, residual_quantile, max_uplift),
    }
    spike_score_audit = {
        "risk_score_column": risk_score_column,
        "input_columns": list(risk_score_input_columns or []),
        "uses_y": False,
        "uses_horizon_truth": False,
        "uses_energy_truth": False,
        "uses_congestion_loss_truth": False,
        "uses_global_rank": False,
        "uses_historical_percentile": True,
        "fit_split": "validation",
        "threshold_source": "validation",
        "test_refit": False,
        "availability_status": "PASS" if risk_score_column in validation_frame.columns else "FAIL",
        "notes": "Risk threshold and uplift are fit from validation source only; test is apply-only.",
    }

    active_rows = []
    active_month_rows = []
    width_rows = []
    pinball_rows = []
    daily_rows = []
    event_rows = []
    for split_name, (before, after, overlay) in before_after.items():
        active_rows.extend(_active_day_diagnostic_rows(split_name, before, after, overlay))
        active_month_rows.extend(_active_days_by_month_rows(split_name, before, after, overlay))
        width_rows.extend(_width_by_regime_rows(split_name, before, after, overlay))
        pinball_rows.extend(_pinball_by_quantile_rows(split_name, before, after))
        daily_rows.extend(_daily_max_gap_rows(split_name, before, after, overlay))
        event_rows.extend(_event_day_rows(split_name, before, after, overlay))

    conservative = pd.concat(
        [
            evaluate_event_risk_tail_overlay_variants(
                eval_frame=validation_eval,
                calibration_frame=validation_calibration,
                mode="validation_holdout",
                risk_score_column=risk_score_column,
                risk_aggregations=[risk_aggregation],
                risk_threshold_quantiles=[0.80, 0.90, 0.95],
                residual_quantiles=[0.99, 1.0],
                max_uplifts=[25.0, 50.0],
                target_quantiles=target_quantiles,
                calibration_method=calibration_method,
                calibration_group_by=calibration_group_by,
                calibration_min_group_size=calibration_min_group_size,
                interval_coverage_floors=interval_coverage_floors,
                regime_threshold=regime_threshold,
            )
        ],
        ignore_index=True,
    )
    conservative["width98_ratio"] = conservative["width_98"] / float(
        conservative.loc[conservative["variant"].eq("hour_cqr"), "width_98"].iloc[0]
    )
    conservative["decision"] = np.where(
        conservative["variant"].eq(implementation_audit["selected_variant"]),
        "selected",
        "candidate",
    )

    return EventRiskTailOverlayAuditArtifacts(
        implementation_audit=implementation_audit,
        spike_score_audit=spike_score_audit,
        active_day_diagnostics=pd.DataFrame(active_rows),
        active_days_by_month=pd.DataFrame(active_month_rows),
        width_by_regime=pd.DataFrame(width_rows),
        pinball_by_quantile=pd.DataFrame(pinball_rows),
        conservative_variant_grid=conservative,
        daily_max_gap_detail=pd.DataFrame(daily_rows),
        event_day_before_after=pd.DataFrame(event_rows),
    )


def evaluate_event_risk_tail_overlay_variants(
    *,
    eval_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    mode: str,
    risk_score_column: str = "spike_score",
    risk_aggregations: Iterable[str] = ("mean",),
    risk_threshold_quantiles: Iterable[float] = (0.85, 0.90, 0.95),
    residual_quantiles: Iterable[float] = (0.50, 0.75, 0.90),
    max_uplifts: Iterable[float] = (25.0, 50.0, 100.0),
    target_quantiles: Iterable[float] = (0.99, 0.995),
    calibration_method: str = "cqr_asymmetric",
    calibration_group_by: str | None = "hour",
    calibration_min_group_size: int = 24,
    interval_coverage_floors: dict[str, float] | None = None,
    regime_threshold: float = 0.50,
) -> pd.DataFrame:
    calibration_base = enforce_monotonic_quantiles(calibration_frame)
    eval_post = _hourly_cqr(
        eval_frame,
        calibration_frame=calibration_frame,
        calibration_method=calibration_method,
        calibration_group_by=calibration_group_by,
        calibration_min_group_size=calibration_min_group_size,
        interval_coverage_floors=interval_coverage_floors,
        risk_score_column=risk_score_column,
        regime_threshold=regime_threshold,
    )
    rows: list[dict[str, object]] = [
        _summarize_variant(
            eval_post,
            mode=mode,
            variant="hour_cqr",
            risk_score_column=risk_score_column,
            overlay=None,
        )
    ]
    for risk_aggregation in risk_aggregations:
        for risk_threshold_quantile in risk_threshold_quantiles:
            for residual_quantile in residual_quantiles:
                for max_uplift in max_uplifts:
                    overlay = fit_event_risk_tail_overlay(
                        calibration_base,
                        risk_score_column=risk_score_column,
                        risk_threshold_quantile=float(risk_threshold_quantile),
                        risk_aggregation=str(risk_aggregation),
                        residual_quantile=float(residual_quantile),
                        max_uplift=float(max_uplift),
                        target_quantiles=target_quantiles,
                    )
                    adjusted = apply_event_risk_tail_overlay(eval_post, overlay)
                    rows.append(
                        _summarize_variant(
                            adjusted,
                            mode=mode,
                            variant=_variant_name(
                                str(risk_aggregation),
                                risk_threshold_quantile,
                                residual_quantile,
                                max_uplift,
                            ),
                            risk_score_column=risk_score_column,
                            overlay=overlay,
                        )
                    )
    return pd.DataFrame(rows).sort_values(["pinball", "q99_excess_mean", "variant"]).reset_index(drop=True)


def fit_event_risk_tail_overlay(
    predictions: pd.DataFrame,
    *,
    risk_score_column: str = "spike_score",
    risk_threshold_quantile: float = 0.90,
    risk_aggregation: str = "mean",
    residual_quantile: float = 0.75,
    max_uplift: float | None = None,
    target_quantiles: Iterable[float] = (0.99, 0.995),
) -> EventRiskTailOverlay:
    _validate_prediction_frame(predictions, risk_score_column)
    risk_threshold_quantile = _bounded_unit_interval(risk_threshold_quantile, "risk_threshold_quantile")
    residual_quantile = _bounded_unit_interval(residual_quantile, "residual_quantile")
    target_quantile_tuple = tuple(sorted(float(value) for value in target_quantiles))
    if not target_quantile_tuple:
        raise ValueError("target_quantiles must contain at least one quantile.")
    if max_uplift is not None and float(max_uplift) <= 0.0:
        raise ValueError("max_uplift must be > 0 when configured.")

    corrected = enforce_monotonic_quantiles(predictions)
    daily_risk = _daily_risk_scores(corrected, risk_score_column, risk_aggregation)
    threshold = float(daily_risk.quantile(risk_threshold_quantile, interpolation="higher"))
    high_risk_days = daily_risk[daily_risk >= threshold].index

    grid = _prediction_grid(corrected)
    q99_column = _resolve_quantile_column(grid.columns, 0.99)
    if q99_column is None:
        raise ValueError("event-risk tail overlay requires q0.99 predictions.")
    y_true = corrected.groupby("ds", sort=True)["y"].first()
    residual = y_true.reindex(grid.index).astype(float) - grid[q99_column].astype(float)
    days = pd.DatetimeIndex(grid.index).floor("D")
    active_residual = residual.loc[pd.Index(days).isin(high_risk_days)].clip(lower=0.0)
    uplift = 0.0 if active_residual.empty else float(np.quantile(active_residual.to_numpy(dtype=float), residual_quantile))
    if max_uplift is not None:
        uplift = float(np.clip(uplift, 0.0, float(max_uplift)))

    return EventRiskTailOverlay(
        risk_score_column=risk_score_column,
        risk_threshold=threshold,
        risk_aggregation=risk_aggregation,
        residual_quantile=residual_quantile,
        uplift=uplift,
        target_quantiles=target_quantile_tuple,
    )


def apply_event_risk_tail_overlay(
    predictions: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> pd.DataFrame:
    _validate_prediction_frame(predictions, overlay.risk_score_column)
    corrected = predictions.copy()
    daily_risk = _daily_risk_scores(corrected, overlay.risk_score_column, overlay.risk_aggregation)
    active_days = set(daily_risk[daily_risk >= overlay.risk_threshold].index)
    days = pd.to_datetime(corrected["ds"]).dt.floor("D")
    day_mask = days.isin(active_days)
    quantile_values = corrected["quantile"].astype(float)
    quantile_mask = np.zeros(len(corrected), dtype=bool)
    for quantile in overlay.target_quantiles:
        quantile_mask |= np.isclose(quantile_values.to_numpy(dtype=float), float(quantile))
    corrected.loc[day_mask & quantile_mask, "y_pred"] = (
        corrected.loc[day_mask & quantile_mask, "y_pred"].astype(float) + float(overlay.uplift)
    )
    return enforce_monotonic_quantiles(corrected)


def _before_after_frames(
    *,
    eval_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    risk_score_column: str,
    risk_threshold_quantile: float,
    risk_aggregation: str,
    residual_quantile: float,
    max_uplift: float,
    target_quantiles: Iterable[float],
    calibration_method: str,
    calibration_group_by: str | None,
    calibration_min_group_size: int,
    interval_coverage_floors: dict[str, float] | None,
    regime_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, EventRiskTailOverlay]:
    before = _hourly_cqr(
        eval_frame,
        calibration_frame=calibration_frame,
        calibration_method=calibration_method,
        calibration_group_by=calibration_group_by,
        calibration_min_group_size=calibration_min_group_size,
        interval_coverage_floors=interval_coverage_floors,
        risk_score_column=risk_score_column,
        regime_threshold=regime_threshold,
    )
    overlay = fit_event_risk_tail_overlay(
        enforce_monotonic_quantiles(calibration_frame),
        risk_score_column=risk_score_column,
        risk_threshold_quantile=risk_threshold_quantile,
        risk_aggregation=risk_aggregation,
        residual_quantile=residual_quantile,
        max_uplift=max_uplift,
        target_quantiles=target_quantiles,
    )
    after = apply_event_risk_tail_overlay(before, overlay)
    return before, after, overlay


def _quantile_changed(before: pd.DataFrame, after: pd.DataFrame, quantile: float) -> bool:
    before_grid = _prediction_grid(before)
    after_grid = _prediction_grid(after)
    before_column = _resolve_quantile_column(before_grid.columns, quantile)
    after_column = _resolve_quantile_column(after_grid.columns, quantile)
    if before_column is None or after_column is None:
        return False
    return not np.allclose(before_grid[before_column].to_numpy(dtype=float), after_grid[after_column].to_numpy(dtype=float))


def _active_mask(frame: pd.DataFrame, overlay: EventRiskTailOverlay) -> pd.Series:
    point = _point_context(frame, overlay.risk_score_column)
    daily_risk = _daily_risk_scores(frame, overlay.risk_score_column, overlay.risk_aggregation)
    active_days = set(daily_risk[daily_risk >= overlay.risk_threshold].index)
    return point["day"].isin(active_days)


def _point_context(frame: pd.DataFrame, risk_score_column: str) -> pd.DataFrame:
    context = frame.loc[:, ["ds", "y", risk_score_column]].drop_duplicates("ds").copy()
    context["ds"] = pd.to_datetime(context["ds"])
    context["day"] = context["ds"].dt.floor("D")
    context["month"] = context["ds"].dt.to_period("M").astype(str)
    return context.sort_values("ds").reset_index(drop=True)


def _wide_metrics_frame(before: pd.DataFrame, after: pd.DataFrame, risk_score_column: str) -> pd.DataFrame:
    before_grid = _prediction_grid(before)
    after_grid = _prediction_grid(after)
    point = _point_context(before, risk_score_column).set_index("ds")
    result = point.copy()
    for label, grid in [("before", before_grid), ("after", after_grid)]:
        for quantile in [0.01, 0.50, 0.99]:
            column = _resolve_quantile_column(grid.columns, quantile)
            if column is not None:
                result[f"q{int(quantile * 100):02d}_{label}"] = grid[column]
        q01_column = _resolve_quantile_column(grid.columns, 0.01)
        q99_column = _resolve_quantile_column(grid.columns, 0.99)
        if q01_column is not None and q99_column is not None:
            result[f"width98_{label}"] = grid[q99_column] - grid[q01_column]
        if q99_column is not None:
            result[f"q99_excess_{label}"] = (result["y"].astype(float) - grid[q99_column].astype(float)).clip(lower=0.0)
            result[f"q99_gap_{label}"] = result["y"].astype(float) - grid[q99_column].astype(float)
    return result.reset_index()


def _active_day_diagnostic_rows(
    split: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> list[dict[str, object]]:
    metrics = _wide_metrics_frame(before, after, overlay.risk_score_column)
    active = _active_mask(before, overlay).reset_index(drop=True)
    rows = []
    for label, mask in [("active", active), ("inactive", ~active)]:
        subset = metrics.loc[mask].copy()
        day_count = int(subset["day"].nunique()) if not subset.empty else 0
        rows.append(
            {
                "split": split,
                "segment": label,
                "active_days": int(metrics.loc[active, "day"].nunique()),
                "total_days": int(metrics["day"].nunique()),
                "active_rate": float(active.groupby(metrics["day"]).first().mean()),
                "segment_days": day_count,
                "q99_excess_before": _safe_mean(subset, "q99_excess_before"),
                "q99_excess_after": _safe_mean(subset, "q99_excess_after"),
                "width98_before": _safe_mean(subset, "width98_before"),
                "width98_after": _safe_mean(subset, "width98_after"),
            }
        )
    return rows


def _active_days_by_month_rows(
    split: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> list[dict[str, object]]:
    metrics = _wide_metrics_frame(before, after, overlay.risk_score_column)
    active = _active_mask(before, overlay).reset_index(drop=True)
    metrics["active"] = active
    rows = []
    for month, subset in metrics.groupby("month", sort=True):
        day_active = subset.groupby("day", sort=True)["active"].first()
        rows.append(
            {
                "split": split,
                "month": month,
                "active_days": int(day_active.sum()),
                "total_days": int(day_active.size),
                "active_rate": float(day_active.mean()),
                "q99_excess_before": _safe_mean(subset, "q99_excess_before"),
                "q99_excess_after": _safe_mean(subset, "q99_excess_after"),
                "width98_before": _safe_mean(subset, "width98_before"),
                "width98_after": _safe_mean(subset, "width98_after"),
            }
        )
    return rows


def _width_by_regime_rows(
    split: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> list[dict[str, object]]:
    metrics = _wide_metrics_frame(before, after, overlay.risk_score_column)
    metrics["regime"] = _price_regime_labels(metrics["y"].astype(float))
    daily_max_index = metrics.groupby("day", sort=True)["y"].idxmax()
    metrics["is_daily_max"] = metrics.index.isin(daily_max_index)
    metrics["active"] = _active_mask(before, overlay).reset_index(drop=True)
    masks: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=metrics.index)),
        ("normal", metrics["regime"].eq("normal")),
        ("high", metrics["regime"].eq("high")),
        ("spike", metrics["regime"].eq("spike")),
        ("extreme", metrics["regime"].eq("extreme")),
        ("daily_max", metrics["is_daily_max"]),
        ("active_day_normal", metrics["active"] & metrics["regime"].eq("normal")),
        ("active_day_high", metrics["active"] & metrics["regime"].eq("high")),
        ("active_day_extreme", metrics["active"] & metrics["regime"].eq("extreme")),
        ("inactive_day_normal", ~metrics["active"] & metrics["regime"].eq("normal")),
        ("inactive_day_extreme", ~metrics["active"] & metrics["regime"].eq("extreme")),
    ]
    rows = []
    for regime, mask in masks:
        subset = metrics.loc[mask].copy()
        width_before = _safe_mean(subset, "width98_before")
        width_after = _safe_mean(subset, "width98_after")
        rows.append(
            {
                "split": split,
                "regime": regime,
                "count": int(len(subset)),
                "width98_before": width_before,
                "width98_after": width_after,
                "width98_delta": width_after - width_before,
                "width98_ratio": width_after / width_before if width_before and not np.isnan(width_before) else np.nan,
                "pinball_before": _subset_pinball(before, subset["ds"]) if not subset.empty else np.nan,
                "pinball_after": _subset_pinball(after, subset["ds"]) if not subset.empty else np.nan,
                "q99_excess_before": _safe_mean(subset, "q99_excess_before"),
                "q99_excess_after": _safe_mean(subset, "q99_excess_after"),
            }
        )
    return rows


def _pinball_by_quantile_rows(split: str, before: pd.DataFrame, after: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    before_frame = before.copy()
    after_frame = after.copy()
    before_frame["quantile"] = before_frame["quantile"].astype(float)
    after_frame["quantile"] = after_frame["quantile"].astype(float)
    for quantile in sorted(before_frame["quantile"].unique()):
        before_subset = before_frame.loc[np.isclose(before_frame["quantile"].astype(float), quantile)]
        after_subset = after_frame.loc[np.isclose(after_frame["quantile"].astype(float), quantile)]
        before_loss = _pinball_loss_series(before_subset)
        after_loss = _pinball_loss_series(after_subset)
        rows.append(
            {
                "split": split,
                "quantile": float(quantile),
                "pinball_before": before_loss,
                "pinball_after": after_loss,
                "delta": after_loss - before_loss,
                "count": int(len(before_subset)),
            }
        )
    return rows


def _daily_max_gap_rows(
    split: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> list[dict[str, object]]:
    metrics = _wide_metrics_frame(before, after, overlay.risk_score_column)
    metrics["active"] = _active_mask(before, overlay).reset_index(drop=True)
    rows = []
    for day, subset in metrics.groupby("day", sort=True):
        max_idx = subset["y"].idxmax()
        row = metrics.loc[max_idx]
        rows.append(
            {
                "split": split,
                "day": day,
                "y_max": float(row["y"]),
                "q99_before": float(row.get("q99_before", np.nan)),
                "q99_after": float(row.get("q99_after", np.nan)),
                "gap_before": float(row.get("q99_gap_before", np.nan)),
                "gap_after": float(row.get("q99_gap_after", np.nan)),
                "active": bool(row["active"]),
                "spike_score_mean": float(subset[overlay.risk_score_column].mean()),
                "width98_before": float(row.get("width98_before", np.nan)),
                "width98_after": float(row.get("width98_after", np.nan)),
            }
        )
    return rows


def _event_day_rows(
    split: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlay: EventRiskTailOverlay,
) -> list[dict[str, object]]:
    daily_rows = pd.DataFrame(_daily_max_gap_rows(split, before, after, overlay))
    if daily_rows.empty:
        return []
    daily_rows["gap_improvement"] = daily_rows["gap_before"] - daily_rows["gap_after"]
    return daily_rows.sort_values("gap_before", ascending=False).head(20).to_dict("records")


def _price_regime_labels(y: pd.Series) -> pd.Series:
    p80 = float(y.quantile(0.80))
    p90 = float(y.quantile(0.90))
    p95 = float(y.quantile(0.95))
    labels = pd.Series("normal", index=y.index, dtype=object)
    labels.loc[(y >= p80) & (y < p90)] = "high"
    labels.loc[(y >= p90) & (y < p95)] = "spike"
    labels.loc[y >= p95] = "extreme"
    return labels


def _subset_pinball(frame: pd.DataFrame, timestamps: pd.Series) -> float:
    subset = frame.loc[pd.to_datetime(frame["ds"]).isin(pd.to_datetime(timestamps))].copy()
    if subset.empty:
        return np.nan
    return _pinball_loss_series(subset)


def _pinball_loss_series(frame: pd.DataFrame) -> float:
    errors = frame["y"].astype(float).to_numpy() - frame["y_pred"].astype(float).to_numpy()
    quantiles = frame["quantile"].astype(float).to_numpy()
    losses = np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return float(np.mean(losses))


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(frame[column].astype(float).mean())


def _hourly_cqr(
    frame: pd.DataFrame,
    *,
    calibration_frame: pd.DataFrame,
    calibration_method: str,
    calibration_group_by: str | None,
    calibration_min_group_size: int,
    interval_coverage_floors: dict[str, float] | None,
    risk_score_column: str,
    regime_threshold: float,
) -> pd.DataFrame:
    return postprocess_quantile_predictions(
        frame,
        monotonic=True,
        calibration_frame=calibration_frame,
        calibration_method=calibration_method,
        calibration_group_by=calibration_group_by,
        calibration_interval_coverage_floors=interval_coverage_floors,
        calibration_min_group_size=calibration_min_group_size,
        calibration_regime_score_column=risk_score_column,
        calibration_regime_threshold=float(regime_threshold),
    )


def _summarize_variant(
    frame: pd.DataFrame,
    *,
    mode: str,
    variant: str,
    risk_score_column: str,
    overlay: EventRiskTailOverlay | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "mode": mode,
        "variant": variant,
        "risk_aggregation": overlay.risk_aggregation if overlay else None,
        "risk_threshold": overlay.risk_threshold if overlay else np.nan,
        "residual_quantile": overlay.residual_quantile if overlay else np.nan,
        "uplift": overlay.uplift if overlay else 0.0,
        "target_quantiles": ",".join(str(value) for value in overlay.target_quantiles) if overlay else "",
    }
    row.update(compute_metrics(frame))
    row.update(compute_quantile_diagnostics(frame))
    row.update(compute_spike_score_diagnostics(frame, score_column=risk_score_column, threshold=0.50))
    daily_risk = _daily_risk_scores(frame, risk_score_column, overlay.risk_aggregation if overlay else "mean")
    if overlay is None:
        row["active_day_share"] = 0.0
    else:
        row["active_day_share"] = float((daily_risk >= overlay.risk_threshold).mean())
    return row


def _variant_name(
    risk_aggregation: str,
    risk_threshold_quantile: float,
    residual_quantile: float,
    max_uplift: float,
) -> str:
    threshold_label = _percent_label(float(risk_threshold_quantile))
    residual_label = _percent_label(float(residual_quantile))
    cap = float(max_uplift)
    cap_label = int(cap) if cap.is_integer() else str(cap).replace(".", "p")
    return f"overlay_{risk_aggregation}_p{threshold_label}_r{residual_label}_cap{cap_label}"


def _percent_label(value: float) -> str:
    return str(int(round(float(value) * 100.0)))


def _validate_prediction_frame(predictions: pd.DataFrame, risk_score_column: str) -> None:
    if not is_quantile_prediction_frame(predictions):
        raise ValueError("event-risk tail overlay requires a quantile prediction frame.")
    if risk_score_column not in predictions.columns:
        raise ValueError(f"event-risk tail overlay requires prediction column {risk_score_column!r}.")


def _bounded_unit_interval(value: float, name: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _daily_risk_scores(frame: pd.DataFrame, risk_score_column: str, risk_aggregation: str) -> pd.Series:
    context = frame.loc[:, ["ds", risk_score_column]].drop_duplicates("ds").copy()
    context["ds"] = pd.to_datetime(context["ds"])
    context["day"] = context["ds"].dt.floor("D")
    aggregation = str(risk_aggregation).lower()
    if aggregation == "mean":
        return context.groupby("day", sort=True)[risk_score_column].mean().astype(float)
    if aggregation == "max":
        return context.groupby("day", sort=True)[risk_score_column].max().astype(float)
    raise ValueError(f"Unsupported risk_aggregation: {risk_aggregation!r}")


def _prediction_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    quantile_predictions = predictions.copy()
    quantile_predictions["ds"] = pd.to_datetime(quantile_predictions["ds"])
    quantile_predictions["quantile"] = quantile_predictions["quantile"].astype(float)
    return quantile_predictions.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)


def _resolve_quantile_column(columns: pd.Index, target: float) -> float | None:
    for column in columns.tolist():
        if np.isclose(float(column), target):
            return float(column)
    return None
