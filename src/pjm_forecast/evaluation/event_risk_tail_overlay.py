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
