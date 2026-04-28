from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecast.evaluation.event_risk_tail_overlay import (
    apply_event_risk_tail_overlay,
    evaluate_event_risk_tail_overlay_grid,
    fit_event_risk_tail_overlay,
)


def _frame(values: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    rows = []
    for ds_text, y, q99, spike_score in values:
        ds = pd.Timestamp(ds_text)
        for quantile, prediction in [(0.5, q99 - 10.0), (0.95, q99 - 5.0), (0.99, q99), (0.995, q99 + 5.0)]:
            rows.append(
                {
                    "ds": ds,
                    "y": y,
                    "y_pred": prediction,
                    "quantile": quantile,
                    "model": "nhits",
                    "split": "validation",
                    "seed": 7,
                    "metadata": "{}",
                    "spike_score": spike_score,
                }
            )
    return pd.DataFrame(rows)


def test_event_risk_tail_overlay_uses_calibration_only_for_threshold_and_uplift() -> None:
    calibration = _frame(
        [
            ("2026-01-01 00:00:00", 100.0, 100.0, 0.20),
            ("2026-01-02 00:00:00", 180.0, 100.0, 0.90),
        ]
    )
    overlay = fit_event_risk_tail_overlay(
        calibration,
        risk_score_column="spike_score",
        risk_threshold_quantile=0.5,
        risk_aggregation="mean",
        residual_quantile=0.5,
        max_uplift=50.0,
        target_quantiles=[0.99, 0.995],
    )

    assert overlay.risk_threshold == pytest.approx(0.90)
    assert overlay.uplift == pytest.approx(50.0)

    evaluation = _frame(
        [
            ("2026-02-01 00:00:00", -9999.0, 10.0, 0.20),
            ("2026-02-02 00:00:00", -9999.0, 10.0, 0.95),
        ]
    )
    adjusted = apply_event_risk_tail_overlay(evaluation, overlay)
    grid = adjusted.pivot(index="ds", columns="quantile", values="y_pred")

    assert grid.loc[pd.Timestamp("2026-02-01 00:00:00"), 0.99] == pytest.approx(10.0)
    assert grid.loc[pd.Timestamp("2026-02-02 00:00:00"), 0.99] == pytest.approx(60.0)
    assert grid.loc[pd.Timestamp("2026-02-02 00:00:00"), 0.995] == pytest.approx(65.0)


def test_evaluate_event_risk_tail_overlay_grid_reports_baseline_and_candidates() -> None:
    validation = _frame(
        [
            ("2026-01-01 00:00:00", 100.0, 100.0, 0.20),
            ("2026-01-02 00:00:00", 180.0, 100.0, 0.90),
            ("2026-01-03 00:00:00", 110.0, 100.0, 0.20),
            ("2026-01-04 00:00:00", 170.0, 100.0, 0.95),
        ]
    )
    test = _frame(
        [
            ("2026-02-01 00:00:00", 100.0, 100.0, 0.20),
            ("2026-02-02 00:00:00", 200.0, 100.0, 0.95),
        ]
    )

    result = evaluate_event_risk_tail_overlay_grid(
        validation,
        test_frame=test,
        validation_holdout_days=2,
        risk_threshold_quantiles=[0.5],
        residual_quantiles=[0.5],
        max_uplifts=[50.0],
        target_quantiles=[0.99, 0.995],
        interval_coverage_floors=None,
    )

    validation_variants = set(result.validation_summary["variant"])
    test_variants = set(result.test_summary["variant"])

    assert {"hour_cqr", "overlay_mean_p50_r50_cap50"} <= validation_variants
    assert validation_variants == test_variants
    assert {"q99_exceedance_rate", "q99_excess_mean", "active_day_share", "uplift"} <= set(
        result.validation_summary.columns
    )
