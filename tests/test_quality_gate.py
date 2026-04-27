from __future__ import annotations

import numpy as np

from pjm_forecast.evaluation.quality_gate import QualityDecision, QualityMetrics, evaluate_quality_gate
from pjm_forecast.evaluation.tail_tradeoff import compute_width_adjusted_tail_tradeoff


def test_width_adjusted_tail_tradeoff_penalizes_width_inflation_without_pinball_gain() -> None:
    tradeoff = compute_width_adjusted_tail_tradeoff(
        baseline_q99_excess_mean=1.0,
        candidate_q99_excess_mean=0.5,
        baseline_width_98=50.0,
        candidate_width_98=100.0,
        baseline_pinball=3.0,
        candidate_pinball=3.1,
        baseline_mae=10.0,
        candidate_mae=10.0,
    )

    assert tradeoff["tail_gain"] == 0.5
    assert tradeoff["width98_ratio"] == 2.0
    assert tradeoff["width_inflation_flag"] is True
    assert np.isclose(tradeoff["tail_gain_per_width"], 0.01)


def test_quality_gate_promotes_candidate_only_when_main_and_tail_metrics_improve() -> None:
    baseline = QualityMetrics(
        mae=10.0,
        pinball=3.0,
        q99_exceedance_rate=0.05,
        q99_excess_mean=1.0,
        worst_q99_underprediction=100.0,
        width_98=50.0,
        normal_p50_mae=6.0,
    )
    candidate = QualityMetrics(
        mae=9.8,
        pinball=2.9,
        q99_exceedance_rate=0.03,
        q99_excess_mean=0.8,
        worst_q99_underprediction=90.0,
        width_98=55.0,
        normal_p50_mae=5.9,
    )

    result = evaluate_quality_gate(baseline=baseline, candidate=candidate)

    assert result.decision == QualityDecision.PROMOTE
    assert result.mae_delta < 0.0
    assert result.pinball_delta < 0.0
    assert result.tail_gain > 0.0


def test_quality_gate_keeps_tail_signal_as_context_when_main_metrics_worsen() -> None:
    baseline = QualityMetrics(
        mae=10.0,
        pinball=3.0,
        q99_exceedance_rate=0.05,
        q99_excess_mean=1.0,
        worst_q99_underprediction=100.0,
        width_98=50.0,
    )
    candidate = QualityMetrics(
        mae=11.0,
        pinball=3.2,
        q99_exceedance_rate=0.02,
        q99_excess_mean=0.7,
        worst_q99_underprediction=80.0,
        width_98=60.0,
    )

    result = evaluate_quality_gate(baseline=baseline, candidate=candidate)

    assert result.decision == QualityDecision.CONTEXT_ONLY
    assert "main-model gate failed" in result.reason

