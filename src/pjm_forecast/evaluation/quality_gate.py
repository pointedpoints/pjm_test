from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from .tail_tradeoff import compute_width_adjusted_tail_tradeoff


class QualityDecision(StrEnum):
    PROMOTE = "PROMOTE"
    TAIL_ONLY = "TAIL_ONLY"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    REFERENCE = "REFERENCE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class QualityMetrics:
    mae: float
    pinball: float
    q99_exceedance_rate: float
    q99_excess_mean: float
    worst_q99_underprediction: float
    width_98: float
    normal_p50_mae: float | None = None
    extreme_p50_mae: float | None = None
    daily_max_q99_gap: float | None = None


@dataclass(frozen=True)
class QualityGateResult:
    decision: QualityDecision
    reason: str
    width98_ratio: float
    tail_gain: float
    tail_gain_per_width: float
    pinball_delta: float
    mae_delta: float


def evaluate_quality_gate(
    *,
    baseline: QualityMetrics,
    candidate: QualityMetrics,
    validation_direction_consistent: bool = True,
    allowed_feature_contract: bool = True,
    main_model_candidate: bool = True,
) -> QualityGateResult:
    tradeoff = compute_width_adjusted_tail_tradeoff(
        baseline_q99_excess_mean=baseline.q99_excess_mean,
        candidate_q99_excess_mean=candidate.q99_excess_mean,
        baseline_width_98=baseline.width_98,
        candidate_width_98=candidate.width_98,
        baseline_pinball=baseline.pinball,
        candidate_pinball=candidate.pinball,
        baseline_mae=baseline.mae,
        candidate_mae=candidate.mae,
    )
    reason_parts: list[str] = []
    if not allowed_feature_contract:
        return _result(QualityDecision.REJECT, "feature availability contract failed", tradeoff)
    if not validation_direction_consistent:
        return _result(QualityDecision.REJECT, "validation/test direction is inconsistent", tradeoff)

    mae_delta = float(tradeoff["mae_delta"])
    pinball_delta = float(tradeoff["pinball_delta"])
    width98_ratio = float(tradeoff["width98_ratio"])
    tail_gain = float(tradeoff["tail_gain"])
    normal_delta = _delta(candidate.normal_p50_mae, baseline.normal_p50_mae)

    if bool(tradeoff["width_inflation_flag"]):
        reason_parts.append("width98 ratio exceeds 1.8 without pinball improvement")
    if mae_delta > 0.0:
        reason_parts.append("MAE worsens")
    if pinball_delta > 0.0:
        reason_parts.append("pinball worsens")
    if normal_delta is not None and normal_delta > 0.0:
        reason_parts.append("normal-regime P50 worsens")

    if reason_parts and main_model_candidate:
        if tail_gain > 0.0 and not bool(tradeoff["width_inflation_flag"]) and pinball_delta <= 0.0:
            return _result(QualityDecision.TAIL_ONLY, "tail improves but main-model gate failed: " + "; ".join(reason_parts), tradeoff)
        if tail_gain > 0.0:
            return _result(QualityDecision.CONTEXT_ONLY, "main-model gate failed; keep only as context: " + "; ".join(reason_parts), tradeoff)
        return _result(QualityDecision.REJECT, "; ".join(reason_parts), tradeoff)

    if width98_ratio > 1.8 and pinball_delta >= 0.0:
        return _result(QualityDecision.REJECT, "tail gain is dominated by width inflation", tradeoff)
    if main_model_candidate and mae_delta <= 0.0 and pinball_delta <= 0.0 and tail_gain > 0.0:
        return _result(QualityDecision.PROMOTE, "MAE, pinball, and q99 excess improve without gate violations", tradeoff)
    if tail_gain > 0.0 and pinball_delta <= 0.0:
        return _result(QualityDecision.TAIL_ONLY, "tail improves without pinball degradation, but not a main-model candidate", tradeoff)
    if tail_gain > 0.0:
        return _result(QualityDecision.CONTEXT_ONLY, "tail signal exists but main metrics are not promotion-safe", tradeoff)
    return _result(QualityDecision.REFERENCE, "no promotion signal; keep as comparison reference", tradeoff)


def reference_result(reason: str = "fixed comparison baseline") -> QualityGateResult:
    tradeoff = {
        "width98_ratio": 1.0,
        "tail_gain": 0.0,
        "tail_gain_per_width": 0.0,
        "pinball_delta": 0.0,
        "mae_delta": 0.0,
    }
    return _result(QualityDecision.REFERENCE, reason, tradeoff)


def _result(decision: QualityDecision, reason: str, tradeoff: dict[str, float | bool]) -> QualityGateResult:
    return QualityGateResult(
        decision=decision,
        reason=reason,
        width98_ratio=_finite_float(tradeoff["width98_ratio"]),
        tail_gain=_finite_float(tradeoff["tail_gain"]),
        tail_gain_per_width=_finite_float(tradeoff["tail_gain_per_width"]),
        pinball_delta=_finite_float(tradeoff["pinball_delta"]),
        mae_delta=_finite_float(tradeoff["mae_delta"]),
    )


def _delta(candidate_value: float | None, baseline_value: float | None) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    if math.isnan(float(candidate_value)) or math.isnan(float(baseline_value)):
        return None
    return float(candidate_value) - float(baseline_value)


def _finite_float(value: float | bool) -> float:
    return float(value)
