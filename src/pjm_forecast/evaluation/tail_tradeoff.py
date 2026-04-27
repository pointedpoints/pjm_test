from __future__ import annotations

import math


DEFAULT_EPSILON = 1e-9


def compute_width_adjusted_tail_tradeoff(
    *,
    baseline_q99_excess_mean: float,
    candidate_q99_excess_mean: float,
    baseline_width_98: float,
    candidate_width_98: float,
    baseline_pinball: float,
    candidate_pinball: float,
    baseline_mae: float,
    candidate_mae: float,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float | bool]:
    """Summarize whether tail improvement is bought by interval inflation."""
    width_delta = float(candidate_width_98) - float(baseline_width_98)
    tail_gain = float(baseline_q99_excess_mean) - float(candidate_q99_excess_mean)
    denominator = max(width_delta, float(epsilon))
    width98_ratio = _safe_ratio(candidate_width_98, baseline_width_98)
    pinball_delta = float(candidate_pinball) - float(baseline_pinball)
    mae_delta = float(candidate_mae) - float(baseline_mae)
    tail_gain_per_width = tail_gain / denominator
    return {
        "width98_ratio": width98_ratio,
        "width98_delta": width_delta,
        "tail_gain": tail_gain,
        "tail_gain_per_width": float(tail_gain_per_width),
        "pinball_delta": pinball_delta,
        "mae_delta": mae_delta,
        "width_inflation_flag": bool(width98_ratio > 1.8 and pinball_delta >= 0.0),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf
    return float(numerator) / float(denominator)
