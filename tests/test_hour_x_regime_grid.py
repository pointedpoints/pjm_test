from __future__ import annotations

import pandas as pd

from pjm_forecast.evaluation.hour_x_regime_grid import evaluate_hour_x_regime_threshold_grid


def _frame(*, split: str, days: int) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=24 * days, freq="h")
    for index, ts in enumerate(timestamps):
        y = 20.0 + float(index % 24)
        spike_score = 0.8 if ts.hour >= 18 else 0.2
        for quantile, offset in [(0.1, -2.0), (0.5, 0.0), (0.9, 2.0)]:
            rows.append(
                {
                    "ds": ts,
                    "y": y,
                    "y_pred": y + offset,
                    "model": "nhits",
                    "split": split,
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                    "spike_score": spike_score,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_hour_x_regime_threshold_grid_outputs_baseline_and_threshold_rows() -> None:
    validation = _frame(split="validation", days=4)
    test = _frame(split="test", days=2)

    result = evaluate_hour_x_regime_threshold_grid(
        validation,
        test_frame=test,
        thresholds=[0.5, 0.67],
        validation_holdout_days=2,
        min_group_size=1,
    )

    validation_variants = set(result.validation_summary["variant"])
    test_variants = set(result.test_summary["variant"])

    assert {"raw_monotonic", "hour_cqr", "hour_regime_cqr_t50", "hour_regime_cqr_t67"} <= validation_variants
    assert validation_variants == test_variants
    assert {"pinball", "q50_mae", "q99_exceedance_rate", "spike_share"} <= set(result.validation_summary.columns)
