from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.evaluation.spike_score_diagnostics import compute_spike_score_diagnostics


def test_compute_spike_score_diagnostics_summarizes_context_and_error_by_regime() -> None:
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=4, freq="h").repeat(3),
            "y": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0],
            "y_pred": [8.0, 9.0, 12.0, 18.0, 21.0, 25.0, 25.0, 30.0, 35.0, 30.0, 42.0, 50.0],
            "model": ["nhits"] * 12,
            "split": ["test"] * 12,
            "seed": [7] * 12,
            "quantile": [0.1, 0.5, 0.9] * 4,
            "metadata": ["{}"] * 12,
            "spike_score": [0.1] * 3 + [0.6] * 3 + [0.8] * 3 + [1.0] * 3,
        }
    )

    diagnostics = compute_spike_score_diagnostics(frame, threshold=0.67)

    assert diagnostics["has_spike_score"] is True
    assert diagnostics["row_count"] == 4
    assert diagnostics["non_null_rate"] == 1.0
    assert diagnostics["spike_share"] == 0.5
    assert diagnostics["normal_count"] == 2
    assert diagnostics["spike_count"] == 2
    assert diagnostics["normal_p50_mae"] == 1.0
    assert diagnostics["spike_p50_mae"] == 1.0
    assert diagnostics["spike_y_mean"] == 35.0


def test_compute_spike_score_diagnostics_marks_missing_context() -> None:
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "y": [1.0, 2.0],
            "y_pred": [1.1, 1.9],
            "quantile": [pd.NA, pd.NA],
        }
    )

    diagnostics = compute_spike_score_diagnostics(frame)

    assert diagnostics["has_spike_score"] is False
    assert diagnostics["row_count"] == 2
    assert np.isnan(diagnostics["spike_share"])
