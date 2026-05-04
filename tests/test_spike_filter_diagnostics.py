from __future__ import annotations

import pandas as pd

from pjm_forecast.evaluation.spike_filter_diagnostics import compute_retrain_spike_filter_diagnostics
from pjm_forecast.spike_filter import SpikeFilterConfig


def _feature_frame(days: int = 90) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["PJM_COMED"] * (days * 24),
            "ds": pd.date_range("2024-01-01", periods=days * 24, freq="h"),
            "y": [20.0] * (days * 24),
        }
    )
    frame.loc[frame["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"] = 500.0
    return frame


def test_retrain_spike_filter_diagnostics_reports_anchor_windows() -> None:
    feature_df = _feature_frame()
    forecast_days = [
        pd.Timestamp("2024-03-11 00:00:00"),
        pd.Timestamp("2024-03-12 00:00:00"),
        pd.Timestamp("2024-03-18 00:00:00"),
    ]

    diagnostics = compute_retrain_spike_filter_diagnostics(
        feature_df=feature_df,
        forecast_days=forecast_days,
        rolling_window_days=80,
        retrain_weekday=0,
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    assert list(diagnostics["forecast_day"]) == [pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-18")]
    assert diagnostics["rows"].gt(0).all()
    assert diagnostics["spike_count"].ge(1).any()
    assert diagnostics["max_spike_residual"].gt(0).any()
