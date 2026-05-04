from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter


def _hourly_frame(days: int = 90) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01 00:00:00")
    for offset in range(days * 24):
        ds = start + pd.Timedelta(hours=offset)
        base = 20.0 + float(ds.hour)
        rows.append({"unique_id": "PJM_COMED", "ds": ds, "y": base})
    return pd.DataFrame(rows)


def test_spike_filter_caps_only_after_minimum_prior_hour_history() -> None:
    frame = _hourly_frame(days=70)
    spike_ts = pd.Timestamp("2024-03-05 17:00:00")
    frame.loc[frame["ds"].eq(spike_ts), "y"] = 500.0

    result = apply_spike_filter(
        frame,
        SpikeFilterConfig(window_observations=365, min_history=60, quantile=0.95, iqr_multiplier=3.0),
    )

    spike_row = result.loc[result["ds"].eq(spike_ts)].iloc[0]
    assert bool(spike_row["is_training_spike"])
    assert spike_row["y_train_clean"] < spike_row["y"]
    assert spike_row["spike_residual"] == spike_row["y"] - spike_row["y_train_clean"]

    first_17 = result.loc[result["ds"].dt.hour.eq(17)].iloc[0]
    assert not bool(first_17["is_training_spike"])
    assert np.isnan(first_17["spike_threshold"])
    assert first_17["y_train_clean"] == first_17["y"]


def test_spike_filter_is_causal_with_later_extreme_values() -> None:
    prefix = _hourly_frame(days=70)
    target_ts = pd.Timestamp("2024-03-01 17:00:00")
    prefix.loc[prefix["ds"].eq(target_ts), "y"] = 120.0

    full = pd.concat(
        [
            prefix,
            pd.DataFrame(
                {
                    "unique_id": ["PJM_COMED"] * 24,
                    "ds": pd.date_range("2024-03-11 00:00:00", periods=24, freq="h"),
                    "y": [10000.0] * 24,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("ds").reset_index(drop=True)

    config = SpikeFilterConfig(window_observations=365, min_history=60)
    prefix_row = apply_spike_filter(prefix, config).loc[lambda df: df["ds"].eq(target_ts)].iloc[0]
    full_row = apply_spike_filter(full, config).loc[lambda df: df["ds"].eq(target_ts)].iloc[0]

    assert prefix_row["spike_threshold"] == full_row["spike_threshold"]
    assert prefix_row["y_train_clean"] == full_row["y_train_clean"]
    assert bool(prefix_row["is_training_spike"]) == bool(full_row["is_training_spike"])
