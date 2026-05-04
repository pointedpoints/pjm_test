from __future__ import annotations

import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter


def compute_retrain_spike_filter_diagnostics(
    *,
    feature_df: pd.DataFrame,
    forecast_days: list[pd.Timestamp],
    rolling_window_days: int,
    retrain_weekday: int,
    filter_config: SpikeFilterConfig,
) -> pd.DataFrame:
    rows = []
    for index, forecast_day in enumerate(forecast_days):
        if index > 0 and forecast_day.weekday() != retrain_weekday:
            continue
        history_end = forecast_day - pd.Timedelta(hours=1)
        window_start = forecast_day - pd.Timedelta(days=rolling_window_days)
        history_df = feature_df.loc[(feature_df["ds"] >= window_start) & (feature_df["ds"] <= history_end)].copy()
        filtered = apply_spike_filter(history_df, filter_config)
        spike_mask = filtered["is_training_spike"].astype(bool)
        residual = filtered["spike_residual"].astype(float)
        rows.append(
            {
                "forecast_day": forecast_day.normalize(),
                "window_start": window_start,
                "window_end": history_end,
                "rows": len(filtered),
                "spike_count": int(spike_mask.sum()),
                "spike_share": 0.0 if len(filtered) == 0 else float(spike_mask.mean()),
                "mean_spike_residual": 0.0 if not spike_mask.any() else float(residual.loc[spike_mask].mean()),
                "max_spike_residual": 0.0 if not spike_mask.any() else float(residual.loc[spike_mask].max()),
            }
        )
    return pd.DataFrame(rows)
