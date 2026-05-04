from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpikeFilterConfig:
    window_observations: int = 365
    min_history: int = 60
    quantile: float = 0.95
    fallback_quantile: float = 0.975
    iqr_multiplier: float = 3.0
    iqr_epsilon: float = 1e-8
    target_column: str = "y"


def apply_spike_filter(frame: pd.DataFrame, config: SpikeFilterConfig | None = None) -> pd.DataFrame:
    cfg = config or SpikeFilterConfig()
    required = ["ds", cfg.target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"spike filter input is missing required columns: {missing}")
    if cfg.window_observations <= 0:
        raise ValueError("window_observations must be positive.")
    if cfg.min_history <= 0:
        raise ValueError("min_history must be positive.")
    if not 0.0 < cfg.quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    if not 0.0 < cfg.fallback_quantile < 1.0:
        raise ValueError("fallback_quantile must be in (0, 1).")

    result = frame.copy()
    result["ds"] = pd.to_datetime(result["ds"])
    original_index = result.index
    result = result.sort_values("ds").reset_index(drop=False).rename(columns={"index": "_original_index"})
    result["_hour"] = result["ds"].dt.hour

    thresholds = pd.Series(np.nan, index=result.index, dtype=float)
    for _, group in result.groupby("_hour", sort=False):
        values = group[cfg.target_column].astype(float)
        shifted = values.shift(1)
        rolling = shifted.rolling(window=cfg.window_observations, min_periods=cfg.min_history)
        q_low = rolling.quantile(0.25)
        q_high = rolling.quantile(0.75)
        iqr = q_high - q_low
        robust_threshold = rolling.quantile(cfg.quantile) + cfg.iqr_multiplier * iqr
        fallback_threshold = rolling.quantile(cfg.fallback_quantile)
        threshold = robust_threshold.where(iqr > cfg.iqr_epsilon, fallback_threshold)
        thresholds.loc[group.index] = threshold

    y = result[cfg.target_column].astype(float)
    result["spike_threshold"] = thresholds
    result["is_training_spike"] = result["spike_threshold"].notna() & (y > result["spike_threshold"])
    result["y_train_clean"] = y.where(~result["is_training_spike"], result["spike_threshold"])
    result["spike_residual"] = y - result["y_train_clean"]
    result.loc[~result["is_training_spike"], "spike_residual"] = 0.0

    result = result.drop(columns=["_hour"]).set_index("_original_index").loc[original_index]
    result.index.name = None
    return result
