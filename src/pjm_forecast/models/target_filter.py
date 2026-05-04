from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter

from .base import ForecastModel


@dataclass
class SpikeFilteredTargetModel(ForecastModel):
    base_model: ForecastModel
    filter_config: SpikeFilterConfig = field(default_factory=SpikeFilterConfig)
    name: str = "spike_filtered_target"
    supports_fitted_snapshot: bool = False
    last_filter_diagnostics: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.name = getattr(self.base_model, "name", self.name)
        self.supports_fitted_snapshot = bool(getattr(self.base_model, "supports_fitted_snapshot", False))

    def fit(self, train_df: pd.DataFrame) -> None:
        filtered = apply_spike_filter(train_df, self.filter_config)
        self.last_filter_diagnostics = summarize_spike_filter(filtered)
        fit_frame = train_df.copy()
        fit_frame["y"] = filtered["y_train_clean"].astype(float)
        self.base_model.fit(fit_frame)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        return self.base_model.predict(history_df=history_df, future_df=future_df)

    def save(self, path: Path) -> None:
        self.base_model.save(path)

    @classmethod
    def load(cls, path: Path) -> "SpikeFilteredTargetModel":
        raise NotImplementedError("Load the wrapped base model through its own adapter.")


def summarize_spike_filter(filtered: pd.DataFrame) -> dict[str, float]:
    spike_mask = filtered["is_training_spike"].astype(bool)
    residual = filtered["spike_residual"].astype(float)
    rows = float(len(filtered))
    spike_count = float(spike_mask.sum())
    return {
        "rows": rows,
        "spike_count": spike_count,
        "spike_share": 0.0 if rows == 0.0 else spike_count / rows,
        "mean_spike_residual": 0.0 if spike_count == 0.0 else float(residual.loc[spike_mask].mean()),
        "max_spike_residual": 0.0 if spike_count == 0.0 else float(residual.loc[spike_mask].max()),
    }
