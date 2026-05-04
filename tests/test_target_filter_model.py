from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.models.base import ForecastModel
from pjm_forecast.models.target_filter import SpikeFilteredTargetModel
from pjm_forecast.spike_filter import SpikeFilterConfig


class RecordingModel(ForecastModel):
    name = "recording"

    def __init__(self) -> None:
        self.fit_frame: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.fit_frame = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ds": future_df["ds"], "y_pred": 1.0})

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "RecordingModel":
        raise NotImplementedError


def _train_frame() -> pd.DataFrame:
    train = pd.DataFrame(
        {
            "unique_id": ["PJM_COMED"] * (70 * 24),
            "ds": pd.date_range("2024-01-01", periods=70 * 24, freq="h"),
            "y": [20.0] * (70 * 24),
        }
    )
    train.loc[train["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"] = 500.0
    return train


def test_spike_filtered_target_model_replaces_only_training_y() -> None:
    train = _train_frame()
    future = train.tail(24).copy()
    base = RecordingModel()
    model = SpikeFilteredTargetModel(
        base_model=base,
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    model.fit(train)
    predictions = model.predict(history_df=train, future_df=future)

    assert base.fit_frame is not None
    original_spike = train.loc[train["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"].iloc[0]
    fitted_spike = base.fit_frame.loc[base.fit_frame["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"].iloc[0]
    assert fitted_spike < original_spike
    assert "y_train_clean" not in base.fit_frame.columns
    assert list(predictions.columns) == ["ds", "y_pred"]


def test_spike_filtered_target_model_exposes_last_diagnostics() -> None:
    train = _train_frame()
    model = SpikeFilteredTargetModel(
        base_model=RecordingModel(),
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    model.fit(train)

    diagnostics = model.last_filter_diagnostics
    assert diagnostics["rows"] == float(len(train))
    assert diagnostics["spike_count"] >= 1.0
    assert diagnostics["max_spike_residual"] > 0.0
