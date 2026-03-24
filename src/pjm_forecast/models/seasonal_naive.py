from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .base import ForecastModel


@dataclass
class SeasonalNaiveModel(ForecastModel):
    seasonal_lag_hours: int = 168
    name: str = "seasonal_naive"

    def fit(self, train_df: pd.DataFrame) -> None:
        self._history = train_df.set_index("ds")["y"].copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        self._history = history_df.set_index("ds")["y"].copy()

        predictions = []
        for timestamp in future_df["ds"]:
            lag_timestamp = timestamp - pd.Timedelta(hours=self.seasonal_lag_hours)
            if lag_timestamp not in self._history.index:
                raise KeyError(f"Missing seasonal lag {self.seasonal_lag_hours} for timestamp {timestamp}.")
            predictions.append(self._history.loc[lag_timestamp])

        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": predictions})

    def save(self, path: Path) -> None:
        payload = {"seasonal_lag_hours": self.seasonal_lag_hours}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SeasonalNaiveModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
