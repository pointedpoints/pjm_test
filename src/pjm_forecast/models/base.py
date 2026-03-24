from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ForecastModel(ABC):
    name: str

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the model on the current training window."""

    @abstractmethod
    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with columns ds and y_pred."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist lightweight model metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Load a persisted model instance."""

