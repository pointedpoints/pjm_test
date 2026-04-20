from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ForecastModel(ABC):
    name: str
    supports_fitted_snapshot: bool = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the model on the current training window."""

    @abstractmethod
    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with columns ds, y_pred, and optional quantile."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state to a file or directory path."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Load a persisted model instance."""
