from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastModel


_RESERVED_COLUMNS = {"unique_id", "ds", "y"}


def _normalize_quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    quantiles = sorted({float(value) for value in values})
    invalid = [value for value in quantiles if value <= 0.0 or value >= 1.0]
    if invalid:
        raise ValueError(f"Quantiles must be within (0, 1): {invalid}")
    return quantiles


@dataclass
class _BaseTreeQuantileModel(ForecastModel):
    feature_columns: list[str]
    quantiles: list[float]
    random_seed: int = 7
    model_params: dict[str, Any] = field(default_factory=dict)
    name: str = "tree_quantile"

    def __post_init__(self) -> None:
        self.quantiles = _normalize_quantiles(self.quantiles)
        self.feature_columns = [str(column) for column in self.feature_columns if str(column) not in _RESERVED_COLUMNS]
        if not self.feature_columns:
            raise ValueError(f"{self.name} requires at least one feature column.")
        self._models: dict[float, Any] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        x_train, y_train = self._xy(train_df)
        self._models = {}
        for quantile in self.quantiles:
            model = self._build_estimator(quantile)
            model.fit(x_train, y_train)
            self._models[quantile] = model

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        del history_df
        if not self._models:
            raise RuntimeError(f"{self.name} must be fitted before predict().")

        x_future = self._x(future_df)
        if not self.quantiles:
            raise RuntimeError(f"{self.name} requires configured quantiles for prediction.")

        frames = []
        for quantile in self.quantiles:
            predictions = np.asarray(self._models[quantile].predict(x_future), dtype=float).reshape(-1)
            frames.append(
                pd.DataFrame(
                    {
                        "ds": future_df["ds"].to_numpy(),
                        "quantile": quantile,
                        "y_pred": predictions,
                    }
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True).sort_values(["ds", "quantile"]).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "quantiles": self.quantiles,
            "random_seed": self.random_seed,
            "model_params": self.model_params,
            "name": self.name,
            "models": self._models,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        models = payload.pop("models", {})
        instance = cls(**payload)
        instance._models = models
        return instance

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        self._require_columns(frame, include_target=True)
        x_train = frame.loc[:, self.feature_columns].copy()
        y_train = frame["y"].astype(float).copy()
        return x_train, y_train

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(frame, include_target=False)
        return frame.loc[:, self.feature_columns].copy()

    def _require_columns(self, frame: pd.DataFrame, *, include_target: bool) -> None:
        required = list(self.feature_columns)
        if include_target:
            required.append("y")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.name} input frame is missing required columns: {missing}")
        if frame[required].isna().any().any():
            counts = frame[required].isna().sum()
            raise ValueError(f"{self.name} input frame contains missing values: {counts[counts > 0].to_dict()}")

    def _build_estimator(self, quantile: float) -> Any:
        raise NotImplementedError


@dataclass
class LightGBMQuantileModel(_BaseTreeQuantileModel):
    name: str = "lightgbm_quantile"

    def _build_estimator(self, quantile: float) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBMQuantileModel requires lightgbm to be installed.") from exc

        params = {
            "objective": "quantile",
            "alpha": quantile,
            "random_state": self.random_seed,
            "verbosity": -1,
            **self.model_params,
        }
        return LGBMRegressor(**params)


@dataclass
class XGBoostQuantileModel(_BaseTreeQuantileModel):
    name: str = "xgboost_quantile"

    def _build_estimator(self, quantile: float) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoostQuantileModel requires xgboost to be installed.") from exc

        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": quantile,
            "random_state": self.random_seed,
            "tree_method": "hist",
            "verbosity": 0,
            **self.model_params,
        }
        return XGBRegressor(**params)
