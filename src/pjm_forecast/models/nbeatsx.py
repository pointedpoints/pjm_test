from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastModel


@dataclass
class AsinhQuantileScaler:
    q_: float | None = None
    max_abs_value_: float = 8.0

    def fit(self, values: pd.Series) -> "AsinhQuantileScaler":
        clean = values.astype(float).dropna()
        if clean.empty:
            raise ValueError("AsinhQuantileScaler requires at least one non-null value.")

        q = float(np.quantile(np.abs(clean), 0.95))
        if q <= 1e-8:
            q = 1.0
        self.q_ = q
        return self

    def transform_series(self, values: pd.Series) -> pd.Series:
        if self.q_ is None:
            raise RuntimeError("AsinhQuantileScaler must be fit before transform.")
        transformed = np.arcsinh(values.astype(float) / self.q_)
        return pd.Series(transformed, index=values.index, dtype=float)

    def inverse_transform_array(self, values: np.ndarray) -> np.ndarray:
        if self.q_ is None:
            raise RuntimeError("AsinhQuantileScaler must be fit before inverse transform.")
        clipped = np.clip(values, -self.max_abs_value_, self.max_abs_value_)
        return np.sinh(clipped) * self.q_


@dataclass
class ZScoreScaler:
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "ZScoreScaler":
        self.mean_.clear()
        self.std_.clear()
        for column in columns:
            clean = frame[column].astype(float).dropna()
            if clean.empty:
                self.mean_[column] = 0.0
                self.std_[column] = 1.0
                continue
            mean = float(clean.mean())
            std = float(clean.std(ddof=0))
            if std <= 1e-8:
                std = 1.0
            self.mean_[column] = mean
            self.std_[column] = std
        return self

    def transform_frame(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        transformed = frame.copy()
        for column in columns:
            transformed[column] = (transformed[column].astype(float) - self.mean_[column]) / self.std_[column]
        return transformed


@dataclass
class NBEATSxModel(ForecastModel):
    h: int
    freq: str
    input_size: int
    max_steps: int
    learning_rate: float
    batch_size: int
    dropout_prob_theta: float
    scaler_type: str
    stack_types: list[str]
    mlp_units: list[list[int]]
    futr_exog_list: list[str]
    hist_exog_list: list[str]
    target_transform: str = "identity"
    exog_scaler: str = "identity"
    early_stop_patience_steps: int = -1
    val_check_steps: int = 100
    validation_size: int = 168
    windows_batch_size: int = 1024
    random_seed: int = 7
    name: str = "nbeatsx"
    _nf: Any = field(default=None, init=False, repr=False)
    _target_scaler: AsinhQuantileScaler | None = field(default=None, init=False, repr=False)
    _exog_scaler: ZScoreScaler | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from neuralforecast import NeuralForecast  # type: ignore
            from neuralforecast.models import NBEATSx  # type: ignore
        except ImportError as exc:
            raise ImportError("NBEATSxModel requires neuralforecast and torch to be installed.") from exc
        self._neuralforecast_cls = NeuralForecast
        self._nbeatsx_cls = NBEATSx

    def _price_columns(self, frame: pd.DataFrame) -> list[str]:
        return [column for column in frame.columns if column == "y" or column.startswith("price_lag_")]

    def _scaled_exog_columns(self, frame: pd.DataFrame) -> list[str]:
        protected_columns = {
            "unique_id",
            "ds",
            "y",
            "is_weekend",
            "is_holiday",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "month_sin",
            "month_cos",
        }
        return [
            column
            for column in [*self.futr_exog_list, *self.hist_exog_list]
            if column in frame.columns and column not in protected_columns and not column.startswith("price_lag_")
        ]

    def _transform_frame(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        transformed = frame.copy()
        if self.target_transform not in {"identity", "asinh_q95"}:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")
        if self.exog_scaler not in {"identity", "zscore"}:
            raise ValueError(f"Unsupported exog_scaler: {self.exog_scaler}")

        if self.target_transform == "asinh_q95":
            if fit or self._target_scaler is None:
                self._target_scaler = AsinhQuantileScaler().fit(transformed["y"])
            for column in self._price_columns(transformed):
                transformed[column] = self._target_scaler.transform_series(transformed[column])

        exog_columns = self._scaled_exog_columns(transformed)
        if self.exog_scaler == "zscore" and exog_columns:
            if fit or self._exog_scaler is None:
                self._exog_scaler = ZScoreScaler().fit(transformed, exog_columns)
            transformed = self._exog_scaler.transform_frame(transformed, exog_columns)
        return transformed

    def fit(self, train_df: pd.DataFrame) -> None:
        train_df = self._transform_frame(train_df, fit=True)
        model = self._nbeatsx_cls(
            h=self.h,
            input_size=self.input_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            dropout_prob_theta=self.dropout_prob_theta,
            scaler_type=self.scaler_type,
            stack_types=self.stack_types,
            mlp_units=self.mlp_units,
            futr_exog_list=self.futr_exog_list,
            hist_exog_list=self.hist_exog_list,
            early_stop_patience_steps=self.early_stop_patience_steps,
            val_check_steps=self.val_check_steps,
            windows_batch_size=self.windows_batch_size,
            random_seed=self.random_seed,
            logger=False,
            enable_progress_bar=False,
        )
        self._nf = self._neuralforecast_cls(models=[model], freq=self.freq)
        fit_kwargs = {"df": train_df}
        if self.early_stop_patience_steps > 0:
            fit_kwargs["val_size"] = self.validation_size
        self._nf.fit(**fit_kwargs)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("NBEATSxModel must be fit before predict.")

        history_df = self._transform_frame(history_df, fit=False)
        future_df = self._transform_frame(future_df, fit=False)

        expected_future = self._nf.make_future_dataframe(df=history_df)
        future_inputs = future_df.loc[:, ["unique_id", "ds", *self.futr_exog_list]].copy()
        futr_df = expected_future.merge(future_inputs, on=["unique_id", "ds"], how="left")
        if futr_df[self.futr_exog_list].isna().any().any():
            raise ValueError("Missing future exogenous values after aligning to NeuralForecast future grid.")

        prediction_df = self._nf.predict(df=history_df, futr_df=futr_df)
        model_column = [column for column in prediction_df.columns if column not in {"unique_id", "ds"}][0]
        result = prediction_df.rename(columns={model_column: "y_pred"}).loc[:, ["ds", "y_pred"]]
        if self._target_scaler is not None and self.target_transform == "asinh_q95":
            result["y_pred"] = self._target_scaler.inverse_transform_array(result["y_pred"].to_numpy())
        return result

    def save(self, path: Path) -> None:
        payload = {
            "h": self.h,
            "freq": self.freq,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout_prob_theta": self.dropout_prob_theta,
            "scaler_type": self.scaler_type,
            "stack_types": self.stack_types,
            "mlp_units": self.mlp_units,
            "futr_exog_list": self.futr_exog_list,
            "hist_exog_list": self.hist_exog_list,
            "target_transform": self.target_transform,
            "exog_scaler": self.exog_scaler,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "validation_size": self.validation_size,
            "windows_batch_size": self.windows_batch_size,
            "random_seed": self.random_seed,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NBEATSxModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
