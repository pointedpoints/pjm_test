from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .base import ForecastModel


@dataclass
class NBEATSxModel(ForecastModel):
    h: int
    input_size: int
    max_steps: int
    learning_rate: float
    batch_size: int
    dropout_prob_theta: float
    scaler_type: str
    stack_types: list[str]
    mlp_units: list[list[int]]
    futr_exog_list: list[str]
    random_seed: int = 7
    name: str = "nbeatsx"
    _nf: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from neuralforecast import NeuralForecast  # type: ignore
            from neuralforecast.models import NBEATSx  # type: ignore
        except ImportError as exc:
            raise ImportError("NBEATSxModel requires neuralforecast and torch to be installed.") from exc
        self._neuralforecast_cls = NeuralForecast
        self._nbeatsx_cls = NBEATSx

    def fit(self, train_df: pd.DataFrame) -> None:
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
            random_seed=self.random_seed,
        )
        self._nf = self._neuralforecast_cls(models=[model], freq="H")
        self._nf.fit(df=train_df)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("NBEATSxModel must be fit before predict.")
        prediction_df = self._nf.predict(futr_df=future_df)
        model_column = [column for column in prediction_df.columns if column not in {"unique_id", "ds"}][0]
        return prediction_df.rename(columns={model_column: "y_pred"}).loc[:, ["ds", "y_pred"]]

    def save(self, path: Path) -> None:
        payload = {
            "h": self.h,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout_prob_theta": self.dropout_prob_theta,
            "scaler_type": self.scaler_type,
            "stack_types": self.stack_types,
            "mlp_units": self.mlp_units,
            "futr_exog_list": self.futr_exog_list,
            "random_seed": self.random_seed,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NBEATSxModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

