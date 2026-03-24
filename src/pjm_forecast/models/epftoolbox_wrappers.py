from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastModel


def _to_epftoolbox_frame(history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    available = pd.concat([history_df, future_df], axis=0).copy()
    available = available.rename(
        columns={
            "y": "Price",
            "system_load_forecast": "Exogenous 1",
            "zonal_load_forecast": "Exogenous 2",
        }
    )
    available.loc[future_df.index, "Price"] = np.nan
    available = available.set_index("ds")
    return available.loc[:, ["Price", "Exogenous 1", "Exogenous 2"]]


@dataclass
class LEARModel(ForecastModel):
    calibration_window_days: int
    name: str = "lear"

    def __post_init__(self) -> None:
        try:
            from epftoolbox.models import LEAR  # type: ignore
        except ImportError as exc:
            raise ImportError("LEARModel requires epftoolbox to be installed.") from exc
        self._model = LEAR(calibration_window=self.calibration_window_days)

    def fit(self, train_df: pd.DataFrame) -> None:
        self._latest_train = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        available_df = _to_epftoolbox_frame(history_df, future_df)
        next_day = future_df["ds"].min()
        prediction = self._model.recalibrate_and_forecast_next_day(
            df=available_df,
            calibration_window=self.calibration_window_days,
            next_day_date=next_day,
        )
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": np.asarray(prediction).reshape(-1)})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"calibration_window_days": self.calibration_window_days}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LEARModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class DNNModel(ForecastModel):
    experiment_id: int
    hyperparameter_dir: str
    nlayers: int
    dataset: str
    years_test: int
    shuffle_train: bool
    data_augmentation: bool
    calibration_window_years: int
    name: str = "dnn"

    def __post_init__(self) -> None:
        try:
            from epftoolbox.models import DNN  # type: ignore
        except ImportError as exc:
            raise ImportError("DNNModel requires epftoolbox to be installed.") from exc
        self._model = DNN(
            experiment_id=self.experiment_id,
            path_hyperparameter_folder=self.hyperparameter_dir,
            nlayers=self.nlayers,
            dataset=self.dataset,
            years_test=self.years_test,
            shuffle_train=int(self.shuffle_train),
            data_augmentation=int(self.data_augmentation),
            calibration_window=self.calibration_window_years,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        self._latest_train = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        available_df = _to_epftoolbox_frame(history_df, future_df)
        next_day = future_df["ds"].min()
        prediction = self._model.recalibrate_and_forecast_next_day(df=available_df, next_day_date=next_day)
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": np.asarray(prediction).reshape(-1)})

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "hyperparameter_dir": self.hyperparameter_dir,
            "nlayers": self.nlayers,
            "dataset": self.dataset,
            "years_test": self.years_test,
            "shuffle_train": self.shuffle_train,
            "data_augmentation": self.data_augmentation,
            "calibration_window_years": self.calibration_window_years,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DNNModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

