from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pjm_forecast.prepared_data import default_nbeatsx_protected_exog_columns

from .base import ForecastModel


@dataclass
class _FittedMemberState:
    nf: Any
    target_scaler: "AsinhQuantileScaler | None"
    exog_scaler: "ZScoreScaler | None"


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
    protected_exog_columns: list[str] = field(default_factory=default_nbeatsx_protected_exog_columns)
    target_transform: str = "identity"
    exog_scaler: str = "identity"
    early_stop_patience_steps: int = -1
    val_check_steps: int = 100
    validation_size: int = 168
    windows_batch_size: int = 1024
    ensemble_aggregation: str = "mean"
    ensemble_members: list[dict[str, Any]] = field(default_factory=list)
    random_seed: int = 7
    name: str = "nbeatsx"
    supports_fitted_snapshot: bool = True
    _member_states: list[_FittedMemberState] = field(default_factory=list, init=False, repr=False)

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
        protected_columns = {"unique_id", "ds", "y", *self.protected_exog_columns}
        return [
            column
            for column in [*self.futr_exog_list, *self.hist_exog_list]
            if column in frame.columns and column not in protected_columns and not column.startswith("price_lag_")
        ]

    def _transform_frame(
        self,
        frame: pd.DataFrame,
        fit: bool = False,
        target_scaler: AsinhQuantileScaler | None = None,
        exog_scaler: ZScoreScaler | None = None,
    ) -> tuple[pd.DataFrame, AsinhQuantileScaler | None, ZScoreScaler | None]:
        transformed = frame.copy()
        if self.target_transform not in {"identity", "asinh_q95"}:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")
        if self.exog_scaler not in {"identity", "zscore"}:
            raise ValueError(f"Unsupported exog_scaler: {self.exog_scaler}")

        if self.target_transform == "asinh_q95":
            if fit:
                target_scaler = AsinhQuantileScaler().fit(transformed["y"])
            if target_scaler is None:
                raise RuntimeError("Target scaler must be fit before transform.")
            for column in self._price_columns(transformed):
                transformed[column] = target_scaler.transform_series(transformed[column])

        exog_columns = self._scaled_exog_columns(transformed)
        if self.exog_scaler == "zscore" and exog_columns:
            if fit:
                exog_scaler = ZScoreScaler().fit(transformed, exog_columns)
            if exog_scaler is None:
                raise RuntimeError("Exogenous scaler must be fit before transform.")
            transformed = exog_scaler.transform_frame(transformed, exog_columns)
        return transformed, target_scaler, exog_scaler

    def _base_model_kwargs(self) -> dict[str, Any]:
        return {
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
            "hist_exog_list": self.hist_exog_list,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "windows_batch_size": self.windows_batch_size,
            "random_seed": self.random_seed,
            "logger": False,
            "enable_progress_bar": False,
        }

    def _resolved_member_kwargs(self) -> list[dict[str, Any]]:
        base_kwargs = self._base_model_kwargs()
        if not self.ensemble_members:
            return [base_kwargs]
        if self.ensemble_aggregation not in {"mean", "median"}:
            raise ValueError(f"Unsupported ensemble_aggregation: {self.ensemble_aggregation}")

        member_kwargs = []
        for member in self.ensemble_members:
            resolved = dict(base_kwargs)
            member_overrides = dict(member)
            seed_offset = int(member_overrides.pop("seed_offset", 0))
            resolved["random_seed"] = self.random_seed + seed_offset
            resolved.update(member_overrides)
            member_kwargs.append(resolved)
        return member_kwargs

    def fit(self, train_df: pd.DataFrame) -> None:
        self._member_states = []
        for member_kwargs in self._resolved_member_kwargs():
            transformed_train_df, target_scaler, exog_scaler = self._transform_frame(train_df, fit=True)
            model = self._nbeatsx_cls(**member_kwargs)
            nf = self._neuralforecast_cls(models=[model], freq=self.freq)
            fit_kwargs = {"df": transformed_train_df}
            if self.early_stop_patience_steps > 0:
                fit_kwargs["val_size"] = self.validation_size
            nf.fit(**fit_kwargs)
            self._member_states.append(
                _FittedMemberState(
                    nf=nf,
                    target_scaler=target_scaler,
                    exog_scaler=exog_scaler,
                )
            )

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self._member_states:
            raise RuntimeError("NBEATSxModel must be fit before predict.")
        member_predictions = []
        for member_state in self._member_states:
            transformed_history_df, _, _ = self._transform_frame(
                history_df,
                fit=False,
                target_scaler=member_state.target_scaler,
                exog_scaler=member_state.exog_scaler,
            )
            transformed_future_df, _, _ = self._transform_frame(
                future_df,
                fit=False,
                target_scaler=member_state.target_scaler,
                exog_scaler=member_state.exog_scaler,
            )

            expected_future = member_state.nf.make_future_dataframe(df=transformed_history_df)
            future_inputs = transformed_future_df.loc[:, ["unique_id", "ds", *self.futr_exog_list]].copy()
            futr_df = expected_future.merge(future_inputs, on=["unique_id", "ds"], how="left")
            if futr_df[self.futr_exog_list].isna().any().any():
                raise ValueError("Missing future exogenous values after aligning to NeuralForecast future grid.")

            prediction_df = member_state.nf.predict(df=transformed_history_df, futr_df=futr_df)
            model_column = [column for column in prediction_df.columns if column not in {"unique_id", "ds"}][0]
            prediction_df = prediction_df.rename(columns={model_column: "y_pred"}).loc[:, ["ds", "y_pred"]]
            if member_state.target_scaler is not None and self.target_transform == "asinh_q95":
                prediction_df["y_pred"] = member_state.target_scaler.inverse_transform_array(
                    prediction_df["y_pred"].to_numpy()
                )
            member_predictions.append(prediction_df.rename(columns={"y_pred": f"y_pred_{len(member_predictions)}"}))

        result = member_predictions[0]
        for member_prediction in member_predictions[1:]:
            result = result.merge(member_prediction, on="ds", how="inner")
        prediction_columns = [column for column in result.columns if column.startswith("y_pred_")]
        predictions_array = result[prediction_columns].to_numpy(dtype=float)
        if self.ensemble_aggregation == "median":
            result["y_pred"] = np.median(predictions_array, axis=1)
        else:
            result["y_pred"] = predictions_array.mean(axis=1)
        return result.loc[:, ["ds", "y_pred"]]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
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
            "protected_exog_columns": self.protected_exog_columns,
            "target_transform": self.target_transform,
            "exog_scaler": self.exog_scaler,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "validation_size": self.validation_size,
            "windows_batch_size": self.windows_batch_size,
            "ensemble_aggregation": self.ensemble_aggregation,
            "ensemble_members": self.ensemble_members,
            "random_seed": self.random_seed,
        }
        metadata = {
            "model_config": payload,
            "member_states": [],
        }

        for member_index, member_state in enumerate(self._member_states):
            member_dir = path / f"member_{member_index}"
            member_state.nf.save(str(member_dir), save_dataset=False, overwrite=True)
            metadata["member_states"].append(
                {
                    "member_dir": member_dir.name,
                    "target_scaler": None
                    if member_state.target_scaler is None
                    else {
                        "q": member_state.target_scaler.q_,
                        "max_abs_value": member_state.target_scaler.max_abs_value_,
                    },
                    "exog_scaler": None
                    if member_state.exog_scaler is None
                    else {
                        "mean": member_state.exog_scaler.mean_,
                        "std": member_state.exog_scaler.std_,
                    },
                }
            )

        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NBEATSxModel":
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model = cls(**metadata["model_config"])
            model._member_states = []
            for member_payload in metadata["member_states"]:
                member_nf = model._neuralforecast_cls.load(str(path / member_payload["member_dir"]))
                target_scaler_payload = member_payload["target_scaler"]
                exog_scaler_payload = member_payload["exog_scaler"]
                target_scaler = None
                if target_scaler_payload is not None:
                    target_scaler = AsinhQuantileScaler(
                        q_=target_scaler_payload["q"],
                        max_abs_value_=target_scaler_payload["max_abs_value"],
                    )
                exog_scaler = None
                if exog_scaler_payload is not None:
                    exog_scaler = ZScoreScaler(
                        mean_=exog_scaler_payload["mean"],
                        std_=exog_scaler_payload["std"],
                    )
                model._member_states.append(
                    _FittedMemberState(
                        nf=member_nf,
                        target_scaler=target_scaler,
                        exog_scaler=exog_scaler,
                    )
                )
            return model

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
