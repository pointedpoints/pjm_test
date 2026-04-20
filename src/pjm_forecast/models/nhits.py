from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pjm_forecast.prepared_data import default_nbeatsx_protected_exog_columns

from .base import ForecastModel
from .nbeatsx import AsinhQuantileScaler, ZScoreScaler, _FittedMemberState
from .quantile_losses import WeightedHuberMQLoss, WeightedMQLoss


def _build_nhits_class() -> tuple[type[Any], type[Any]]:
    try:
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.models import NHITS  # type: ignore
    except ImportError as exc:  # pragma: no cover - guarded by caller
        raise ImportError("NHITSModel requires neuralforecast and torch to be installed.") from exc
    return NeuralForecast, NHITS


@dataclass
class NHITSModel(ForecastModel):
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
    n_blocks: list[int] | None = None
    n_pool_kernel_size: list[int] | None = None
    n_freq_downsample: list[int] | None = None
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    protected_exog_columns: list[str] = field(default_factory=default_nbeatsx_protected_exog_columns)
    target_transform: str = "identity"
    exog_scaler: str = "identity"
    loss_name: str = "mae"
    loss_delta: float = 1.0
    quantiles: list[float] = field(default_factory=list)
    quantile_weights: list[float] = field(default_factory=list)
    early_stop_patience_steps: int = -1
    val_check_steps: int = 100
    validation_size: int = 168
    windows_batch_size: int = 1024
    ensemble_aggregation: str = "mean"
    ensemble_members: list[dict[str, Any]] = field(default_factory=list)
    random_seed: int = 7
    name: str = "nhits"
    supports_fitted_snapshot: bool = True
    _member_states: list[_FittedMemberState] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._neuralforecast_cls, self._nhits_cls = _build_nhits_class()
        if self.n_blocks is None:
            self.n_blocks = [1] * len(self.stack_types)
        if self.n_pool_kernel_size is None:
            self.n_pool_kernel_size = [2, 2, 1][: len(self.stack_types)]
            if len(self.n_pool_kernel_size) < len(self.stack_types):
                self.n_pool_kernel_size = [2] * len(self.stack_types)
        if self.n_freq_downsample is None:
            self.n_freq_downsample = [4, 2, 1][: len(self.stack_types)]
            if len(self.n_freq_downsample) < len(self.stack_types):
                self.n_freq_downsample = [1] * len(self.stack_types)
        if len(self.n_blocks) != len(self.stack_types):
            raise ValueError("n_blocks must match stack_types length.")
        if len(self.n_pool_kernel_size) != len(self.stack_types):
            raise ValueError("n_pool_kernel_size must match stack_types length.")
        if len(self.n_freq_downsample) != len(self.stack_types):
            raise ValueError("n_freq_downsample must match stack_types length.")
        self.loss_name = str(self.loss_name).lower()
        if self.loss_name not in {"mae", "mqloss", "huber_mqloss"}:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")
        if self.loss_delta <= 0.0:
            raise ValueError("loss_delta must be > 0.")
        if self.loss_name in {"mqloss", "huber_mqloss"}:
            if not self.quantiles:
                raise ValueError("quantiles are required when loss_name is quantile-based.")
            self.quantiles = sorted(float(value) for value in self.quantiles)
            if any(value <= 0.0 or value >= 1.0 for value in self.quantiles):
                raise ValueError("quantiles must be within (0, 1).")
            if not any(abs(value - 0.5) <= 1e-9 for value in self.quantiles):
                raise ValueError("quantiles must include 0.5 for p50-compatible evaluation.")
            if self.quantile_weights and len(self.quantile_weights) != len(self.quantiles):
                raise ValueError("quantile_weights must match quantiles length.")
        else:
            self.quantiles = []
            self.quantile_weights = []

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
            "n_blocks": self.n_blocks,
            "mlp_units": self.mlp_units,
            "n_pool_kernel_size": self.n_pool_kernel_size,
            "n_freq_downsample": self.n_freq_downsample,
            "pooling_mode": self.pooling_mode,
            "interpolation_mode": self.interpolation_mode,
            "futr_exog_list": self.futr_exog_list,
            "hist_exog_list": self.hist_exog_list,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "windows_batch_size": self.windows_batch_size,
            "random_seed": self.random_seed,
            "logger": False,
            "enable_progress_bar": False,
        }

    def _build_loss(self) -> Any:
        from neuralforecast.losses.pytorch import HuberMQLoss, MAE, MQLoss  # type: ignore

        if self.loss_name == "mqloss":
            if self.quantile_weights:
                return WeightedMQLoss(quantiles=list(self.quantiles), quantile_weights=list(self.quantile_weights))
            return MQLoss(quantiles=list(self.quantiles))
        if self.loss_name == "huber_mqloss":
            if self.quantile_weights:
                return WeightedHuberMQLoss(
                    quantiles=list(self.quantiles),
                    delta=float(self.loss_delta),
                    quantile_weights=list(self.quantile_weights),
                )
            return HuberMQLoss(quantiles=list(self.quantiles), delta=float(self.loss_delta))
        return MAE()

    def _build_valid_loss(self) -> Any | None:
        if self.loss_name not in {"mqloss", "huber_mqloss"}:
            return None
        return self._build_loss()

    def _quantile_output_suffixes(self) -> dict[str, float]:
        suffix_map: dict[str, float] = {}
        for quantile in self.quantiles:
            if quantile < 0.5:
                suffix = f"-lo-{np.round(100 - 200 * quantile, 2)}"
            elif quantile > 0.5:
                suffix = f"-hi-{np.round(100 - 200 * (1 - quantile), 2)}"
            else:
                suffix = "-median"
            suffix_map[suffix] = float(quantile)
        return suffix_map

    def _normalize_member_prediction(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        output_columns = [column for column in prediction_df.columns if column not in {"unique_id", "ds"}]
        if self.loss_name not in {"mqloss", "huber_mqloss"}:
            if len(output_columns) != 1:
                raise ValueError(f"Expected exactly one point forecast column, received: {output_columns}")
            return prediction_df.rename(columns={output_columns[0]: "y_pred"}).loc[:, ["ds", "y_pred"]]

        suffix_map = self._quantile_output_suffixes()
        quantile_frames = []
        for column in output_columns:
            quantile = next((value for suffix, value in suffix_map.items() if column.endswith(suffix)), None)
            if quantile is None:
                raise ValueError(f"Unable to map NeuralForecast output column {column!r} to a configured quantile.")
            quantile_frame = prediction_df.loc[:, ["ds", column]].rename(columns={column: "y_pred"}).copy()
            quantile_frame["quantile"] = quantile
            quantile_frames.append(quantile_frame)
        return pd.concat(quantile_frames, axis=0, ignore_index=True).sort_values(["ds", "quantile"]).reset_index(drop=True)

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
            model = self._nhits_cls(**member_kwargs, loss=self._build_loss(), valid_loss=self._build_valid_loss())
            nf = self._neuralforecast_cls(models=[model], freq=self.freq)
            fit_kwargs = {"df": transformed_train_df}
            if self.early_stop_patience_steps > 0:
                fit_kwargs["val_size"] = self.validation_size
            nf.fit(**fit_kwargs)
            self._member_states.append(_FittedMemberState(nf=nf, target_scaler=target_scaler, exog_scaler=exog_scaler))

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self._member_states:
            raise RuntimeError("NHITSModel must be fit before predict.")
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
            if self.futr_exog_list and futr_df[self.futr_exog_list].isna().any().any():
                raise ValueError("Missing future exogenous values after aligning to NeuralForecast future grid.")

            prediction_df = member_state.nf.predict(df=transformed_history_df, futr_df=futr_df)
            prediction_df = self._normalize_member_prediction(prediction_df)
            if member_state.target_scaler is not None and self.target_transform == "asinh_q95":
                prediction_df["y_pred"] = member_state.target_scaler.inverse_transform_array(prediction_df["y_pred"].to_numpy())
            member_predictions.append(prediction_df.rename(columns={"y_pred": f"y_pred_{len(member_predictions)}"}))

        result = member_predictions[0]
        merge_keys = ["ds", "quantile"] if "quantile" in result.columns else ["ds"]
        for member_prediction in member_predictions[1:]:
            result = result.merge(member_prediction, on=merge_keys, how="inner")
        prediction_columns = [column for column in result.columns if column.startswith("y_pred_")]
        predictions_array = result[prediction_columns].to_numpy(dtype=float)
        if self.ensemble_aggregation == "median":
            result["y_pred"] = np.median(predictions_array, axis=1)
        else:
            result["y_pred"] = predictions_array.mean(axis=1)
        output_columns = ["ds", "y_pred"] if "quantile" not in result.columns else ["ds", "quantile", "y_pred"]
        return result.loc[:, output_columns]

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
            "n_blocks": self.n_blocks,
            "n_pool_kernel_size": self.n_pool_kernel_size,
            "n_freq_downsample": self.n_freq_downsample,
            "pooling_mode": self.pooling_mode,
            "interpolation_mode": self.interpolation_mode,
            "protected_exog_columns": self.protected_exog_columns,
            "target_transform": self.target_transform,
            "exog_scaler": self.exog_scaler,
            "loss_name": self.loss_name,
            "loss_delta": self.loss_delta,
            "quantiles": self.quantiles,
            "quantile_weights": self.quantile_weights,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "validation_size": self.validation_size,
            "windows_batch_size": self.windows_batch_size,
            "ensemble_aggregation": self.ensemble_aggregation,
            "ensemble_members": self.ensemble_members,
            "random_seed": self.random_seed,
        }
        metadata = {"model_config": payload, "member_states": []}
        for member_index, member_state in enumerate(self._member_states):
            member_dir = path / f"member_{member_index}"
            member_state.nf.save(str(member_dir), save_dataset=False, overwrite=True)
            metadata["member_states"].append(
                {
                    "member_dir": member_dir.name,
                    "target_scaler": None
                    if member_state.target_scaler is None
                    else {"q": member_state.target_scaler.q_, "max_abs_value": member_state.target_scaler.max_abs_value_},
                    "exog_scaler": None
                    if member_state.exog_scaler is None
                    else {"mean": member_state.exog_scaler.mean_, "std": member_state.exog_scaler.std_},
                }
            )
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NHITSModel":
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
                    exog_scaler = ZScoreScaler(mean_=exog_scaler_payload["mean"], std_=exog_scaler_payload["std"])
                model._member_states.append(_FittedMemberState(nf=member_nf, target_scaler=target_scaler, exog_scaler=exog_scaler))
            return model

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
