from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pjm_forecast.prepared_data import default_nbeatsx_protected_exog_columns

from .base import ForecastModel
from .quantile_losses import WeightedHuberMQLoss, WeightedMQLoss


def _build_spike_aware_nbeatsx_class() -> tuple[type[Any], type[Any]]:
    try:
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.losses.pytorch import HuberMQLoss, MAE, MQLoss  # type: ignore
        from neuralforecast.models.nbeatsx import (  # type: ignore
            NBEATSBlock,
            ExogenousBasis,
            IdentityBasis,
            NBEATSx,
            SeasonalityBasis,
            TrendBasis,
        )
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except ImportError as exc:  # pragma: no cover - guarded by caller
        raise ImportError("NBEATSxModel requires neuralforecast and torch to be installed.") from exc

    class SpikeBasis(nn.Module):
        """Discrete spike basis over selected forecast hours."""

        def __init__(
            self,
            backcast_size: int,
            forecast_size: int,
            spike_hours: list[int],
            spike_kernel: str = "delta",
            spike_radius: int = 0,
            out_features: int = 1,
        ) -> None:
            super().__init__()
            if not spike_hours:
                raise ValueError("SpikeBasis requires at least one spike hour.")
            invalid_hours = [hour for hour in spike_hours if hour < 0 or hour >= forecast_size]
            if invalid_hours:
                raise ValueError(f"SpikeBasis received invalid hours for horizon {forecast_size}: {invalid_hours}")
            if spike_kernel not in {"delta", "triangle", "box"}:
                raise ValueError(f"Unsupported spike_kernel: {spike_kernel}")
            if spike_radius < 0:
                raise ValueError("SpikeBasis requires spike_radius >= 0.")

            self.backcast_size = backcast_size
            self.forecast_size = forecast_size
            self.out_features = out_features
            self.spike_hours = list(dict.fromkeys(int(hour) for hour in spike_hours))
            self.spike_kernel = spike_kernel
            self.spike_radius = int(spike_radius)

            basis = np.zeros((len(self.spike_hours), forecast_size), dtype=np.float32)
            for idx, hour in enumerate(self.spike_hours):
                if self.spike_kernel == "delta" or self.spike_radius == 0:
                    basis[idx, hour] = 1.0
                    continue

                start = max(0, hour - self.spike_radius)
                end = min(forecast_size - 1, hour + self.spike_radius)
                for target_hour in range(start, end + 1):
                    distance = abs(target_hour - hour)
                    if self.spike_kernel == "box":
                        weight = 1.0
                    else:
                        weight = 1.0 - distance / (self.spike_radius + 1)
                    basis[idx, target_hour] = weight
            self.register_buffer("forecast_basis", torch.tensor(basis, dtype=torch.float32))

        @property
        def n_basis(self) -> int:
            return int(self.forecast_basis.shape[0])

        def forward(self, theta: Any) -> tuple[Any, Any]:
            batch_size = theta.shape[0]
            theta = theta.reshape(batch_size, self.out_features, self.n_basis)
            forecast = torch.einsum("bon,nh->boh", theta, self.forecast_basis).permute(0, 2, 1)
            backcast = torch.zeros(
                batch_size,
                self.backcast_size,
                device=theta.device,
                dtype=theta.dtype,
            )
            return backcast, forecast

    class SpikeAwareNBEATSx(NBEATSx):
        def __init__(
            self,
            *args: Any,
            spike_hours: list[int] | None = None,
            spike_kernel: str = "delta",
            spike_radius: int = 0,
            n_blocks: list[int] | None = None,
            loss: Any | None = None,
            **kwargs: Any,
        ) -> None:
            stack_types = list(kwargs.get("stack_types", ["identity", "trend", "seasonality"]))
            if n_blocks is None:
                n_blocks = [1] * len(stack_types)
            if "spike" in stack_types and not spike_hours:
                raise ValueError("stack_types includes 'spike' but spike_hours is empty.")

            self.spike_hours = [] if spike_hours is None else [int(hour) for hour in spike_hours]
            self.spike_kernel = str(spike_kernel)
            self.spike_radius = int(spike_radius)
            kwargs["n_blocks"] = n_blocks
            kwargs["loss"] = MAE() if loss is None else loss
            super().__init__(*args, **kwargs)

        def create_stack(
            self,
            h: int,
            input_size: int,
            stack_types: list[str],
            n_blocks: list[int],
            mlp_units: list[list[int]],
            dropout_prob_theta: float,
            activation: str,
            shared_weights: bool,
            n_polynomials: int,
            n_harmonics: int,
            futr_input_size: int,
            hist_input_size: int,
            stat_input_size: int,
        ) -> list[Any]:
            if len(n_blocks) != len(stack_types):
                raise ValueError("n_blocks must match stack_types length.")

            block_list = []
            for stack_index, stack_type in enumerate(stack_types):
                for block_id in range(n_blocks[stack_index]):
                    if shared_weights and block_id > 0:
                        nbeats_block = block_list[-1]
                    else:
                        if stack_type == "seasonality":
                            n_theta = (
                                2
                                * (self.loss.outputsize_multiplier + 1)
                                * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                            )
                            basis = SeasonalityBasis(
                                harmonics=n_harmonics,
                                backcast_size=input_size,
                                forecast_size=h,
                                out_features=self.loss.outputsize_multiplier,
                            )
                        elif stack_type == "trend":
                            n_theta = (self.loss.outputsize_multiplier + 1) * (n_polynomials + 1)
                            basis = TrendBasis(
                                degree_of_polynomial=n_polynomials,
                                backcast_size=input_size,
                                forecast_size=h,
                                out_features=self.loss.outputsize_multiplier,
                            )
                        elif stack_type == "identity":
                            n_theta = input_size + self.loss.outputsize_multiplier * h
                            basis = IdentityBasis(
                                backcast_size=input_size,
                                forecast_size=h,
                                out_features=self.loss.outputsize_multiplier,
                            )
                        elif stack_type == "exogenous":
                            if futr_input_size + stat_input_size <= 0:
                                raise ValueError("No stats or future exogenous. ExogenousBlock not supported.")
                            n_theta = 2 * (futr_input_size + stat_input_size)
                            basis = ExogenousBasis(forecast_size=h)
                        elif stack_type == "spike":
                            basis = SpikeBasis(
                                backcast_size=input_size,
                                forecast_size=h,
                                spike_hours=self.spike_hours,
                                spike_kernel=self.spike_kernel,
                                spike_radius=self.spike_radius,
                                out_features=self.loss.outputsize_multiplier,
                            )
                            n_theta = basis.n_basis * self.loss.outputsize_multiplier
                        else:
                            raise ValueError(f"Block type {stack_type} not found!")

                        nbeats_block = NBEATSBlock(
                            input_size=input_size,
                            h=h,
                            futr_input_size=futr_input_size,
                            hist_input_size=hist_input_size,
                            stat_input_size=stat_input_size,
                            n_theta=n_theta,
                            mlp_units=mlp_units,
                            basis=basis,
                            dropout_prob=dropout_prob_theta,
                            activation=activation,
                        )
                    block_list.append(nbeats_block)
            return block_list

    return NeuralForecast, SpikeAwareNBEATSx


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
    n_blocks: list[int] | None = None
    spike_hours: list[int] = field(default_factory=list)
    spike_kernel: str = "delta"
    spike_radius: int = 0
    protected_exog_columns: list[str] = field(default_factory=default_nbeatsx_protected_exog_columns)
    target_transform: str = "identity"
    exog_scaler: str = "identity"
    loss_name: str = "mae"
    loss_delta: float = 1.0
    quantiles: list[float] = field(default_factory=list)
    quantile_weights: list[float] = field(default_factory=list)
    quantile_deltas: list[float] = field(default_factory=list)
    monotonicity_penalty: float = 0.0
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
        self._neuralforecast_cls, self._nbeatsx_cls = _build_spike_aware_nbeatsx_class()
        if self.n_blocks is None:
            self.n_blocks = [1] * len(self.stack_types)
        if len(self.n_blocks) != len(self.stack_types):
            raise ValueError("n_blocks must match stack_types length.")
        if "spike" in self.stack_types and not self.spike_hours:
            raise ValueError("stack_types includes 'spike' but spike_hours is empty.")
        if self.spike_kernel not in {"delta", "triangle", "box"}:
            raise ValueError(f"Unsupported spike_kernel: {self.spike_kernel}")
        if self.spike_radius < 0:
            raise ValueError("spike_radius must be >= 0.")
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
            if self.quantile_deltas and len(self.quantile_deltas) != len(self.quantiles):
                raise ValueError("quantile_deltas must match quantiles length.")
            if any(value <= 0.0 for value in self.quantile_deltas):
                raise ValueError("quantile_deltas must be strictly positive.")
            if self.monotonicity_penalty < 0.0:
                raise ValueError("monotonicity_penalty must be >= 0.")
        else:
            self.quantiles = []
            self.quantile_weights = []
            self.quantile_deltas = []
            self.monotonicity_penalty = 0.0

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
            "n_blocks": self.n_blocks,
            "spike_hours": self.spike_hours,
            "spike_kernel": self.spike_kernel,
            "spike_radius": self.spike_radius,
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
            if self.quantile_weights or self.monotonicity_penalty > 0.0:
                return WeightedMQLoss(
                    quantiles=list(self.quantiles),
                    quantile_weights=list(self.quantile_weights) or None,
                    monotonicity_penalty=float(self.monotonicity_penalty),
                )
            return MQLoss(quantiles=list(self.quantiles))
        if self.loss_name == "huber_mqloss":
            if self.quantile_weights or self.quantile_deltas or self.monotonicity_penalty > 0.0:
                return WeightedHuberMQLoss(
                    quantiles=list(self.quantiles),
                    delta=float(self.loss_delta),
                    quantile_weights=list(self.quantile_weights) or None,
                    quantile_deltas=list(self.quantile_deltas) or None,
                    monotonicity_penalty=float(self.monotonicity_penalty),
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
            model = self._nbeatsx_cls(
                **member_kwargs,
                loss=self._build_loss(),
                valid_loss=self._build_valid_loss(),
            )
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
            prediction_df = self._normalize_member_prediction(prediction_df)
            if member_state.target_scaler is not None and self.target_transform == "asinh_q95":
                prediction_df["y_pred"] = member_state.target_scaler.inverse_transform_array(
                    prediction_df["y_pred"].to_numpy()
                )
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
            "spike_hours": self.spike_hours,
            "spike_kernel": self.spike_kernel,
            "spike_radius": self.spike_radius,
            "protected_exog_columns": self.protected_exog_columns,
            "target_transform": self.target_transform,
            "exog_scaler": self.exog_scaler,
            "loss_name": self.loss_name,
            "loss_delta": self.loss_delta,
            "quantiles": self.quantiles,
            "quantile_weights": self.quantile_weights,
            "quantile_deltas": self.quantile_deltas,
            "monotonicity_penalty": self.monotonicity_penalty,
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
