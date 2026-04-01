from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..prepared_data import PreparedDataset
from .features import StackingRows, build_stacking_training_rows


PredictionLoader = Callable[[pd.DataFrame, list[pd.Timestamp], str, int, str, str | None], pd.DataFrame]


@dataclass(frozen=True)
class StackingParams:
    num_leaves: int
    learning_rate: float
    min_child_samples: int


@dataclass(frozen=True)
class StackingTuningResult:
    params: StackingParams
    score_grid: dict[str, float]
    output_model_name: str
    path: Path


class _RegressionModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class _TreeBackend(Protocol):
    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]) -> _RegressionModel: ...


class _ConstantRegressionModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class _LightGBMStackingBackend:
    def __init__(self, seed: int) -> None:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError("LightGBM stacker requires lightgbm to be installed. Install the '[gbm]' extra.") from exc
        self._lgb = lgb
        self.seed = int(seed)

    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]) -> _RegressionModel:
        clean_targets = targets.astype(float)
        if len(clean_targets) < 2 or clean_targets.nunique(dropna=False) < 2:
            return _ConstantRegressionModel(float(clean_targets.mean()) if len(clean_targets) else 0.0)
        model = self._lgb.LGBMRegressor(objective="regression", random_state=self.seed, verbose=-1, **params)
        model.fit(features, clean_targets)
        return model


def build_stacking_backend(model_family: str, seed: int) -> _TreeBackend:
    if model_family == "lightgbm":
        return _LightGBMStackingBackend(seed)
    if model_family == "xgboost":
        raise ValueError("Stacking model_family='xgboost' is not implemented yet.")
    raise ValueError(f"Unsupported stacking model_family: {model_family!r}")


class _ArtifactStoreLike(Protocol):
    def prediction(self, model_name: str, split: str, seed: int, variant: str | None = None) -> Path: ...

    def write_stacking_params(
        self,
        model_name: str,
        selected_params: StackingParams,
        score_grid: dict[str, float],
        output_model_name: str,
        model_family: str,
        base_model_names: list[str] | tuple[str, ...],
    ) -> Path: ...

    def load_stacking_params(self, model_name: str) -> dict[str, object]: ...

    def write_stacking_diagnostics(self, split: str, model_name: str, diagnostics_df: pd.DataFrame) -> Path: ...


class StackingRunner:
    def __init__(
        self,
        config: ProjectConfig,
        prepared_dataset: PreparedDataset,
        artifacts: _ArtifactStoreLike,
        *,
        prediction_loader: PredictionLoader | None = None,
        backend_factory: Callable[[str, int], _TreeBackend] | None = None,
    ) -> None:
        self.config = config
        self.prepared_dataset = prepared_dataset
        self.artifacts = artifacts
        self.prediction_loader = prediction_loader
        self.backend_factory = backend_factory or build_stacking_backend

    def tune(self, split: str = "validation") -> StackingTuningResult:
        if split != "validation":
            raise ValueError("Stacking tuning currently supports validation split only.")

        training_predictions = self._load_prediction_frames(
            forecast_days=self._warmup_days(),
            split_name="stacking_warmup",
            model_names=self._base_model_names(),
        )
        validation_predictions = self._load_prediction_frames(
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            model_names=self._base_model_names(),
        )
        training_rows = build_stacking_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            prediction_frames=training_predictions,
            schema=self.prepared_dataset.schema,
            base_model_names=self._base_model_names(),
        )
        validation_rows = build_stacking_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            prediction_frames=validation_predictions,
            schema=self.prepared_dataset.schema,
            base_model_names=self._base_model_names(),
        )

        best_params: StackingParams | None = None
        best_mae = float("inf")
        score_grid: dict[str, float] = {}
        for num_leaves in self.config.stacking["num_leaves_grid"]:
            for learning_rate in self.config.stacking["learning_rate_grid"]:
                for min_child_samples in self.config.stacking["min_child_samples_grid"]:
                    params = StackingParams(
                        num_leaves=int(num_leaves),
                        learning_rate=float(learning_rate),
                        min_child_samples=int(min_child_samples),
                    )
                    model = self._fit_model(training_rows, params)
                    stacked = self._apply_model(validation_predictions, validation_rows, model)
                    mae = float(np.mean(np.abs(stacked["y"].to_numpy(dtype=float) - stacked["y_pred"].to_numpy(dtype=float))))
                    key = (
                        f"num_leaves={params.num_leaves},"
                        f"learning_rate={params.learning_rate:.4f},"
                        f"min_child_samples={params.min_child_samples}"
                    )
                    score_grid[key] = mae
                    if mae < best_mae:
                        best_mae = mae
                        best_params = params
        if best_params is None:
            raise RuntimeError("Failed to tune stacking parameters.")

        params_path = self.artifacts.write_stacking_params(
            model_name=self.config.stacking_output_model_name,
            selected_params=best_params,
            score_grid=score_grid,
            output_model_name=self.config.stacking_output_model_name,
            model_family=self.config.stacking_model_family,
            base_model_names=self._base_model_names(),
        )
        return StackingTuningResult(
            params=best_params,
            score_grid=score_grid,
            output_model_name=self.config.stacking_output_model_name,
            path=params_path,
        )

    def apply(self, split: str = "test") -> Path:
        if split not in {"validation", "test"}:
            raise ValueError(f"Unsupported stacking split: {split!r}")

        params_payload = self.artifacts.load_stacking_params(self.config.stacking_output_model_name)
        params = StackingParams(
            num_leaves=int(params_payload["selected_params"]["num_leaves"]),
            learning_rate=float(params_payload["selected_params"]["learning_rate"]),
            min_child_samples=int(params_payload["selected_params"]["min_child_samples"]),
        )
        training_predictions = self._training_prediction_frames_for_split(split=split)
        target_predictions = self._load_prediction_frames(
            forecast_days=self.prepared_dataset.split_days(split),
            split_name=split,
            model_names=self._base_model_names(),
        )
        training_rows = build_stacking_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            prediction_frames=training_predictions,
            schema=self.prepared_dataset.schema,
            base_model_names=self._base_model_names(),
        )
        target_rows = build_stacking_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            prediction_frames=target_predictions,
            schema=self.prepared_dataset.schema,
            base_model_names=self._base_model_names(),
        )
        model = self._fit_model(training_rows, params)
        stacked = self._apply_model(target_predictions, target_rows, model)
        self.prepared_dataset.schema.validate_prediction_frame(stacked, require_metadata=False)
        output_path = self.artifacts.prediction(self.config.stacking_output_model_name, split, self._benchmark_seed())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stacked.to_parquet(output_path, index=False)

        diagnostics = compute_stacking_diagnostics(
            stacked,
            base_model_names=self._base_model_names(),
            model_name=self.config.stacking_output_model_name,
        )
        self.artifacts.write_stacking_diagnostics(split, self.config.stacking_output_model_name, diagnostics)
        return output_path

    def _training_prediction_frames_for_split(self, *, split: str) -> dict[str, pd.DataFrame]:
        warmup_predictions = self._load_prediction_frames(
            forecast_days=self._warmup_days(),
            split_name="stacking_warmup",
            model_names=self._base_model_names(),
        )
        if split == "validation":
            return warmup_predictions

        validation_predictions = self._load_prediction_frames(
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            model_names=self._base_model_names(),
        )
        combined: dict[str, pd.DataFrame] = {}
        for model_name in self._base_model_names():
            combined_frame = pd.concat(
                [warmup_predictions[model_name], validation_predictions[model_name]],
                axis=0,
                ignore_index=True,
            )
            combined_frame["split"] = "stacking_training"
            combined[model_name] = combined_frame
        return combined

    def _fit_model(self, training_rows: StackingRows, params: StackingParams) -> _RegressionModel:
        backend = self.backend_factory(self.config.stacking_model_family, self._benchmark_seed())
        backend_params = dict(self.config.stacking.get("regressor_params", {}))
        backend_params.update(
            {
                "num_leaves": params.num_leaves,
                "learning_rate": params.learning_rate,
                "min_child_samples": params.min_child_samples,
            }
        )
        return backend.fit_regressor(
            training_rows.frame.loc[:, list(training_rows.feature_columns)],
            training_rows.frame["y"],
            backend_params,
        )

    def _apply_model(
        self,
        prediction_frames: dict[str, pd.DataFrame],
        target_rows: StackingRows,
        model: _RegressionModel,
    ) -> pd.DataFrame:
        reference_model = self._base_model_names()[0]
        corrected = prediction_frames[reference_model].copy().sort_values("ds").reset_index(drop=True)
        feature_frame = target_rows.frame.loc[:, list(target_rows.feature_columns)]
        corrected["y_pred"] = model.predict(feature_frame).astype(float)
        corrected["model"] = self.config.stacking_output_model_name
        for model_name in self._base_model_names():
            corrected[f"pred_{model_name}"] = target_rows.frame[f"pred_{model_name}"].to_numpy(dtype=float)
        corrected["pred_ensemble_mean"] = target_rows.frame["pred_ensemble_mean"].to_numpy(dtype=float)
        corrected["pred_ensemble_std"] = target_rows.frame["pred_ensemble_std"].to_numpy(dtype=float)
        corrected["pred_ensemble_spread"] = target_rows.frame["pred_ensemble_spread"].to_numpy(dtype=float)
        return corrected

    def _load_prediction_frames(
        self,
        *,
        forecast_days: list[pd.Timestamp],
        split_name: str,
        model_names: list[str],
    ) -> dict[str, pd.DataFrame]:
        return {
            model_name: self._load_predictions(
                forecast_days=forecast_days,
                split_name=split_name,
                seed=self._benchmark_seed(),
                model_name=model_name,
            )
            for model_name in model_names
        }

    def _load_predictions(
        self,
        *,
        forecast_days: list[pd.Timestamp],
        split_name: str,
        seed: int,
        model_name: str,
        variant: str | None = None,
    ) -> pd.DataFrame:
        output_path = self.artifacts.prediction(model_name, split_name, seed, variant)
        if output_path.exists():
            return pd.read_parquet(output_path)
        if self.prediction_loader is None:
            raise FileNotFoundError(f"Missing prediction artifact: {output_path}")
        return self.prediction_loader(
            self.prepared_dataset.feature_df,
            forecast_days,
            split_name,
            seed,
            model_name,
            variant,
        )

    def _warmup_days(self) -> list[pd.Timestamp]:
        validation_start = self.prepared_dataset.split_boundaries["validation_start"]
        warmup_start = validation_start - pd.Timedelta(days=int(self.config.stacking["warmup_days"]))
        warmup_end = validation_start - pd.Timedelta(days=1)
        return self.prepared_dataset.days_between(start=warmup_start, end=warmup_end)

    def _base_model_names(self) -> list[str]:
        return [str(model_name) for model_name in self.config.stacking_base_model_names]

    def _benchmark_seed(self) -> int:
        return int(self.config.project["benchmark_seed"])


def compute_stacking_diagnostics(
    predictions: pd.DataFrame,
    *,
    base_model_names: list[str] | tuple[str, ...],
    model_name: str,
) -> pd.DataFrame:
    y_true = predictions["y"].to_numpy(dtype=float)
    stacked_pred = predictions["y_pred"].to_numpy(dtype=float)
    ensemble_mean = predictions["pred_ensemble_mean"].to_numpy(dtype=float)

    base_maes = {
        base_model_name: float(
            np.mean(np.abs(y_true - predictions[f"pred_{base_model_name}"].to_numpy(dtype=float)))
        )
        for base_model_name in base_model_names
    }
    best_base_model = min(base_maes, key=base_maes.get)
    diagnostics = {
        "model": model_name,
        "overall_mae_stacker": float(np.mean(np.abs(y_true - stacked_pred))),
        "overall_mae_ensemble_mean": float(np.mean(np.abs(y_true - ensemble_mean))),
        "overall_mae_best_base": float(base_maes[best_base_model]),
        "best_base_model": best_base_model,
        "overall_mae_delta_vs_best_base": float(np.mean(np.abs(y_true - stacked_pred)) - base_maes[best_base_model]),
        "overall_mae_delta_vs_ensemble_mean": float(
            np.mean(np.abs(y_true - stacked_pred)) - np.mean(np.abs(y_true - ensemble_mean))
        ),
    }
    for base_model_name, mae in base_maes.items():
        diagnostics[f"overall_mae_{base_model_name}"] = float(mae)
    return pd.DataFrame([diagnostics])
