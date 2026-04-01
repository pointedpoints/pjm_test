from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..prepared_data import PreparedDataset, forecast_day_from_prediction_frame


PredictionLoader = Callable[[pd.DataFrame, list[pd.Timestamp], str, int, str, str | None], pd.DataFrame]


@dataclass(frozen=True)
class SpikeCorrectionParams:
    spike_quantile: float
    gate_threshold: float
    delta_clip_quantile: float


@dataclass(frozen=True)
class SpikeCorrectionTuningResult:
    params: SpikeCorrectionParams
    score_grid: dict[str, float]
    output_model_name: str
    path: Path


@dataclass(frozen=True)
class SpikeTrainingRows:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class _FittedSpikeHelper:
    threshold: float
    delta_clip_value: float
    classifier: "_ProbabilityModel"
    regressor: "_RegressionModel"


class _ProbabilityModel(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


class _RegressionModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class _TreeBackend(Protocol):
    def fit_classifier(self, features: pd.DataFrame, labels: pd.Series, params: dict[str, object]) -> _ProbabilityModel: ...

    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]) -> _RegressionModel: ...


class _ConstantProbabilityModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        size = len(features)
        probs = np.full(size, self.positive_probability, dtype=float)
        return np.column_stack([1.0 - probs, probs])


class _ConstantRegressionModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class _LightGBMSpikeBackend:
    def __init__(self, seed: int) -> None:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError("Spike corrector requires lightgbm to be installed. Install the '[gbm]' extra.") from exc
        self._lgb = lgb
        self.seed = int(seed)

    def fit_classifier(self, features: pd.DataFrame, labels: pd.Series, params: dict[str, object]) -> _ProbabilityModel:
        clean_labels = labels.astype(int)
        if clean_labels.nunique(dropna=False) < 2:
            return _ConstantProbabilityModel(float(clean_labels.mean()) if len(clean_labels) else 0.0)
        model = self._lgb.LGBMClassifier(objective="binary", random_state=self.seed, verbose=-1, **params)
        model.fit(features, clean_labels)
        return model

    def fit_regressor(self, features: pd.DataFrame, targets: pd.Series, params: dict[str, object]) -> _RegressionModel:
        clean_targets = targets.astype(float)
        if len(clean_targets) < 2 or clean_targets.nunique(dropna=False) < 2:
            return _ConstantRegressionModel(float(clean_targets.mean()) if len(clean_targets) else 0.0)
        model = self._lgb.LGBMRegressor(objective="regression", random_state=self.seed, verbose=-1, **params)
        model.fit(features, clean_targets)
        return model


def build_spike_backend(model_family: str, seed: int) -> _TreeBackend:
    if model_family == "lightgbm":
        return _LightGBMSpikeBackend(seed)
    if model_family == "xgboost":
        raise ValueError("Spike corrector model_family='xgboost' is not implemented yet.")
    raise ValueError(f"Unsupported spike corrector model_family: {model_family!r}")


class _ArtifactStoreLike(Protocol):
    def prediction(self, model_name: str, split: str, seed: int, variant: str | None = None) -> Path: ...

    def write_spike_params(
        self,
        model_name: str,
        selected_params: SpikeCorrectionParams,
        score_grid: dict[str, float],
        output_model_name: str,
        model_family: str,
    ) -> Path: ...

    def load_spike_params(self, model_name: str) -> dict[str, object]: ...

    def write_spike_diagnostics(self, split: str, model_name: str, diagnostics_df: pd.DataFrame) -> Path: ...


class SpikeCorrectorRunner:
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
        self.backend_factory = backend_factory or build_spike_backend

    def tune(self, base_model: str = "nbeatsx", split: str = "validation") -> SpikeCorrectionTuningResult:
        if split != "validation":
            raise ValueError("Spike correction tuning currently supports validation split only.")

        training_predictions = self._load_predictions(
            forecast_days=self._warmup_days(),
            split_name="spike_warmup",
            seed=self._benchmark_seed(),
            model_name=base_model,
        )
        validation_predictions = self._load_predictions(
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            seed=self._benchmark_seed(),
            model_name=base_model,
        )
        training_rows = build_spike_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            base_predictions=training_predictions,
            schema=self.prepared_dataset.schema,
        )
        validation_rows = build_spike_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            base_predictions=validation_predictions,
            schema=self.prepared_dataset.schema,
        )

        best_params: SpikeCorrectionParams | None = None
        best_mae = float("inf")
        score_grid: dict[str, float] = {}
        for spike_quantile in self.config.spike_correction["spike_quantile_grid"]:
            for gate_threshold in self.config.spike_correction["gate_threshold_grid"]:
                for delta_clip_quantile in self.config.spike_correction["delta_clip_quantile_grid"]:
                    params = SpikeCorrectionParams(
                        spike_quantile=float(spike_quantile),
                        gate_threshold=float(gate_threshold),
                        delta_clip_quantile=float(delta_clip_quantile),
                    )
                    helper = self._fit_helper(training_rows, params)
                    corrected = self._apply_helper(
                        validation_predictions,
                        validation_rows,
                        helper,
                        params=params,
                        output_model_name=self.config.spike_output_model_name,
                    )
                    mae = float(np.mean(np.abs(corrected["y"].to_numpy(dtype=float) - corrected["y_pred"].to_numpy(dtype=float))))
                    key = (
                        f"spike_q={params.spike_quantile:.4f},"
                        f"gate={params.gate_threshold:.4f},"
                        f"clip_q={params.delta_clip_quantile:.4f}"
                    )
                    score_grid[key] = mae
                    if mae < best_mae:
                        best_mae = mae
                        best_params = params
        if best_params is None:
            raise RuntimeError("Failed to tune spike correction parameters.")

        params_path = self.artifacts.write_spike_params(
            model_name=self.config.spike_output_model_name,
            selected_params=best_params,
            score_grid=score_grid,
            output_model_name=self.config.spike_output_model_name,
            model_family=self.config.spike_model_family,
        )
        return SpikeCorrectionTuningResult(
            params=best_params,
            score_grid=score_grid,
            output_model_name=self.config.spike_output_model_name,
            path=params_path,
        )

    def apply(self, base_model: str = "nbeatsx", split: str = "test") -> Path:
        if split not in {"validation", "test"}:
            raise ValueError(f"Unsupported spike correction split: {split!r}")
        params_payload = self.artifacts.load_spike_params(self.config.spike_output_model_name)
        params = SpikeCorrectionParams(
            spike_quantile=float(params_payload["selected_params"]["spike_quantile"]),
            gate_threshold=float(params_payload["selected_params"]["gate_threshold"]),
            delta_clip_quantile=float(params_payload["selected_params"]["delta_clip_quantile"]),
        )
        training_predictions = self._training_predictions_for_split(base_model=base_model, split=split)
        target_predictions = self._load_predictions(
            forecast_days=self.prepared_dataset.split_days(split),
            split_name=split,
            seed=self._benchmark_seed(),
            model_name=base_model,
        )
        training_rows = build_spike_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            base_predictions=training_predictions,
            schema=self.prepared_dataset.schema,
        )
        target_rows = build_spike_training_rows(
            feature_df=self.prepared_dataset.feature_df,
            base_predictions=target_predictions,
            schema=self.prepared_dataset.schema,
        )
        helper = self._fit_helper(training_rows, params)
        corrected = self._apply_helper(
            target_predictions,
            target_rows,
            helper,
            params=params,
            output_model_name=self.config.spike_output_model_name,
        )
        self.prepared_dataset.schema.validate_prediction_frame(corrected, require_metadata=False)
        output_path = self.artifacts.prediction(self.config.spike_output_model_name, split, self._benchmark_seed())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_parquet(output_path, index=False)

        diagnostics = compute_spike_diagnostics(corrected, threshold=helper.threshold, model_name=self.config.spike_output_model_name)
        self.artifacts.write_spike_diagnostics(split, self.config.spike_output_model_name, diagnostics)
        return output_path

    def _training_predictions_for_split(self, *, base_model: str, split: str) -> pd.DataFrame:
        warmup_predictions = self._load_predictions(
            forecast_days=self._warmup_days(),
            split_name="spike_warmup",
            seed=self._benchmark_seed(),
            model_name=base_model,
        )
        if split == "validation":
            return warmup_predictions
        validation_predictions = self._load_predictions(
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            seed=self._benchmark_seed(),
            model_name=base_model,
        )
        training_predictions = pd.concat([warmup_predictions, validation_predictions], axis=0, ignore_index=True)
        training_predictions["split"] = "spike_training"
        return training_predictions

    def _fit_helper(self, training_rows: SpikeTrainingRows, params: SpikeCorrectionParams) -> _FittedSpikeHelper:
        threshold = float(training_rows.frame["y"].quantile(params.spike_quantile))
        training_frame = training_rows.frame.copy()
        training_frame["spike_label"] = (training_frame["y"] >= threshold).astype(int)
        backend = self.backend_factory(self.config.spike_model_family, self._benchmark_seed())

        classifier = backend.fit_classifier(
            training_frame.loc[:, list(training_rows.feature_columns)],
            training_frame["spike_label"],
            dict(self.config.spike_correction.get("classifier_params", {})),
        )
        spike_rows = training_frame.loc[training_frame["spike_label"] == 1].copy()
        regressor = backend.fit_regressor(
            spike_rows.loc[:, list(training_rows.feature_columns)],
            spike_rows["residual"],
            dict(self.config.spike_correction.get("regressor_params", {})),
        )
        delta_clip_value = float(spike_rows["residual"].abs().quantile(params.delta_clip_quantile)) if not spike_rows.empty else 0.0
        if np.isnan(delta_clip_value):
            delta_clip_value = 0.0
        return _FittedSpikeHelper(
            threshold=threshold,
            delta_clip_value=delta_clip_value,
            classifier=classifier,
            regressor=regressor,
        )

    def _apply_helper(
        self,
        base_predictions: pd.DataFrame,
        target_rows: SpikeTrainingRows,
        helper: _FittedSpikeHelper,
        *,
        params: SpikeCorrectionParams,
        output_model_name: str,
    ) -> pd.DataFrame:
        feature_frame = target_rows.frame.loc[:, list(target_rows.feature_columns)]
        spike_prob = helper.classifier.predict_proba(feature_frame)[:, 1]
        raw_delta = helper.regressor.predict(feature_frame)
        clipped_delta = np.clip(raw_delta.astype(float), -helper.delta_clip_value, helper.delta_clip_value)
        spike_flag = spike_prob >= params.gate_threshold
        applied_delta = np.where(spike_flag, clipped_delta, 0.0)

        corrected = base_predictions.copy()
        corrected["y_pred_base"] = target_rows.frame["y_pred_base"].to_numpy(dtype=float)
        corrected["spike_prob"] = spike_prob.astype(float)
        corrected["spike_flag"] = spike_flag.astype(int)
        corrected["spike_delta"] = applied_delta.astype(float)
        corrected["y_pred"] = corrected["y_pred_base"] + corrected["spike_delta"]
        corrected["model"] = output_model_name
        return corrected

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
        warmup_start = validation_start - pd.Timedelta(days=int(self.config.spike_correction["warmup_days"]))
        warmup_end = validation_start - pd.Timedelta(days=1)
        return self.prepared_dataset.days_between(start=warmup_start, end=warmup_end)

    def _benchmark_seed(self) -> int:
        return int(self.config.project["benchmark_seed"])


def build_spike_training_rows(
    *,
    feature_df: pd.DataFrame,
    base_predictions: pd.DataFrame,
    schema,
) -> SpikeTrainingRows:
    schema.validate_feature_frame(feature_df)
    schema.validate_prediction_frame(base_predictions, require_metadata=False)
    feature_columns = (
        schema.future_exog_columns()
        + schema.calendar_columns()
        + schema.price_lag_columns()
        + schema.load_lag_columns()
    )
    feature_columns = tuple(dict.fromkeys(feature_columns))
    merged = base_predictions.copy().rename(columns={"y_pred": "y_pred_base"})
    feature_slice = feature_df.loc[:, ["ds", *feature_columns]].copy()
    merged = merged.merge(feature_slice, on="ds", how="left", validate="one_to_one")
    if merged[list(feature_columns)].isna().any().any():
        missing_columns = [column for column in feature_columns if merged[column].isna().any()]
        raise ValueError(f"Spike training rows contain missing feature values: {missing_columns}")

    rows: list[pd.DataFrame] = []
    for _, day_df in merged.groupby(merged["ds"].dt.normalize(), sort=True):
        ordered = day_df.sort_values("ds").copy()
        ordered["forecast_day"] = forecast_day_from_prediction_frame(ordered)
        ordered["base_day_pred_max"] = float(ordered["y_pred_base"].max())
        ordered["base_day_pred_min"] = float(ordered["y_pred_base"].min())
        ordered["base_day_pred_spread"] = ordered["base_day_pred_max"] - ordered["base_day_pred_min"]
        ordered["base_hour_rank_desc"] = ordered["y_pred_base"].rank(method="first", ascending=False).astype(float)
        ordered["base_gap_to_day_max"] = ordered["base_day_pred_max"] - ordered["y_pred_base"]
        ordered["base_ramp_prev"] = ordered["y_pred_base"].diff().fillna(0.0)
        ordered["base_ramp_next"] = (ordered["y_pred_base"].shift(-1) - ordered["y_pred_base"]).fillna(0.0)
        ordered["residual"] = ordered["y"] - ordered["y_pred_base"]
        rows.append(ordered)

    frame = pd.concat(rows, axis=0, ignore_index=True)
    model_feature_columns = (
        "y_pred_base",
        "base_day_pred_max",
        "base_day_pred_min",
        "base_day_pred_spread",
        "base_hour_rank_desc",
        "base_gap_to_day_max",
        "base_ramp_prev",
        "base_ramp_next",
        *feature_columns,
    )
    return SpikeTrainingRows(frame=frame, feature_columns=tuple(model_feature_columns))


def compute_spike_diagnostics(predictions: pd.DataFrame, *, threshold: float, model_name: str) -> pd.DataFrame:
    y_true = predictions["y"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    y_pred_base = predictions["y_pred_base"].to_numpy(dtype=float)
    true_spike = y_true >= float(threshold)
    predicted_spike = predictions["spike_flag"].to_numpy(dtype=int).astype(bool)

    tp = int(np.sum(predicted_spike & true_spike))
    fp = int(np.sum(predicted_spike & ~true_spike))
    fn = int(np.sum(~predicted_spike & true_spike))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def _mae(mask: np.ndarray, predictions_array: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs(y_true[mask] - predictions_array[mask])))

    diagnostics = {
        "model": model_name,
        "spike_threshold": float(threshold),
        "overall_mae_base": float(np.mean(np.abs(y_true - y_pred_base))),
        "overall_mae_corrected": float(np.mean(np.abs(y_true - y_pred))),
        "overall_mae_delta": float(np.mean(np.abs(y_true - y_pred)) - np.mean(np.abs(y_true - y_pred_base))),
        "spike_hour_mae_base": _mae(true_spike, y_pred_base),
        "spike_hour_mae_corrected": _mae(true_spike, y_pred),
        "non_spike_hour_mae_base": _mae(~true_spike, y_pred_base),
        "non_spike_hour_mae_corrected": _mae(~true_spike, y_pred),
        "spike_precision": precision,
        "spike_recall": recall,
        "spike_f1": f1,
        "corrected_hour_coverage": float(np.mean(predicted_spike.astype(float))),
    }
    return pd.DataFrame([diagnostics])
