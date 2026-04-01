from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from ..config import ProjectConfig
from ..prepared_data import PreparedDataset
from .residual_memory import RetrievalConfig, RetrievalParams, apply_residual_retrieval, tune_retrieval_params


PredictionLoader = Callable[[pd.DataFrame, list[pd.Timestamp], str, int, str, str | None], pd.DataFrame]


@dataclass(frozen=True)
class RetrievalTuningResult:
    params: RetrievalParams
    score_grid: dict[str, float]
    output_model_name: str
    path: Path


class _ArtifactStoreLike(Protocol):
    def prediction(self, model_name: str, split: str, seed: int, variant: str | None = None) -> Path: ...

    def write_retrieval_params(
        self,
        model_name: str,
        selected_params: RetrievalParams,
        score_grid: dict[str, float],
        output_model_name: str,
    ) -> Path: ...

    def load_retrieval_params(self, model_name: str = "nbeatsx_rag") -> dict[str, object]: ...


class RetrievalRunner:
    def __init__(
        self,
        config: ProjectConfig,
        prepared_dataset: PreparedDataset,
        artifacts: _ArtifactStoreLike,
        *,
        prediction_loader: PredictionLoader | None = None,
    ) -> None:
        self.config = config
        self.prepared_dataset = prepared_dataset
        self.artifacts = artifacts
        self.prediction_loader = prediction_loader

    def tune(self, base_model: str = "nbeatsx", split: str = "validation") -> RetrievalTuningResult:
        if split != "validation":
            raise ValueError("Retrieval tuning currently supports validation split only.")
        retrieval_cfg = self._retrieval_config()
        feature_df = self.prepared_dataset.feature_df
        benchmark_seed = self._benchmark_seed()
        validation_predictions = self._load_predictions(
            feature_df=feature_df,
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            seed=benchmark_seed,
            model_name=base_model,
        )
        warmup_predictions = self._load_predictions(
            feature_df=feature_df,
            forecast_days=self._warmup_days(),
            split_name="retrieval_warmup",
            seed=benchmark_seed,
            model_name=base_model,
        )
        retrieval = self.config.retrieval
        best_params, tuning_scores = tune_retrieval_params(
            feature_df=feature_df,
            validation_predictions=validation_predictions,
            initial_memory_predictions=warmup_predictions,
            config=retrieval_cfg,
            alpha_grid=[float(value) for value in retrieval["alpha_grid"]],
            tau_grid=[float(value) for value in retrieval["tau_grid"]],
            volatility_quantile_grid=[
                None if value is None else float(value) for value in retrieval["volatility_quantile_grid"]
            ],
        )
        params_path = self.artifacts.write_retrieval_params(
            model_name=retrieval_cfg.output_model_name,
            selected_params=best_params,
            score_grid=tuning_scores,
            output_model_name=retrieval_cfg.output_model_name,
        )
        return RetrievalTuningResult(
            params=best_params,
            score_grid=tuning_scores,
            output_model_name=retrieval_cfg.output_model_name,
            path=params_path,
        )

    def apply(self, base_model: str = "nbeatsx", split: str = "test") -> Path:
        if split not in {"validation", "test"}:
            raise ValueError(f"Unsupported retrieval apply split: {split!r}")

        retrieval_cfg = self._retrieval_config()
        params_payload = self.artifacts.load_retrieval_params(retrieval_cfg.output_model_name)
        params = RetrievalParams(
            alpha=float(params_payload["selected_params"]["alpha"]),
            tau=float(params_payload["selected_params"]["tau"]),
            predicted_volatility_threshold=params_payload["selected_params"]["predicted_volatility_threshold"],
        )

        feature_df = self.prepared_dataset.feature_df
        benchmark_seed = self._benchmark_seed()
        base_predictions = self._load_predictions(
            feature_df=feature_df,
            forecast_days=self.prepared_dataset.split_days(split),
            split_name=split,
            seed=benchmark_seed,
            model_name=base_model,
        )
        initial_memory_predictions = self._initial_memory_predictions(
            feature_df=feature_df,
            base_model=base_model,
            target_split=split,
            seed=benchmark_seed,
        )
        corrected = apply_residual_retrieval(
            feature_df=feature_df,
            base_predictions=base_predictions,
            initial_memory_predictions=initial_memory_predictions,
            config=retrieval_cfg,
            params=params,
        )
        self.prepared_dataset.schema.validate_prediction_frame(corrected, require_metadata=False)
        output_path = self.artifacts.prediction(retrieval_cfg.output_model_name, split, benchmark_seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_parquet(output_path, index=False)
        return output_path

    def _initial_memory_predictions(
        self,
        *,
        feature_df: pd.DataFrame,
        base_model: str,
        target_split: str,
        seed: int,
    ) -> pd.DataFrame:
        warmup_predictions = self._load_predictions(
            feature_df=feature_df,
            forecast_days=self._warmup_days(),
            split_name="retrieval_warmup",
            seed=seed,
            model_name=base_model,
        )
        if target_split == "validation":
            return warmup_predictions

        validation_predictions = self._load_predictions(
            feature_df=feature_df,
            forecast_days=self.prepared_dataset.split_days("validation"),
            split_name="validation",
            seed=seed,
            model_name=base_model,
        )
        return pd.concat([warmup_predictions, validation_predictions], axis=0, ignore_index=True)

    def _load_predictions(
        self,
        *,
        feature_df: pd.DataFrame,
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
        return self.prediction_loader(feature_df, forecast_days, split_name, seed, model_name, variant)

    def _warmup_days(self) -> list[pd.Timestamp]:
        retrieval = self.config.retrieval
        validation_start = self.prepared_dataset.split_boundaries["validation_start"]
        warmup_start = validation_start - pd.Timedelta(days=int(retrieval["warmup_days"]))
        warmup_end = validation_start - pd.Timedelta(days=1)
        return self.prepared_dataset.days_between(start=warmup_start, end=warmup_end)

    def _retrieval_config(self) -> RetrievalConfig:
        retrieval = self.config.retrieval
        return RetrievalConfig(
            history_days=int(retrieval["history_days"]),
            price_weight=float(retrieval["weights"]["price"]),
            load_weight=float(retrieval["weights"]["load"]),
            calendar_weight=float(retrieval["weights"]["calendar"]),
            top_k=int(retrieval["top_k"]),
            min_gap_days=int(retrieval["min_gap_days"]),
            residual_clip_quantile=float(retrieval["residual_clip_quantile"]),
            horizon=int(self.config.prediction_horizon),
            load_columns=tuple(self.prepared_dataset.schema.retrieval_load_columns()),
            calendar_columns=tuple(self.prepared_dataset.schema.retrieval_calendar_columns()),
            output_model_name=self.config.retrieval_output_model_name,
        )

    def _benchmark_seed(self) -> int:
        return int(self.config.project["benchmark_seed"])
