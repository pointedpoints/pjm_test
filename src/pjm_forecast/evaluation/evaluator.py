from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .dm import dm_test
from .metrics import compute_metrics, compute_quantile_diagnostics
from .reporting import plot_high_volatility_week, plot_hourly_mae
from .scenarios import compute_scenario_diagnostics
from pjm_forecast.prediction_contract import point_prediction_view, quantile_values
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions


@dataclass(frozen=True)
class LoadedPredictionRun:
    name: str
    path: Path
    model: str
    split: str
    seed: int
    variant: str | None
    raw_frame: pd.DataFrame
    frame: pd.DataFrame


@dataclass(frozen=True)
class EvaluationBundle:
    split: str
    runs: list[LoadedPredictionRun]

    def frames_by_run(self) -> dict[str, pd.DataFrame]:
        return {run.name: run.frame for run in self.runs}


class _PredictionRunLike(Protocol):
    name: str
    path: Path
    model: str
    split: str
    seed: int
    variant: str | None


class _ArtifactStoreLike(Protocol):
    def prediction_runs(self, split: str) -> list[_PredictionRunLike]: ...

    def write_metrics(self, split: str, metrics_df: pd.DataFrame) -> Path: ...

    def write_quantile_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_scenario_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_dm(self, split: str, dm_df: pd.DataFrame) -> Path: ...

    def write_plot(self, split: str, kind: str, plot_writer) -> Path: ...


class Evaluator:
    def __init__(self, schema, artifacts: _ArtifactStoreLike) -> None:
        self.schema = schema
        self.artifacts = artifacts

    def load_runs(self, split: str) -> EvaluationBundle:
        loaded_runs: list[LoadedPredictionRun] = []
        for run in self.artifacts.prediction_runs(split):
            raw_prediction_df = pd.read_parquet(run.path)
            if raw_prediction_df.empty:
                continue
            self.schema.validate_prediction_frame(raw_prediction_df, require_metadata=False, model_name=run.model)
            if str(raw_prediction_df["split"].iloc[0]) != split:
                raise ValueError(f"Prediction run {run.name!r} metadata does not match requested split={split!r}.")
            prediction_df = self._postprocess_run_frame(run, raw_prediction_df)
            loaded_runs.append(
                LoadedPredictionRun(
                    name=run.name,
                    path=run.path,
                    model=run.model,
                    split=run.split,
                    seed=run.seed,
                    variant=run.variant,
                    raw_frame=raw_prediction_df,
                    frame=prediction_df,
                )
            )

        if not loaded_runs:
            raise FileNotFoundError(f"No prediction parquet files found for split={split}.")
        return EvaluationBundle(split=split, runs=loaded_runs)

    def _postprocess_run_frame(self, run: _PredictionRunLike, frame: pd.DataFrame) -> pd.DataFrame:
        postprocess_cfg = self.schema.config.report.get("quantile_postprocess", {})
        if not postprocess_cfg:
            return frame

        monotonic = bool(postprocess_cfg.get("monotonic", True))
        calibration_cfg = postprocess_cfg.get("calibration", {})
        calibration_method = str(calibration_cfg.get("method", "cqr"))
        calibration_group_by = calibration_cfg.get("group_by")
        calibration_interval_coverage_floors = calibration_cfg.get("interval_coverage_floors")
        calibration_min_group_size = int(calibration_cfg.get("min_group_size", 1))
        calibration_regime_score_column = str(calibration_cfg.get("regime_score_column", "spike_score"))
        calibration_regime_threshold = float(calibration_cfg.get("regime_threshold", 0.67))
        calibration_frame: pd.DataFrame | None = None
        if run.split == "test" and bool(calibration_cfg.get("enabled", False)):
            source_split = str(calibration_cfg.get("source_split", "validation"))
            calibration_frame = self._load_matching_run_frame(
                split=source_split,
                model=run.model,
                seed=run.seed,
                variant=run.variant,
            )
            if calibration_frame is not None and quantile_values(frame) != quantile_values(calibration_frame):
                raise ValueError(
                    f"Calibration frame quantile grid does not match run {run.name!r}. "
                    f"target={quantile_values(frame)}, calibration={quantile_values(calibration_frame)}"
                )
        return postprocess_quantile_predictions(
            frame,
            monotonic=monotonic,
            calibration_frame=calibration_frame,
            calibration_method=calibration_method,
            calibration_group_by=calibration_group_by,
            calibration_interval_coverage_floors=calibration_interval_coverage_floors,
            calibration_min_group_size=calibration_min_group_size,
            calibration_regime_score_column=calibration_regime_score_column,
            calibration_regime_threshold=calibration_regime_threshold,
        )

    def _load_matching_run_frame(
        self,
        *,
        split: str,
        model: str,
        seed: int,
        variant: str | None,
    ) -> pd.DataFrame | None:
        for run in self.artifacts.prediction_runs(split):
            if run.model != model or run.seed != seed or run.variant != variant:
                continue
            frame = pd.read_parquet(run.path)
            self.schema.validate_prediction_frame(frame, require_metadata=False, model_name=run.model)
            return frame
        return None

    def _load_matching_processed_run_frame(
        self,
        *,
        split: str,
        model: str,
        seed: int,
        variant: str | None,
    ) -> pd.DataFrame | None:
        for run in self.artifacts.prediction_runs(split):
            if run.model != model or run.seed != seed or run.variant != variant:
                continue
            raw_frame = pd.read_parquet(run.path)
            self.schema.validate_prediction_frame(raw_frame, require_metadata=False, model_name=run.model)
            return self._postprocess_run_frame(run, raw_frame)
        return None

    def compute_metrics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            rows.append(
                {
                    "run": run.name,
                    "model": run.model,
                    "seed": run.seed,
                    **compute_metrics(run.frame),
                }
            )
        metrics_df = pd.DataFrame(rows)
        primary_metric = "pinball" if "pinball" in metrics_df.columns and metrics_df["pinball"].notna().any() else "mae"
        metrics_df = metrics_df.sort_values([primary_metric, "model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_metrics(bundle.split, metrics_df)
        return metrics_df

    def compute_quantile_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            raw_diagnostics = compute_quantile_diagnostics(run.raw_frame)
            post_diagnostics = compute_quantile_diagnostics(run.frame)
            row: dict[str, object] = {
                "run": run.name,
                "model": run.model,
                "seed": run.seed,
                "has_quantiles": bool(post_diagnostics["has_quantiles"]),
            }
            for key, value in raw_diagnostics.items():
                if key == "has_quantiles":
                    continue
                row[f"raw_{key}"] = value
            for key, value in post_diagnostics.items():
                if key == "has_quantiles":
                    continue
                row[f"post_{key}"] = value
            rows.append(row)
        diagnostics_df = pd.DataFrame(rows)
        diagnostics_df = diagnostics_df.sort_values(["model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_quantile_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_scenario_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        scenario_cfg = self.schema.config.report.get("scenario_evaluation", {})
        rows = []
        for run in bundle.runs:
            if bundle.split == "test" and bool(scenario_cfg.get("enabled", False)):
                train_frame = self._load_matching_processed_run_frame(
                    split=str(scenario_cfg.get("source_split", "validation")),
                    model=run.model,
                    seed=run.seed,
                    variant=run.variant,
                )
            else:
                train_frame = run.frame
            diagnostics = compute_scenario_diagnostics(
                train_frame,
                run.frame,
                family=str(scenario_cfg.get("copula_family", "student_t")),
                n_samples=int(scenario_cfg.get("n_samples", 256)),
                dof_grid=scenario_cfg.get("dof_grid"),
                random_seed=int(scenario_cfg.get("random_seed", run.seed)),
                tail_policy=str(scenario_cfg.get("tail_policy", "flat")),
            )
            row: dict[str, object] = {"run": run.name, "model": run.model, "seed": run.seed, **diagnostics}
            rows.append(row)
        diagnostics_df = pd.DataFrame(rows).sort_values(["model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_scenario_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_dm(self, bundle: EvaluationBundle) -> pd.DataFrame:
        dm_rows: list[dict[str, object]] = []
        for left_index, left_run in enumerate(bundle.runs):
            for right_run in bundle.runs[left_index + 1 :]:
                left_point = point_prediction_view(left_run.frame)
                right_point = point_prediction_view(right_run.frame)
                if not left_point["ds"].equals(right_point["ds"]):
                    continue
                dm_rows.append(
                    {
                        "left": left_run.name,
                        "right": right_run.name,
                        **dm_test(
                            y_true=left_point["y"].to_numpy(),
                            y_pred_a=left_point["y_pred"].to_numpy(),
                            y_pred_b=right_point["y_pred"].to_numpy(),
                        ),
                    }
                )
        dm_df = pd.DataFrame(dm_rows, columns=["left", "right", "statistic", "p_value"])
        self.artifacts.write_dm(bundle.split, dm_df)
        return dm_df

    def render_plots(self, bundle: EvaluationBundle, metrics_df: pd.DataFrame, split: str) -> dict[str, Path]:
        prediction_frames = bundle.frames_by_run()
        plot_paths = {
            "hourly_mae": self.artifacts.write_plot(
                split,
                "hourly_mae",
                lambda output_path: plot_hourly_mae(prediction_frames, output_path),
            )
        }
        best_run_name = str(metrics_df.iloc[0]["run"])
        plot_paths["high_vol_week"] = self.artifacts.write_plot(
            split,
            "high_vol_week",
            lambda output_path: plot_high_volatility_week(prediction_frames[best_run_name], output_path),
        )
        return plot_paths
