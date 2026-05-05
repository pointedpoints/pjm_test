from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .dm import dm_test
from .event_risk_tail_overlay import apply_event_risk_tail_overlay, fit_event_risk_tail_overlay
from .metrics import compute_metrics, compute_quantile_diagnostics
from .normal_day import compute_normal_day_diagnostics
from .regime_metrics import compute_regime_metrics
from .reporting import plot_high_volatility_week, plot_hourly_mae
from .relative_error import compute_relative_error_diagnostics
from .scenarios import compute_scenario_diagnostics
from .scorecard import build_experiment_scorecard_row
from .spike_score_diagnostics import compute_spike_score_diagnostics
from .tail_regime import compute_tail_regime_diagnostics
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

    def write_regime_metrics(self, split: str, regime_metrics_df: pd.DataFrame) -> Path: ...

    def write_spike_score_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_normal_day_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_relative_error(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_tail_regime_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_experiment_scorecard(self, split: str, scorecard_df: pd.DataFrame) -> Path: ...

    def write_scenario_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_dm(self, split: str, dm_df: pd.DataFrame) -> Path: ...

    def write_plot(self, split: str, kind: str, plot_writer) -> Path: ...


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _report_section(schema: object, section_name: str) -> Mapping[str, object]:
    config = getattr(schema, "config", None)
    if config is None:
        return {}
    report = config.get("report", {}) if isinstance(config, Mapping) else getattr(config, "report", {})
    if not isinstance(report, Mapping):
        return {}
    section = report.get(section_name, {})
    return section if isinstance(section, Mapping) else {}


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
        median_bias_cfg = postprocess_cfg.get("median_bias", {})
        median_bias_enabled = bool(median_bias_cfg.get("enabled", False))
        median_bias_group_by = median_bias_cfg.get("group_by", calibration_group_by)
        median_bias_min_group_size = int(median_bias_cfg.get("min_group_size", calibration_min_group_size))
        median_bias_regime_score_column = str(median_bias_cfg.get("regime_score_column", calibration_regime_score_column))
        median_bias_regime_threshold = float(median_bias_cfg.get("regime_threshold", calibration_regime_threshold))
        median_bias_max_abs_adjustment = median_bias_cfg.get("max_abs_adjustment")
        if median_bias_max_abs_adjustment is not None:
            median_bias_max_abs_adjustment = float(median_bias_max_abs_adjustment)
        event_tail_cfg = postprocess_cfg.get("event_risk_tail_overlay", {})
        event_tail_enabled = bool(event_tail_cfg.get("enabled", False))
        calibration_frame: pd.DataFrame | None = None
        event_calibration_frame: pd.DataFrame | None = None
        if run.split == "test" and (bool(calibration_cfg.get("enabled", False)) or median_bias_enabled):
            source_split = str(median_bias_cfg.get("source_split", calibration_cfg.get("source_split", "validation")))
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
        if run.split == "test" and event_tail_enabled:
            event_source_split = str(event_tail_cfg.get("source_split", "validation"))
            event_calibration_frame = self._load_matching_run_frame(
                split=event_source_split,
                model=run.model,
                seed=run.seed,
                variant=run.variant,
            )
            if event_calibration_frame is not None and quantile_values(frame) != quantile_values(event_calibration_frame):
                raise ValueError(
                    f"Event-risk overlay frame quantile grid does not match run {run.name!r}. "
                    f"target={quantile_values(frame)}, calibration={quantile_values(event_calibration_frame)}"
                )
        processed = postprocess_quantile_predictions(
            frame,
            monotonic=monotonic,
            calibration_frame=calibration_frame,
            calibration_method=calibration_method,
            calibration_group_by=calibration_group_by,
            calibration_interval_coverage_floors=calibration_interval_coverage_floors,
            calibration_min_group_size=calibration_min_group_size,
            calibration_regime_score_column=calibration_regime_score_column,
            calibration_regime_threshold=calibration_regime_threshold,
            median_bias_enabled=median_bias_enabled,
            median_bias_group_by=median_bias_group_by,
            median_bias_min_group_size=median_bias_min_group_size,
            median_bias_regime_score_column=median_bias_regime_score_column,
            median_bias_regime_threshold=median_bias_regime_threshold,
            median_bias_max_abs_adjustment=median_bias_max_abs_adjustment,
        )
        event_risk_score_column = str(event_tail_cfg.get("risk_score_column", calibration_regime_score_column))
        if (
            run.split == "test"
            and event_tail_enabled
            and event_calibration_frame is not None
            and event_risk_score_column in processed.columns
            and event_risk_score_column in event_calibration_frame.columns
        ):
            overlay = fit_event_risk_tail_overlay(
                event_calibration_frame,
                risk_score_column=event_risk_score_column,
                risk_threshold_quantile=float(event_tail_cfg.get("risk_threshold_quantile", 0.90)),
                risk_aggregation=str(event_tail_cfg.get("risk_aggregation", "mean")),
                residual_quantile=float(event_tail_cfg.get("residual_quantile", 1.0)),
                max_uplift=_optional_float(event_tail_cfg.get("max_uplift")),
                target_quantiles=event_tail_cfg.get("target_quantiles", [0.99, 0.995]),
            )
            processed = apply_event_risk_tail_overlay(processed, overlay)
        return processed

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

    def compute_regime_metrics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            regime_df = compute_regime_metrics(run.frame)
            regime_df.insert(0, "seed", run.seed)
            regime_df.insert(0, "model", run.model)
            regime_df.insert(0, "run", run.name)
            rows.append(regime_df)
        metrics_df = pd.concat(rows, axis=0, ignore_index=True)
        metrics_df = metrics_df.sort_values(["model", "seed", "run", "regime"]).reset_index(drop=True)
        self.artifacts.write_regime_metrics(bundle.split, metrics_df)
        return metrics_df

    def compute_spike_score_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        calibration_cfg = self.schema.config.report.get("quantile_postprocess", {}).get("calibration", {})
        score_column = str(calibration_cfg.get("regime_score_column", "spike_score"))
        threshold = float(calibration_cfg.get("regime_threshold", 0.67))
        rows = []
        for run in bundle.runs:
            rows.append(
                {
                    "run": run.name,
                    "model": run.model,
                    "seed": run.seed,
                    **compute_spike_score_diagnostics(run.frame, score_column=score_column, threshold=threshold),
                }
            )
        diagnostics_df = pd.DataFrame(rows).sort_values(["model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_spike_score_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_normal_day_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        normal_cfg = _report_section(self.schema, "normal_day_evaluation")
        rows = []
        for run in bundle.runs:
            diagnostics = compute_normal_day_diagnostics(
                run.frame,
                actual_daily_max_quantile=float(normal_cfg.get("actual_daily_max_quantile", 0.95)),
                low_risk_score_column=str(normal_cfg.get("low_risk_score_column", "spike_score")),
                low_risk_threshold=float(normal_cfg.get("low_risk_threshold", 0.50)),
                low_risk_aggregation=str(normal_cfg.get("low_risk_aggregation", "mean")),
            )
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        diagnostics_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
        if not diagnostics_df.empty:
            diagnostics_df = diagnostics_df.sort_values(["model", "seed", "run", "segment"]).reset_index(drop=True)
        self.artifacts.write_normal_day_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_relative_error(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            diagnostics = compute_relative_error_diagnostics(run.frame)
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        diagnostics_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
        if not diagnostics_df.empty:
            diagnostics_df = diagnostics_df.sort_values(["model", "seed", "run", "slice_type", "slice"]).reset_index(drop=True)
        self.artifacts.write_relative_error(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_tail_regime_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            diagnostics = compute_tail_regime_diagnostics(run.frame)
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        diagnostics_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
        if not diagnostics_df.empty:
            diagnostics_df = diagnostics_df.sort_values(["model", "seed", "run", "regime"]).reset_index(drop=True)
        self.artifacts.write_tail_regime_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df

    def compute_experiment_scorecard(
        self,
        bundle: EvaluationBundle,
        metrics_df: pd.DataFrame,
        relative_error_df: pd.DataFrame,
        tail_regime_df: pd.DataFrame,
        normal_day_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            metric_match = metrics_df.loc[metrics_df["run"].eq(run.name)]
            metrics = metric_match.iloc[0].to_dict() if not metric_match.empty else {}
            relative_error = relative_error_df.loc[relative_error_df["run"].eq(run.name)] if not relative_error_df.empty else pd.DataFrame()
            tail_regime = tail_regime_df.loc[tail_regime_df["run"].eq(run.name)] if not tail_regime_df.empty else pd.DataFrame()
            normal_day = (
                normal_day_df.loc[normal_day_df["run"].eq(run.name)]
                if normal_day_df is not None and not normal_day_df.empty
                else pd.DataFrame()
            )
            rows.append(
                build_experiment_scorecard_row(
                    run_name=run.name,
                    model=run.model,
                    seed=run.seed,
                    metrics=metrics,
                    relative_error=relative_error,
                    tail_regime=tail_regime,
                    normal_day=normal_day,
                )
            )
        scorecard_df = pd.DataFrame(rows)
        primary_metric = "pinball" if "pinball" in scorecard_df.columns else "mae"
        if primary_metric in scorecard_df.columns:
            scorecard_df = scorecard_df.sort_values([primary_metric, "model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_experiment_scorecard(bundle.split, scorecard_df)
        return scorecard_df

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
