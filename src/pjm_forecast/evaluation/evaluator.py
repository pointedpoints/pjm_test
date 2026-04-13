from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .dm import dm_test
from .metrics import compute_metrics
from .reporting import plot_high_volatility_week, plot_hourly_mae


@dataclass(frozen=True)
class LoadedPredictionRun:
    name: str
    path: Path
    model: str
    split: str
    seed: int
    variant: str | None
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

    def write_dm(self, split: str, dm_df: pd.DataFrame) -> Path: ...

    def write_plot(self, split: str, kind: str, plot_writer) -> Path: ...


class Evaluator:
    def __init__(self, schema, artifacts: _ArtifactStoreLike) -> None:
        self.schema = schema
        self.artifacts = artifacts

    def load_runs(self, split: str) -> EvaluationBundle:
        loaded_runs: list[LoadedPredictionRun] = []
        for run in self.artifacts.prediction_runs(split):
            prediction_df = pd.read_parquet(run.path)
            if prediction_df.empty:
                continue
            self.schema.validate_prediction_frame(prediction_df, require_metadata=False)
            if str(prediction_df["split"].iloc[0]) != split:
                raise ValueError(f"Prediction run {run.name!r} metadata does not match requested split={split!r}.")
            loaded_runs.append(
                LoadedPredictionRun(
                    name=run.name,
                    path=run.path,
                    model=run.model,
                    split=run.split,
                    seed=run.seed,
                    variant=run.variant,
                    frame=prediction_df,
                )
            )

        if not loaded_runs:
            raise FileNotFoundError(f"No prediction parquet files found for split={split}.")
        return EvaluationBundle(split=split, runs=loaded_runs)

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
        metrics_df = pd.DataFrame(rows).sort_values(["mae", "model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_metrics(bundle.split, metrics_df)
        return metrics_df

    def compute_dm(self, bundle: EvaluationBundle) -> pd.DataFrame:
        dm_rows: list[dict[str, object]] = []
        for left_index, left_run in enumerate(bundle.runs):
            for right_run in bundle.runs[left_index + 1 :]:
                if not left_run.frame["ds"].equals(right_run.frame["ds"]):
                    continue
                dm_rows.append(
                    {
                        "left": left_run.name,
                        "right": right_run.name,
                        **dm_test(
                            y_true=left_run.frame["y"].to_numpy(),
                            y_pred_a=left_run.frame["y_pred"].to_numpy(),
                            y_pred_b=right_run.frame["y_pred"].to_numpy(),
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
