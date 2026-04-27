from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.config import load_config
from pjm_forecast.evaluation.evaluator import Evaluator
from pjm_forecast.evaluation.scenarios import compute_scenario_diagnostics
from pjm_forecast.prepared_data import FeatureSchema


@dataclass(frozen=True)
class _Run:
    name: str
    path: Path
    model: str
    split: str
    seed: int
    variant: str | None = None


class _Artifacts:
    def __init__(self, root: Path, runs_by_split: dict[str, list[_Run]]) -> None:
        self.root = root
        self.runs_by_split = runs_by_split

    def prediction_runs(self, split: str) -> list[_Run]:
        return list(self.runs_by_split.get(split, []))

    def write_metrics(self, split: str, metrics_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_metrics.csv"
        metrics_df.to_csv(path, index=False)
        return path

    def write_quantile_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_quantile_diagnostics.csv"
        diagnostics_df.to_csv(path, index=False)
        return path

    def write_scenario_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_scenario_diagnostics.csv"
        diagnostics_df.to_csv(path, index=False)
        return path

    def write_regime_metrics(self, split: str, regime_metrics_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_regime_metrics.csv"
        regime_metrics_df.to_csv(path, index=False)
        return path

    def write_spike_score_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_spike_score_diagnostics.csv"
        diagnostics_df.to_csv(path, index=False)
        return path

    def write_dm(self, split: str, dm_df: pd.DataFrame) -> Path:
        path = self.root / f"{split}_dm.csv"
        dm_df.to_csv(path, index=False)
        return path

    def write_plot(self, split: str, kind: str, plot_writer) -> Path:
        path = self.root / f"{split}_{kind}.png"
        plot_writer(path)
        return path


def _quantile_frame(
    *,
    split: str,
    quantiles: list[float],
    y_true: list[float],
    predicted_by_quantile: dict[float, list[float]],
) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01 00:00:00", periods=len(y_true), freq="h")
    rows = []
    for index, ts in enumerate(ds):
        for quantile in quantiles:
            rows.append(
                {
                    "ds": ts,
                    "y": y_true[index],
                    "y_pred": predicted_by_quantile[quantile][index],
                    "model": "quantile_dummy",
                    "split": split,
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    return pd.DataFrame(rows)


def _point_frame(*, split: str, model: str = "seasonal_naive") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=4, freq="h"),
            "y": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.5, 2.5, 3.5, 4.5],
            "model": [model] * 4,
            "split": [split] * 4,
            "seed": [7] * 4,
            "quantile": [pd.NA] * 4,
            "metadata": ["{}"] * 4,
        }
    )


def test_evaluator_writes_quantile_diagnostics_for_raw_and_post_frames(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["report"]["quantile_postprocess"] = {
        "monotonic": True,
        "calibration": {
            "enabled": True,
            "source_split": "validation",
            "method": "cqr",
        },
    }
    config.raw["report"]["scenario_evaluation"] = {
        "enabled": True,
        "source_split": "validation",
        "copula_family": "student_t",
        "dof_grid": [3.0, 5.0],
        "n_samples": 32,
        "random_seed": 7,
    }
    config.raw["models"]["quantile_dummy"] = {
        "loss_name": "mqloss",
        "quantiles": [0.1, 0.5, 0.9],
    }
    schema = FeatureSchema(config)

    validation_frame = _quantile_frame(
        split="validation",
        quantiles=[0.1, 0.5, 0.9],
        y_true=[10.0, 20.0],
        predicted_by_quantile={
            0.1: [8.0, 18.0],
            0.5: [9.0, 19.0],
            0.9: [10.0, 20.0],
        },
    )
    test_quantile_frame = _quantile_frame(
        split="test",
        quantiles=[0.1, 0.5, 0.9],
        y_true=[12.0, 18.0],
        predicted_by_quantile={
            0.1: [11.0, 17.0],
            0.5: [10.0, 18.5],
            0.9: [9.0, 20.0],
        },
    )
    test_point_frame = _point_frame(split="test")

    validation_path = tmp_path / "quantile_validation.parquet"
    test_quantile_path = tmp_path / "quantile_test.parquet"
    test_point_path = tmp_path / "point_test.parquet"
    validation_frame.to_parquet(validation_path, index=False)
    test_quantile_frame.to_parquet(test_quantile_path, index=False)
    test_point_frame.to_parquet(test_point_path, index=False)

    artifacts = _Artifacts(
        tmp_path,
        {
            "validation": [_Run("quantile_dummy_validation_seed7", validation_path, "quantile_dummy", "validation", 7)],
            "test": [
                _Run("quantile_dummy_test_seed7", test_quantile_path, "quantile_dummy", "test", 7),
                _Run("seasonal_naive_test_seed7", test_point_path, "seasonal_naive", "test", 7),
            ],
        },
    )
    evaluator = Evaluator(schema=schema, artifacts=artifacts)

    bundle = evaluator.load_runs("test")
    evaluator.compute_metrics(bundle)
    diagnostics_df = evaluator.compute_quantile_diagnostics(bundle)
    regime_df = evaluator.compute_regime_metrics(bundle)
    spike_df = evaluator.compute_spike_score_diagnostics(bundle)
    scenario_df = evaluator.compute_scenario_diagnostics(bundle)

    quantile_row = diagnostics_df.loc[diagnostics_df["run"] == "quantile_dummy_test_seed7"].iloc[0]
    point_row = diagnostics_df.loc[diagnostics_df["run"] == "seasonal_naive_test_seed7"].iloc[0]
    scenario_row = scenario_df.loc[scenario_df["run"] == "quantile_dummy_test_seed7"].iloc[0]

    assert bool(quantile_row["has_quantiles"]) is True
    assert quantile_row["raw_crossing_rate"] > quantile_row["post_crossing_rate"]
    assert quantile_row["raw_pinball"] != quantile_row["post_pinball"]
    assert bool(point_row["has_quantiles"]) is False
    assert np.isnan(point_row["raw_crossing_rate"])
    assert np.isnan(point_row["post_coverage_80"])
    assert {"all", "normal", "daily_max"}.issubset(set(regime_df["regime"]))
    assert "p50_mae" in set(regime_df.columns)
    assert "has_spike_score" in set(spike_df.columns)
    assert bool(scenario_row["has_scenarios"]) is True
    assert scenario_row["energy_score"] >= 0.0
    assert (tmp_path / "test_quantile_diagnostics.csv").exists()
    assert (tmp_path / "test_regime_metrics.csv").exists()
    assert (tmp_path / "test_spike_score_diagnostics.csv").exists()
    assert (tmp_path / "test_scenario_diagnostics.csv").exists()
    assert (tmp_path / "test_metrics.csv").exists()


def test_evaluator_rejects_mismatched_calibration_quantile_grid(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    config.raw["report"]["quantile_postprocess"] = {
        "monotonic": True,
        "calibration": {
            "enabled": True,
            "source_split": "validation",
            "method": "cqr",
        },
    }
    schema = FeatureSchema(config)

    validation_frame = _quantile_frame(
        split="validation",
        quantiles=[0.2, 0.5, 0.8],
        y_true=[10.0, 20.0],
        predicted_by_quantile={
            0.2: [8.0, 18.0],
            0.5: [9.0, 19.0],
            0.8: [10.0, 20.0],
        },
    )
    test_frame = _quantile_frame(
        split="test",
        quantiles=[0.1, 0.5, 0.9],
        y_true=[12.0, 18.0],
        predicted_by_quantile={
            0.1: [8.0, 17.0],
            0.5: [9.0, 18.0],
            0.9: [10.0, 19.0],
        },
    )
    validation_frame["model"] = "quantile_custom"
    test_frame["model"] = "quantile_custom"

    validation_path = tmp_path / "quantile_custom_validation.parquet"
    test_path = tmp_path / "quantile_custom_test.parquet"
    validation_frame.to_parquet(validation_path, index=False)
    test_frame.to_parquet(test_path, index=False)

    artifacts = _Artifacts(
        tmp_path,
        {
            "validation": [_Run("quantile_custom_validation_seed7", validation_path, "quantile_custom", "validation", 7)],
            "test": [_Run("quantile_custom_test_seed7", test_path, "quantile_custom", "test", 7)],
        },
    )
    evaluator = Evaluator(schema=schema, artifacts=artifacts)

    with pytest.raises(ValueError, match="Calibration frame quantile grid does not match run"):
        evaluator.load_runs("test")
