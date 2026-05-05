from pathlib import Path
from uuid import uuid4

import pandas as pd

from pjm_forecast.evaluation.evaluator import EvaluationBundle, Evaluator, LoadedPredictionRun
from pjm_forecast.evaluation.scorecard import build_experiment_scorecard_row
from pjm_forecast.workspace import ArtifactStore
from scripts.experiments.scorecard_baselines import baseline_model_names


def test_artifact_store_writes_scorecard_outputs() -> None:
    tmp_path = Path(".tmp") / "scorecard_tests" / uuid4().hex
    store = ArtifactStore(
        directories={
            "metrics_dir": tmp_path / "metrics",
            "plots_dir": tmp_path / "plots",
            "prediction_dir": tmp_path / "predictions",
            "processed_data_dir": tmp_path / "processed",
            "hyperparameter_dir": tmp_path / "hyperparameters",
            "report_dir": tmp_path / "report",
            "artifact_dir": tmp_path,
        }
    )
    frame = pd.DataFrame([{"run": "model_test_seed7", "q50_wape": 0.25}])

    normal_day_path = store.write_normal_day_diagnostics("test", frame)
    relative_path = store.write_relative_error("test", frame)
    tail_path = store.write_tail_regime_diagnostics("test", frame)
    scorecard_path = store.write_experiment_scorecard("test", frame)

    assert normal_day_path == tmp_path / "metrics" / "test_normal_day_diagnostics.csv"
    assert relative_path == tmp_path / "metrics" / "test_relative_error.csv"
    assert tail_path == tmp_path / "metrics" / "test_tail_regime_diagnostics.csv"
    assert scorecard_path == tmp_path / "metrics" / "test_experiment_scorecard.csv"
    assert pd.read_csv(scorecard_path).loc[0, "run"] == "model_test_seed7"


def test_scorecard_row_pulls_normal_and_tail_slices() -> None:
    relative = pd.DataFrame(
        [
            {
                "slice_type": "all",
                "slice": "all",
                "wape": 0.20,
                "smape": 0.30,
                "median_ape": 0.10,
                "p75_ape": 0.25,
                "p90_ape": 0.50,
            },
            {
                "slice_type": "actual_price_bin",
                "slice": "20-30",
                "wape": 0.25,
                "smape": 0.24,
                "median_ape": 0.18,
                "p75_ape": 0.33,
                "p90_ape": 0.54,
            },
            {
                "slice_type": "actual_price_bin",
                "slice": "30-50",
                "wape": 0.20,
                "smape": 0.20,
                "median_ape": 0.15,
                "p75_ape": 0.27,
                "p90_ape": 0.42,
            },
        ]
    )
    tail = pd.DataFrame(
        [
            {
                "regime": "all",
                "q99_upper_coverage": 0.984,
                "q995_upper_coverage": 0.990,
                "q99_excess_mean": 0.47,
                "q99_excess_max": 279.0,
            },
            {
                "regime": "actual_gt_p99",
                "q99_upper_coverage": 0.636,
                "q995_upper_coverage": 0.682,
                "q99_excess_mean": 33.0,
                "q99_excess_max": 279.0,
            },
        ]
    )
    metrics = {"mae": 10.0, "rmse": 20.0, "smape": 28.0, "pinball": 3.2}

    row = build_experiment_scorecard_row(
        run_name="nhits_test_seed7",
        model="nhits",
        seed=7,
        metrics=metrics,
        relative_error=relative,
        tail_regime=tail,
    )

    assert row["run"] == "nhits_test_seed7"
    assert row["q50_wape_all"] == 0.20
    assert row["q50_wape_20_30"] == 0.25
    assert row["q99_coverage_gt_p99"] == 0.636
    assert row["q99_excess_mean_gt_p99"] == 33.0


def test_scorecard_row_allows_baseline_without_tail_quantiles() -> None:
    relative = pd.DataFrame(
        [
            {
                "slice_type": "all",
                "slice": "all",
                "wape": 0.20,
                "smape": 0.30,
                "median_ape": 0.10,
                "p75_ape": 0.25,
                "p90_ape": 0.50,
            }
        ]
    )

    row = build_experiment_scorecard_row(
        run_name="lightgbm_q_validation_seed7",
        model="lightgbm_q",
        seed=7,
        metrics={"mae": 9.0, "rmse": 18.0, "smape": 20.0, "pinball": 2.5},
        relative_error=relative,
        tail_regime=pd.DataFrame(),
    )

    assert row["run"] == "lightgbm_q_validation_seed7"
    assert row["q50_wape_all"] == 0.20
    assert "q99_coverage_all" not in row


def test_scorecard_row_pulls_normal_day_relative_error_fields() -> None:
    normal_day = pd.DataFrame(
        [
            {
                "segment": "actual_normal_day",
                "q50_wape": 0.18,
                "median_ape": 0.12,
                "p75_ape": 0.24,
                "p90_ape": 0.41,
                "smape": 0.20,
            },
            {
                "segment": "forecast_low_risk_day",
                "q50_wape": 0.16,
                "median_ape": 0.10,
                "p75_ape": 0.22,
                "p90_ape": 0.38,
                "smape": 0.18,
            },
        ]
    )

    row = build_experiment_scorecard_row(
        run_name="nhits_normal_cap_validation_seed7",
        model="nhits_normal_cap",
        seed=7,
        metrics={"mae": 8.0, "rmse": 15.0, "smape": 25.0, "pinball": 2.8},
        relative_error=pd.DataFrame(),
        tail_regime=pd.DataFrame(),
        normal_day=normal_day,
    )

    assert row["actual_normal_day_q50_wape"] == 0.18
    assert row["actual_normal_day_median_ape"] == 0.12
    assert row["actual_normal_day_p75_ape"] == 0.24
    assert row["actual_normal_day_p90_ape"] == 0.41
    assert row["actual_normal_day_smape"] == 0.20
    assert row["forecast_low_risk_day_q50_wape"] == 0.16
    assert row["forecast_low_risk_day_median_ape"] == 0.10
    assert row["forecast_low_risk_day_p75_ape"] == 0.22
    assert row["forecast_low_risk_day_p90_ape"] == 0.38
    assert row["forecast_low_risk_day_smape"] == 0.18


def test_evaluator_writes_scorecard_artifacts() -> None:
    artifacts = _CapturingArtifacts()
    evaluator = Evaluator(schema=object(), artifacts=artifacts)
    bundle = EvaluationBundle(
        split="test",
        runs=[
            LoadedPredictionRun(
                name="nhits_test_seed7",
                path=Path("nhits_test_seed7.parquet"),
                model="nhits",
                split="test",
                seed=7,
                variant=None,
                raw_frame=_prediction_frame(),
                frame=_prediction_frame(),
            )
        ],
    )

    metrics = evaluator.compute_metrics(bundle)
    normal_day = evaluator.compute_normal_day_diagnostics(bundle)
    relative_error = evaluator.compute_relative_error(bundle)
    tail_regime = evaluator.compute_tail_regime_diagnostics(bundle)
    scorecard = evaluator.compute_experiment_scorecard(bundle, metrics, relative_error, tail_regime, normal_day)

    assert set(artifacts.written) == {"metrics", "normal_day", "relative_error", "tail_regime", "scorecard"}
    assert artifacts.written["normal_day"].loc[0, "run"] == "nhits_test_seed7"
    assert normal_day.loc[0, "run"] == "nhits_test_seed7"
    assert artifacts.written["relative_error"].loc[0, "run"] == "nhits_test_seed7"
    assert artifacts.written["tail_regime"].loc[0, "run"] == "nhits_test_seed7"
    assert scorecard.loc[0, "q50_wape_all"] > 0
    assert scorecard.loc[0, "actual_normal_day_q50_wape"] > 0
    assert scorecard.loc[0, "q99_coverage_all"] <= 1.0


def test_experiment_scorecard_uses_matching_normal_day_rows_per_run() -> None:
    artifacts = _CapturingArtifacts()
    evaluator = Evaluator(schema=object(), artifacts=artifacts)
    bundle = EvaluationBundle(
        split="validation",
        runs=[
            _loaded_run("nhits_a_validation_seed7", "nhits_a", seed=7),
            _loaded_run("nhits_b_validation_seed11", "nhits_b", seed=11),
        ],
    )
    metrics = pd.DataFrame(
        [
            {
                "run": "nhits_a_validation_seed7",
                "model": "nhits_a",
                "seed": 7,
                "mae": 1.0,
                "rmse": 1.5,
                "smape": 2.0,
                "pinball": 2.0,
            },
            {
                "run": "nhits_b_validation_seed11",
                "model": "nhits_b",
                "seed": 11,
                "mae": 2.0,
                "rmse": 2.5,
                "smape": 3.0,
                "pinball": 1.0,
            },
        ]
    )
    normal_day = pd.DataFrame(
        [
            {
                "run": "nhits_a_validation_seed7",
                "segment": "actual_normal_day",
                "q50_wape": 0.11,
                "median_ape": 0.12,
                "p75_ape": 0.13,
                "p90_ape": 0.14,
                "smape": 0.15,
            },
            {
                "run": "nhits_a_validation_seed7",
                "segment": "forecast_low_risk_day",
                "q50_wape": 0.21,
                "median_ape": 0.22,
                "p75_ape": 0.23,
                "p90_ape": 0.24,
                "smape": 0.25,
            },
            {
                "run": "nhits_b_validation_seed11",
                "segment": "actual_normal_day",
                "q50_wape": 0.71,
                "median_ape": 0.72,
                "p75_ape": 0.73,
                "p90_ape": 0.74,
                "smape": 0.75,
            },
            {
                "run": "nhits_b_validation_seed11",
                "segment": "forecast_low_risk_day",
                "q50_wape": 0.81,
                "median_ape": 0.82,
                "p75_ape": 0.83,
                "p90_ape": 0.84,
                "smape": 0.85,
            },
        ]
    )

    scorecard = evaluator.compute_experiment_scorecard(
        bundle,
        metrics,
        pd.DataFrame(),
        pd.DataFrame(),
        normal_day,
    ).set_index("run")

    assert scorecard.loc["nhits_a_validation_seed7", "actual_normal_day_q50_wape"] == 0.11
    assert scorecard.loc["nhits_a_validation_seed7", "forecast_low_risk_day_q50_wape"] == 0.21
    assert scorecard.loc["nhits_b_validation_seed11", "actual_normal_day_q50_wape"] == 0.71
    assert scorecard.loc["nhits_b_validation_seed11", "forecast_low_risk_day_q50_wape"] == 0.81


def test_baseline_model_names_are_defaulted_for_comed_scorecard() -> None:
    assert baseline_model_names(None) == [
        "seasonal_naive",
        "lear",
        "lightgbm_quantile",
        "xgboost_quantile",
        "nhits_tail_grid_weighted_main",
    ]
    assert baseline_model_names("a,b") == ["a", "b"]


def _prediction_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h")
    rows = []
    for quantile, predictions in [
        (0.50, [11.0, 17.0, 24.0, 70.0]),
        (0.99, [14.0, 23.0, 28.0, 80.0]),
        (0.995, [15.0, 24.0, 29.0, 81.0]),
    ]:
        for ds, actual, prediction in zip(timestamps, [10.0, 20.0, 30.0, 100.0], predictions, strict=True):
            rows.append(
                {
                    "ds": ds,
                    "y": actual,
                    "y_pred": prediction,
                    "model": "nhits",
                    "split": "test",
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    return pd.DataFrame(rows)


def _loaded_run(name: str, model: str, *, seed: int) -> LoadedPredictionRun:
    return LoadedPredictionRun(
        name=name,
        path=Path(f"{name}.parquet"),
        model=model,
        split="validation",
        seed=seed,
        variant=None,
        raw_frame=pd.DataFrame(),
        frame=pd.DataFrame(),
    )


class _CapturingArtifacts:
    def __init__(self) -> None:
        self.written: dict[str, pd.DataFrame] = {}

    def write_metrics(self, split: str, metrics_df: pd.DataFrame) -> Path:
        self.written["metrics"] = metrics_df
        return Path(f"{split}_metrics.csv")

    def write_normal_day_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        self.written["normal_day"] = diagnostics_df
        return Path(f"{split}_normal_day_diagnostics.csv")

    def write_relative_error(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        self.written["relative_error"] = diagnostics_df
        return Path(f"{split}_relative_error.csv")

    def write_tail_regime_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        self.written["tail_regime"] = diagnostics_df
        return Path(f"{split}_tail_regime_diagnostics.csv")

    def write_experiment_scorecard(self, split: str, scorecard_df: pd.DataFrame) -> Path:
        self.written["scorecard"] = scorecard_df
        return Path(f"{split}_experiment_scorecard.csv")
