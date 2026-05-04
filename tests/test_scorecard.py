from pathlib import Path
from uuid import uuid4

import pandas as pd

from pjm_forecast.evaluation.scorecard import build_experiment_scorecard_row
from pjm_forecast.workspace import ArtifactStore


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

    relative_path = store.write_relative_error("test", frame)
    tail_path = store.write_tail_regime_diagnostics("test", frame)
    scorecard_path = store.write_experiment_scorecard("test", frame)

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
