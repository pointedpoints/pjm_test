from pathlib import Path
from uuid import uuid4

import pandas as pd

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
