from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from pjm_forecast.workspace import Workspace


def test_workspace_evaluate_discovers_prediction_files_without_filename_regex(tmp_path: Path) -> None:
    base_config = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    base_config["project"]["root_override"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    workspace = Workspace.open(config_path)

    prediction_df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=24, freq="h"),
            "y": [1.0] * 24,
            "y_pred": [1.5] * 24,
            "model": ["nbeatsx"] * 24,
            "split": ["test"] * 24,
            "seed": [7] * 24,
            "quantile": [pd.NA] * 24,
            "metadata": ["{}"] * 24,
        }
    )
    prediction_df.to_parquet(workspace.directories["prediction_dir"] / "custom_name_without_contract.parquet", index=False)

    workspace.evaluate("test")

    metrics_df = pd.read_csv(workspace.artifacts.metrics("test"))
    assert "custom_name_without_contract" in set(metrics_df["run"])
