from __future__ import annotations

from pathlib import Path

import yaml

from pjm_forecast.config import load_config
from pjm_forecast.kaggle import build_kaggle_config


def test_project_root_override_changes_resolved_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "root_override": str(tmp_path / "override_root"),
                    "directories": {"artifact_dir": "artifacts"},
                },
                "dataset": {},
                "features": {},
                "backtest": {},
                "tuning": {},
                "models": {},
                "report": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.resolve_path("artifacts") == (tmp_path / "override_root" / "artifacts").resolve()


def test_build_kaggle_config_patches_fast_mode(tmp_path: Path) -> None:
    base_config = Path("configs/pjm_day_ahead_v1.yaml").resolve()
    output_config = tmp_path / "kaggle.yaml"
    built = build_kaggle_config(
        base_config_path=base_config,
        output_config_path=output_config,
        output_root=tmp_path / "run",
        local_csv_path=tmp_path / "PJM.csv",
        fast_mode=True,
    )
    payload = yaml.safe_load(built.read_text(encoding="utf-8"))
    assert payload["project"]["root_override"] == str((tmp_path / "run"))
    assert payload["dataset"]["local_csv_path"] == str(tmp_path / "PJM.csv")
    assert payload["tuning"]["n_trials"] <= 4
    assert payload["tuning"]["use_ensemble_in_tuning"] is False
    assert payload["backtest"]["benchmark_models"] == ["nbeatsx"]
