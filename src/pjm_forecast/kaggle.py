from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def build_kaggle_config(
    base_config_path: str | Path,
    output_config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    local_csv_path: str | Path | None = None,
    fast_mode: bool = True,
    include_retrieval: bool = True,
) -> Path:
    base_config_path = Path(base_config_path)
    output_config_path = Path(output_config_path)
    output_root = Path(output_root) if output_root is not None else KAGGLE_WORKING_ROOT / "pjm_remaster_run"

    with base_config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)

    payload.setdefault("project", {})["root_override"] = str(output_root)
    payload.setdefault("dataset", {})["local_csv_path"] = None if local_csv_path is None else str(local_csv_path)
    payload.setdefault("tuning", {})["optuna_storage"] = (
        f"sqlite:///{(output_root / 'artifacts' / 'hyperparameters' / 'optuna_nbeatsx.db').as_posix()}"
    )
    payload["tuning"]["optuna_study_name"] = "nbeatsx_tuning"

    if fast_mode:
        payload["tuning"]["n_trials"] = min(int(payload["tuning"]["n_trials"]), 4)
        payload["tuning"]["use_ensemble_in_tuning"] = False
        payload["backtest"]["benchmark_models"] = ["nbeatsx"]
        payload["project"]["random_seeds"] = [payload["project"]["benchmark_seed"]]

    if not include_retrieval and "retrieval" in payload:
        payload["retrieval"]["enabled"] = False

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return output_config_path
