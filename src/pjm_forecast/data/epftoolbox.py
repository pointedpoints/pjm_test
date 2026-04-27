from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from pjm_forecast.config import ProjectConfig


def download_dataset_if_needed(config: ProjectConfig, raw_dir: Path) -> Path:
    dataset_cfg = config.dataset
    local_csv_path = dataset_cfg.get("local_csv_path")
    if local_csv_path:
        resolved_local = config.resolve_path(local_csv_path) if not Path(local_csv_path).is_absolute() else Path(local_csv_path)
        if not resolved_local.exists():
            raise FileNotFoundError(f"Configured dataset.local_csv_path does not exist: {resolved_local}")
        return resolved_local
    target_path = raw_dir / dataset_cfg["source_filename"]
    if target_path.exists():
        return target_path

    with urlopen(dataset_cfg["source_url"]) as response:
        target_path.write_bytes(response.read())
    return target_path
