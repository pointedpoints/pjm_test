from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema, PreparedDataset


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


def load_panel_dataset(config: ProjectConfig, csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    raw_df.columns = [column.strip() for column in raw_df.columns]
    return FeatureSchema(config).normalize_panel_frame(raw_df)


def build_split_boundaries(config: ProjectConfig, panel_df: pd.DataFrame) -> dict[str, str]:
    split_boundaries = PreparedDataset.build_split_boundaries(config, panel_df)
    return {key: value.isoformat() for key, value in split_boundaries.items()}


def save_split_boundaries(split_boundaries: dict[str, str], output_path: Path) -> None:
    output_path.write_text(json.dumps(split_boundaries, indent=2), encoding="utf-8")
