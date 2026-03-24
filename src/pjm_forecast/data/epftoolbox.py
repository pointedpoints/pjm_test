from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from pjm_forecast.config import ProjectConfig


def download_dataset_if_needed(config: ProjectConfig, raw_dir: Path) -> Path:
    dataset_cfg = config.dataset
    target_path = raw_dir / dataset_cfg["source_filename"]
    if target_path.exists():
        return target_path

    with urlopen(dataset_cfg["source_url"]) as response:
        target_path.write_bytes(response.read())
    return target_path


def load_panel_dataset(config: ProjectConfig, csv_path: Path) -> pd.DataFrame:
    dataset_cfg = config.dataset
    raw_df = pd.read_csv(csv_path)
    raw_df.columns = [column.strip() for column in raw_df.columns]

    renamed = raw_df.rename(
        columns={
            dataset_cfg["timestamp_col"]: "ds",
            dataset_cfg["price_col"]: "y",
            dataset_cfg["exogenous_columns"]["system_load_forecast"]: "system_load_forecast",
            dataset_cfg["exogenous_columns"]["zonal_load_forecast"]: "zonal_load_forecast",
        }
    )
    panel_df = renamed.loc[:, ["ds", "y", "system_load_forecast", "zonal_load_forecast"]].copy()
    panel_df["ds"] = pd.to_datetime(panel_df["ds"], utc=False)
    panel_df["unique_id"] = dataset_cfg["unique_id"]
    panel_df = panel_df.sort_values("ds").reset_index(drop=True)

    if panel_df.isna().any().any():
        missing = panel_df.isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing.to_dict()}")

    return panel_df.loc[:, ["unique_id", "ds", "y", "system_load_forecast", "zonal_load_forecast"]]


def build_split_boundaries(config: ProjectConfig, panel_df: pd.DataFrame) -> dict[str, str]:
    days = pd.Index(panel_df["ds"].dt.normalize().drop_duplicates().sort_values())
    years_test = config.backtest["years_test"]
    validation_days = config.backtest["validation_days"]
    test_days = years_test * 364

    if len(days) <= test_days + validation_days:
        raise ValueError("Not enough daily observations for requested validation/test split.")

    train_end = days[-(test_days + validation_days + 1)]
    validation_start = days[-(test_days + validation_days)]
    validation_end = days[-(test_days + 1)]
    test_start = days[-test_days]
    test_end = days[-1]

    return {
        "train_end": train_end.isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
    }


def save_split_boundaries(split_boundaries: dict[str, str], output_path: Path) -> None:
    output_path.write_text(json.dumps(split_boundaries, indent=2), encoding="utf-8")

