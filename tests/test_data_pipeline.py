from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.config import load_config
from pjm_forecast.data.epftoolbox import build_split_boundaries, load_panel_dataset
from pjm_forecast.features.engineering import build_feature_frame, nbeatsx_futr_exog_columns


def _write_csv(tmp_path: Path, hours: int = 24 * 1200) -> Path:
    rows = []
    start = pd.Timestamp("2013-01-01 00:00:00")
    for offset in range(hours):
        rows.append(
            {
                "Date": start + pd.Timedelta(hours=offset),
                "Zonal COMED price": float(offset),
                "System load forecast": float(10_000 + offset),
                "Zonal COMED load foecast": float(2_000 + offset),
            }
        )
    csv_path = tmp_path / "PJM.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_panel_dataset_normalizes_columns(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    panel_df = load_panel_dataset(config, csv_path)
    assert list(panel_df.columns) == ["unique_id", "ds", "y", "system_load_forecast", "zonal_load_forecast"]
    assert panel_df["unique_id"].nunique() == 1


def test_feature_frame_contains_calendar_and_lags(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    panel_df = load_panel_dataset(config, csv_path)
    feature_df = build_feature_frame(config, panel_df)
    assert "hour_sin" in feature_df.columns
    assert "is_holiday" in feature_df.columns
    assert "price_lag_168" in feature_df.columns
    assert "system_load_forecast_lag_168" in feature_df.columns
    assert "zonal_load_forecast_lag_168" in feature_df.columns


def test_build_split_boundaries_returns_ordered_ranges(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    panel_df = load_panel_dataset(config, csv_path)
    split_boundaries = build_split_boundaries(config, panel_df)
    assert split_boundaries["validation_start"] > split_boundaries["train_end"]
    assert split_boundaries["test_start"] > split_boundaries["validation_end"]


def test_nbeatsx_futr_exog_columns_include_calendar_and_lags() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    columns = nbeatsx_futr_exog_columns(config)
    assert "system_load_forecast" in columns
    assert "zonal_load_forecast" in columns
    assert "is_weekend" in columns
    assert "is_holiday" in columns
    assert "hour_sin" in columns
    assert "day_of_year_cos" in columns
    assert "price_lag_168" in columns
    assert "system_load_forecast_lag_168" in columns
    assert "zonal_load_forecast_lag_168" in columns
