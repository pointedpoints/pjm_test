from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pjm_forecast.config import load_config
from pjm_forecast.prepared_data import FeatureSchema, PreparedDataset


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


def test_prepared_dataset_from_source_builds_canonical_panel_and_feature_contract(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)
    assert list(prepared.panel_df.columns) == prepared.schema.panel_columns()
    assert list(prepared.feature_df.columns) == prepared.schema.feature_columns()
    assert prepared.panel_df["unique_id"].nunique() == 1
    assert prepared.panel_df["ds"].dt.tz is None
    assert prepared.panel_df["ds"].is_monotonic_increasing
    assert prepared.panel_df["ds"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_feature_schema_rejects_panel_gaps(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)
    broken_panel = prepared.panel_df.drop(index=prepared.panel_df.index[24]).reset_index(drop=True)
    with pytest.raises(ValueError, match="contiguous hourly data"):
        FeatureSchema(config).validate_panel_frame(broken_panel)


def test_prepared_dataset_split_days_match_requested_spans(tmp_path: Path) -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)
    validation_days = prepared.split_days("validation")
    test_days = prepared.split_days("test")
    assert len(validation_days) == config.backtest["validation_days"]
    assert len(test_days) == config.backtest["years_test"] * 364
    assert prepared.split_boundaries["validation_start"] > prepared.split_boundaries["train_end"]
    assert prepared.split_boundaries["test_start"] > prepared.split_boundaries["validation_end"]


def test_feature_schema_exposes_model_column_groups() -> None:
    config = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    schema = FeatureSchema(config)
    futr_columns = schema.nbeatsx_futr_exog_columns()
    hist_columns = schema.nbeatsx_hist_exog_columns()
    assert "system_load_forecast" in futr_columns
    assert "zonal_load_forecast" in futr_columns
    assert "is_weekend" in futr_columns
    assert "is_holiday" in futr_columns
    assert "hour_sin" in futr_columns
    assert "day_of_year_cos" in futr_columns
    assert "price_lag_168" not in futr_columns
    assert "system_load_forecast_lag_168" not in futr_columns
    assert "zonal_load_forecast_lag_168" not in futr_columns
    assert "price_lag_168" in hist_columns
    assert "system_load_forecast_lag_168" in hist_columns
    assert "zonal_load_forecast_lag_168" in hist_columns
