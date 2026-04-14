from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

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


def _write_temp_config(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    mutate(payload)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


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


def test_current_processed_nbeatsx_exogenous_contract_uses_minimal_hist_signals() -> None:
    config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    schema = FeatureSchema(config)
    contract = schema.nbeatsx_exogenous_contract()

    assert "system_load_forecast" not in contract.signal_futr_exog_columns
    assert "system_load_forecast_lag_24" not in contract.hist_exog_columns
    assert "zonal_load_forecast" in contract.signal_futr_exog_columns
    assert "zonal_load_forecast_lag_24" in contract.hist_exog_columns
    assert "zonal_load_forecast_lag_168" not in contract.hist_exog_columns
    assert "weather_temp_spread" in contract.signal_futr_exog_columns
    assert "weather_apparent_temp_mean" not in contract.signal_futr_exog_columns
    assert "heating_degree_18" in contract.signal_futr_exog_columns
    assert "cooling_degree_22" in contract.signal_futr_exog_columns
    assert "load_cooling_pressure" in contract.signal_futr_exog_columns
    assert "weather_temp_spread_lag_24" not in contract.hist_exog_columns
    assert "weather_temp_spread_lag_168" not in contract.hist_exog_columns
    assert contract.future_only_signal_columns() == [
        "weather_temp_mean",
        "weather_temp_spread",
        "weather_wind_speed_mean",
        "weather_cloud_cover_mean",
        "weather_precip_area_fraction",
        "heating_degree_18",
        "cooling_degree_22",
        "load_cooling_pressure",
    ]


def test_feature_schema_builds_configured_derived_ramps(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["features"]["future_exog"].append("zonal_load_forecast_delta_24"),
            payload["features"].__setitem__("lag_sources", list(payload["features"]["future_exog"])),
            payload["features"]["lag_sources"].append("zonal_load_forecast_delta_24"),
            payload["features"].__setitem__(
                "derived_ramps",
                [{"source": "zonal_load_forecast", "lag": 24, "name": "zonal_load_forecast_delta_24"}],
            ),
        ),
    )
    config = load_config(config_path)
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)

    assert "zonal_load_forecast_delta_24" in prepared.feature_df.columns
    assert prepared.feature_df["zonal_load_forecast_delta_24"].iloc[0] == 0.0
    assert prepared.feature_df["zonal_load_forecast_delta_24"].iloc[24] == 24.0
    assert "zonal_load_forecast_delta_24_lag_24" in prepared.feature_df.columns


def test_feature_schema_builds_degree_day_and_interaction_features(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["features"]["future_exog"].extend(
                [
                    "heating_degree_18",
                    "cooling_degree_22",
                    "weekend_heating_degree_18",
                ]
            ),
            payload["features"].__setitem__(
                "derived_features",
                [
                    {
                        "kind": "degree_day",
                        "source": "system_load_forecast",
                        "mode": "heating",
                        "base": 10020.0,
                        "name": "heating_degree_18",
                    },
                    {
                        "kind": "degree_day",
                        "source": "system_load_forecast",
                        "mode": "cooling",
                        "base": 10020.0,
                        "name": "cooling_degree_22",
                    },
                    {
                        "kind": "multiply",
                        "left": "is_weekend",
                        "right": "heating_degree_18",
                        "name": "weekend_heating_degree_18",
                    },
                ],
            ),
        ),
    )
    config = load_config(config_path)
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)

    assert prepared.feature_df["heating_degree_18"].iloc[0] == 20.0
    assert prepared.feature_df["heating_degree_18"].iloc[21] == 0.0
    assert prepared.feature_df["cooling_degree_22"].iloc[0] == 0.0
    assert prepared.feature_df["cooling_degree_22"].iloc[30] == 10.0
    assert prepared.feature_df["weekend_heating_degree_18"].iloc[0] == 0.0


def test_feature_schema_allows_hidden_source_for_derived_feature(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["features"].__setitem__("future_exog", ["zonal_load_forecast", "heating_degree_hidden"]),
            payload["features"].__setitem__("lag_sources", ["zonal_load_forecast"]),
            payload["features"].__setitem__(
                "derived_features",
                [
                    {
                        "kind": "degree_day",
                        "source": "system_load_forecast",
                        "mode": "heating",
                        "base": 10020.0,
                        "name": "heating_degree_hidden",
                    }
                ],
            ),
        ),
    )
    config = load_config(config_path)
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)
    contract = prepared.schema.nbeatsx_exogenous_contract()

    assert "system_load_forecast" in prepared.panel_df.columns
    assert "system_load_forecast" not in contract.signal_futr_exog_columns
    assert "heating_degree_hidden" in contract.signal_futr_exog_columns
