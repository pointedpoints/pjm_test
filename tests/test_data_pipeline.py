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


def _build_panel(start: str, hours: int = 24 * 7) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    rows = []
    for offset in range(hours):
        rows.append(
            {
                "unique_id": "PJM_COMED",
                "ds": start_ts + pd.Timedelta(hours=offset),
                "y": float(offset),
                "system_load_forecast": float(10_000 + offset),
                "zonal_load_forecast": float(2_000 + offset),
                "weather_temp_mean": float(-5.0 + 0.1 * (offset % 24)),
                "weather_temp_spread": 1.5,
                "weather_apparent_temp_mean": float(-7.0 + 0.1 * (offset % 24)),
                "weather_wind_speed_mean": 5.0,
                "weather_cloud_cover_mean": 25.0,
                "weather_precip_area_fraction": 0.0,
            }
        )
    return pd.DataFrame(rows)


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
    assert "price_lag_24" in contract.hist_exog_columns
    assert "price_lag_168" not in contract.hist_exog_columns
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


def test_feature_schema_builds_hour_indicator_and_sum_features(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["features"]["future_exog"].extend(
                [
                    "hour_7_pulse",
                    "hour_19_pulse",
                    "pressure_sum",
                ]
            ),
            payload["features"].__setitem__(
                "derived_features",
                [
                    {"kind": "hour_indicator", "hour": 7, "name": "hour_7_pulse"},
                    {"kind": "hour_indicator", "hour": 19, "name": "hour_19_pulse"},
                    {
                        "kind": "sum",
                        "inputs": ["hour_7_pulse", "hour_19_pulse"],
                        "name": "pressure_sum",
                    },
                ],
            ),
        ),
    )
    config = load_config(config_path)
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)

    assert prepared.feature_df.loc[7, "hour_7_pulse"] == 1.0
    assert prepared.feature_df.loc[19, "hour_19_pulse"] == 1.0
    assert prepared.feature_df.loc[7, "pressure_sum"] == 1.0
    assert prepared.feature_df.loc[19, "pressure_sum"] == 1.0
    assert prepared.feature_df.loc[8, "pressure_sum"] == 0.0


def test_feature_schema_builds_prior_day_price_stat_features(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["features"]["future_exog"].extend(
                [
                    "prior_day_price_spread",
                    "prior_day_price_max_ramp",
                    "prior_day_price_max",
                ]
            ),
            payload["features"].__setitem__(
                "derived_features",
                [
                    {"kind": "prior_day_price_stat", "source": "y", "stat": "spread", "name": "prior_day_price_spread"},
                    {"kind": "prior_day_price_stat", "source": "y", "stat": "max_ramp", "name": "prior_day_price_max_ramp"},
                    {"kind": "prior_day_price_stat", "source": "y", "stat": "max", "name": "prior_day_price_max"},
                ],
            ),
        ),
    )
    config = load_config(config_path)
    csv_path = _write_csv(tmp_path)
    prepared = PreparedDataset.from_source(config, csv_path)

    first_day = prepared.feature_df.iloc[:24]
    second_day = prepared.feature_df.iloc[24:48]
    third_day = prepared.feature_df.iloc[48:72]

    assert first_day["prior_day_price_spread"].eq(0.0).all()
    assert second_day["prior_day_price_spread"].eq(23.0).all()
    assert second_day["prior_day_price_max_ramp"].eq(1.0).all()
    assert second_day["prior_day_price_max"].eq(23.0).all()
    assert third_day["prior_day_price_max"].eq(47.0).all()
    assert "prior_day_price_spread" in prepared.schema.nbeatsx_futr_exog_columns()


def test_feature_schema_builds_pre_holiday_features() -> None:
    config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    raw = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    if "days_to_next_holiday" not in raw["features"]["future_exog"]:
        raw["features"]["future_exog"].append("days_to_next_holiday")
    if "is_pre_holiday_window_3" not in raw["features"]["future_exog"]:
        raw["features"]["future_exog"].append("is_pre_holiday_window_3")
    if not any(item.get("name") == "days_to_next_holiday" for item in raw["features"].get("derived_features", [])):
        raw["features"].setdefault("derived_features", []).append(
            {"kind": "days_to_next_holiday", "name": "days_to_next_holiday"}
        )
    if not any(item.get("name") == "is_pre_holiday_window_3" for item in raw["features"].get("derived_features", [])):
        raw["features"].setdefault("derived_features", []).append(
            {"kind": "pre_holiday_window", "max_days": 3, "name": "is_pre_holiday_window_3"}
        )
    temp_config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    temp_config = type(temp_config)(raw=raw, path=config.path)
    temp_config.validate_runtime_contracts()

    schema = FeatureSchema(temp_config)
    panel_df = _build_panel("2025-12-22 00:00:00", hours=24 * 5)
    feature_df = schema.build_feature_frame(panel_df)

    christmas_eve = feature_df[feature_df["ds"] == pd.Timestamp("2025-12-24 12:00:00")].iloc[0]
    christmas = feature_df[feature_df["ds"] == pd.Timestamp("2025-12-25 12:00:00")].iloc[0]
    dec_22 = feature_df[feature_df["ds"] == pd.Timestamp("2025-12-22 12:00:00")].iloc[0]

    assert christmas_eve["days_to_next_holiday"] == 1.0
    assert christmas_eve["is_pre_holiday_window_3"] == 1.0
    assert christmas["days_to_next_holiday"] == 0.0
    assert christmas["is_pre_holiday_window_3"] == 0.0
    assert dec_22["days_to_next_holiday"] == 3.0
    assert dec_22["is_pre_holiday_window_3"] == 1.0


def test_feature_schema_builds_year_end_semantic_calendar_features() -> None:
    config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    raw = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    for feature_name in ["days_since_prev_holiday", "days_to_year_end", "is_year_end_window"]:
        if feature_name not in raw["features"]["future_exog"]:
            raw["features"]["future_exog"].append(feature_name)
    additions = [
        {"kind": "days_since_prev_holiday", "name": "days_since_prev_holiday"},
        {"kind": "days_to_year_end", "name": "days_to_year_end"},
        {"kind": "year_end_window", "name": "is_year_end_window"},
    ]
    for item in additions:
        if not any(existing.get("name") == item["name"] for existing in raw["features"].get("derived_features", [])):
            raw["features"].setdefault("derived_features", []).append(item)
    temp_config = type(config)(raw=raw, path=config.path)
    temp_config.validate_runtime_contracts()

    schema = FeatureSchema(temp_config)
    panel_df = _build_panel("2025-12-30 00:00:00", hours=24 * 5)
    feature_df = schema.build_feature_frame(panel_df)

    dec_30 = feature_df[feature_df["ds"] == pd.Timestamp("2025-12-30 12:00:00")].iloc[0]
    jan_02 = feature_df[feature_df["ds"] == pd.Timestamp("2026-01-02 12:00:00")].iloc[0]
    jan_03 = feature_df[feature_df["ds"] == pd.Timestamp("2026-01-03 12:00:00")].iloc[0]

    assert dec_30["days_since_prev_holiday"] == 5.0
    assert dec_30["days_to_year_end"] == 1.0
    assert dec_30["is_year_end_window"] == 1.0
    assert jan_02["days_since_prev_holiday"] == 1.0
    assert jan_02["days_to_year_end"] == 363.0
    assert jan_02["is_year_end_window"] == 1.0
    assert jan_03["is_year_end_window"] == 1.0
