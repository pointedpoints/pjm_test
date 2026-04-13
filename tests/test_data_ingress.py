from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from pjm_forecast.config import load_config
from pjm_forecast.data.ingress import prepare_dataset


def _write_weather_ready_csv(path: Path) -> Path:
    rows = []
    start = pd.Timestamp("2024-01-01 00:00:00")
    for offset in range(24 * 40):
        ts = start + pd.Timedelta(hours=offset)
        rows.append(
            {
                "Date": ts,
                "Zonal COMED price": float(offset),
                "System load forecast": float(10_000 + offset),
                "Zonal COMED load foecast": float(2_000 + offset),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(tmp_path: Path, csv_path: Path) -> Path:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_current_processed.yaml").read_text(encoding="utf-8"))
    payload["project"]["root_override"] = str(tmp_path / "run")
    payload["dataset"]["local_csv_path"] = str(csv_path)
    payload["backtest"]["years_test"] = 0
    payload["backtest"]["validation_days"] = 14
    payload["backtest"]["rolling_window_days"] = 14
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_prepare_dataset_handles_official_weather_ready_source_with_weather_enrichment(tmp_path: Path) -> None:
    csv_path = _write_weather_ready_csv(tmp_path / "raw" / "weather_ready.csv")
    config = load_config(_write_config(tmp_path, csv_path))

    def _fake_weather(cfg, ds_index, raw_dir):
        del cfg, raw_dir
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(ds_index, utc=False),
                "weather_temp_mean": [1.0] * len(ds_index),
                "weather_temp_spread": [2.0] * len(ds_index),
                "weather_apparent_temp_mean": [0.5] * len(ds_index),
                "weather_wind_speed_mean": [3.0] * len(ds_index),
                "weather_cloud_cover_mean": [75.0] * len(ds_index),
                "weather_precip_area_fraction": [0.25] * len(ds_index),
            }
        )

    result = prepare_dataset(config, tmp_path / "raw", weather_builder=_fake_weather)

    assert result.weather_df is not None
    assert len(result.prepared.panel_df) == 24 * 40
    assert "weather_cloud_cover_mean" in result.prepared.panel_df.columns
    assert "weather_cloud_cover_mean_lag_24" in result.prepared.feature_df.columns
    assert result.prepared.panel_df["ds"].min() == pd.Timestamp("2024-01-01 00:00:00")


def test_prepare_dataset_rejects_weather_frames_with_missing_hours(tmp_path: Path) -> None:
    csv_path = _write_weather_ready_csv(tmp_path / "raw" / "weather_ready.csv")
    config = load_config(_write_config(tmp_path, csv_path))

    def _bad_weather(cfg, ds_index, raw_dir):
        del cfg, raw_dir
        ds_values = pd.Index(pd.to_datetime(ds_index, utc=False))
        return pd.DataFrame(
            {
                "ds": ds_values[:-1],
                "weather_temp_mean": [1.0] * (len(ds_values) - 1),
                "weather_temp_spread": [2.0] * (len(ds_values) - 1),
                "weather_apparent_temp_mean": [0.5] * (len(ds_values) - 1),
                "weather_wind_speed_mean": [3.0] * (len(ds_values) - 1),
                "weather_cloud_cover_mean": [75.0] * (len(ds_values) - 1),
                "weather_precip_area_fraction": [0.25] * (len(ds_values) - 1),
            }
        )

    try:
        prepare_dataset(config, tmp_path / "raw", weather_builder=_bad_weather)
    except ValueError as exc:
        assert "align exactly" in str(exc)
    else:
        raise AssertionError("Expected prepare_dataset to reject misaligned weather data.")
