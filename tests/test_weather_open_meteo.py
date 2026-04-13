from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import yaml

from pjm_forecast.config import load_config
from pjm_forecast.weather import build_open_meteo_weather_frame


def _write_weather_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_current_processed.yaml").read_text(encoding="utf-8"))
    payload["project"]["root_override"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _request_fn(url: str) -> bytes:
    query = parse_qs(urlparse(url).query)
    lat = float(query["latitude"][0])
    start_day = pd.Timestamp(query["start_date"][0])
    end_day = pd.Timestamp(query["end_date"][0])
    ds = pd.date_range(start_day, end_day + pd.Timedelta(hours=23), freq="h")
    base = lat - 40.0
    payload = {
        "hourly": {
            "time": [value.strftime("%Y-%m-%dT%H:%M") for value in ds],
            "temperature_2m": [base + hour.hour for hour in ds],
            "apparent_temperature": [base + hour.hour + 1.5 for hour in ds],
            "wind_speed_10m": [10.0 + base] * len(ds),
            "precipitation": [0.0 if hour.hour < 12 else 0.2 for hour in ds],
            "cloud_cover": [50.0 + base] * len(ds),
        }
    }
    return json.dumps(payload).encode("utf-8")


def test_build_open_meteo_weather_frame_aggregates_cloud_and_precipitation(tmp_path: Path) -> None:
    config = load_config(_write_weather_config(tmp_path))
    ds_index = pd.date_range("2024-01-01 00:00:00", periods=48, freq="h")

    weather_df = build_open_meteo_weather_frame(
        config,
        ds_index,
        tmp_path / "raw",
        request_fn=_request_fn,
    )

    assert list(weather_df.columns) == ["ds", *config.weather_output_columns()]
    assert len(weather_df) == 48
    assert weather_df["weather_cloud_cover_mean"].between(50.0, 53.0).all()
    assert weather_df.loc[weather_df["ds"].dt.hour < 12, "weather_precip_area_fraction"].eq(0.0).all()
    assert weather_df.loc[weather_df["ds"].dt.hour >= 12, "weather_precip_area_fraction"].eq(1.0).all()
    assert weather_df["weather_temp_spread"].gt(0.0).all()
