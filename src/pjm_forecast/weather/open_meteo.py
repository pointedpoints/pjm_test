from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from pjm_forecast.config import ProjectConfig


OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SUPPORTED_HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]


@dataclass(frozen=True)
class WeatherPoint:
    name: str
    latitude: float
    longitude: float
    weight: float


def build_open_meteo_weather_frame(
    config: ProjectConfig,
    ds_index: pd.Series | pd.Index,
    raw_dir: Path,
    *,
    request_fn: Callable[[str], bytes] | None = None,
) -> pd.DataFrame:
    weather_cfg = config.weather
    points = [
        WeatherPoint(
            name=str(point["name"]),
            latitude=float(point["latitude"]),
            longitude=float(point["longitude"]),
            weight=float(point["weight"]),
        )
        for point in weather_cfg.get("points", [])
    ]
    if not points:
        raise ValueError("No weather points configured.")

    ds_values = pd.Index(pd.to_datetime(ds_index, utc=False))
    start = pd.Timestamp(ds_values.min()).normalize()
    end = pd.Timestamp(ds_values.max()).normalize()
    expected_ds = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")

    cache_root = _cache_root(config, raw_dir)
    point_frames = [
        _load_or_fetch_point_frame(
            point=point,
            expected_ds=expected_ds,
            cache_root=cache_root,
            weather_cfg=weather_cfg,
            request_fn=request_fn,
        )
        for point in points
    ]
    return _aggregate_point_frames(point_frames, output_columns=config.weather_output_columns(), weather_cfg=weather_cfg)


def _cache_root(config: ProjectConfig, raw_dir: Path) -> Path:
    cache_dir = config.weather.get("cache_dir")
    if cache_dir:
        return config.resolve_path(str(cache_dir))
    return raw_dir / "weather"


def _load_or_fetch_point_frame(
    *,
    point: WeatherPoint,
    expected_ds: pd.DatetimeIndex,
    cache_root: Path,
    weather_cfg: dict[str, object],
    request_fn: Callable[[str], bytes] | None,
) -> pd.DataFrame:
    cache_path = cache_root / _cache_filename(point=point, expected_ds=expected_ds, weather_cfg=weather_cfg)
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        frame = _fetch_point_frame(point=point, expected_ds=expected_ds, weather_cfg=weather_cfg, request_fn=request_fn)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    _validate_point_frame(frame, expected_ds=expected_ds, point_name=point.name)
    return frame


def _cache_filename(*, point: WeatherPoint, expected_ds: pd.DatetimeIndex, weather_cfg: dict[str, object]) -> str:
    model = str(weather_cfg.get("model", "best_match"))
    start = pd.Timestamp(expected_ds.min()).strftime("%Y%m%d")
    end = pd.Timestamp(expected_ds.max()).strftime("%Y%m%d")
    return f"open_meteo_{model}_{point.name}_{start}_{end}.parquet"


def _fetch_point_frame(
    *,
    point: WeatherPoint,
    expected_ds: pd.DatetimeIndex,
    weather_cfg: dict[str, object],
    request_fn: Callable[[str], bytes] | None,
) -> pd.DataFrame:
    start_day = pd.Timestamp(expected_ds.min()).strftime("%Y-%m-%d")
    end_day = pd.Timestamp(expected_ds.max()).strftime("%Y-%m-%d")
    params = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "hourly": ",".join(SUPPORTED_HOURLY_VARIABLES),
        "models": str(weather_cfg.get("model", "best_match")),
        "timezone": str(weather_cfg.get("timezone", "auto")),
        "start_date": start_day,
        "end_date": end_day,
    }
    payload = _fetch_json(f"{OPEN_METEO_HISTORICAL_FORECAST_URL}?{urlencode(params)}", request_fn=request_fn)
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(f"Open-Meteo response for point {point.name!r} is missing 'hourly'.")

    frame = pd.DataFrame({"ds": pd.to_datetime(hourly["time"], utc=False)})
    for variable in SUPPORTED_HOURLY_VARIABLES:
        if variable not in hourly:
            raise ValueError(f"Open-Meteo response for point {point.name!r} is missing hourly variable {variable!r}.")
        frame[variable] = pd.Series(hourly[variable], dtype="float64")
    frame["point_name"] = point.name
    frame["weight"] = point.weight
    return frame


def _fetch_json(url: str, *, request_fn: Callable[[str], bytes] | None) -> dict[str, object]:
    if request_fn is not None:
        payload = request_fn(url)
    else:
        with urlopen(url) as response:
            payload = response.read()
    data = json.loads(payload)
    if data.get("error"):
        raise ValueError(f"Open-Meteo request failed: {data}")
    return data


def _validate_point_frame(frame: pd.DataFrame, *, expected_ds: pd.DatetimeIndex, point_name: str) -> None:
    if frame["ds"].duplicated().any():
        raise ValueError(f"Weather point {point_name!r} returned duplicate ds values.")
    actual_ds = pd.Index(pd.to_datetime(frame["ds"], utc=False))
    if len(actual_ds) != len(expected_ds) or not actual_ds.equals(pd.Index(expected_ds)):
        raise ValueError(f"Weather point {point_name!r} does not align to the expected hourly ds range.")
    if frame[SUPPORTED_HOURLY_VARIABLES].isna().any().any():
        missing = frame[SUPPORTED_HOURLY_VARIABLES].isna().sum()
        raise ValueError(f"Weather point {point_name!r} contains missing values: {missing[missing > 0].to_dict()}")


def _aggregate_point_frames(
    point_frames: list[pd.DataFrame],
    *,
    output_columns: list[str],
    weather_cfg: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    precip_threshold = float(weather_cfg.get("precipitation_area_fraction_threshold_mm", 0.1))

    for ds, ds_frame in pd.concat(point_frames, axis=0, ignore_index=True).groupby("ds", sort=True):
        weights = ds_frame["weight"].astype(float)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            raise ValueError("Weather point weights must sum to a positive value for every ds.")
        rows.append(
            {
                "ds": pd.Timestamp(ds),
                "weather_temp_mean": _weighted_mean(ds_frame["temperature_2m"], weights),
                "weather_temp_spread": float(ds_frame["temperature_2m"].max() - ds_frame["temperature_2m"].min()),
                "weather_apparent_temp_mean": _weighted_mean(ds_frame["apparent_temperature"], weights),
                "weather_wind_speed_mean": _weighted_mean(ds_frame["wind_speed_10m"], weights),
                "weather_cloud_cover_mean": _weighted_mean(ds_frame["cloud_cover"], weights),
                "weather_precip_area_fraction": float(
                    ((ds_frame["precipitation"].astype(float) >= precip_threshold).astype(float) * weights).sum() / total_weight
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    columns = ["ds", *output_columns]
    missing_columns = [column for column in columns if column not in result.columns]
    if missing_columns:
        raise ValueError(f"Weather aggregation is missing requested output columns: {missing_columns}")
    return result.loc[:, columns]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float((values.astype(float) * weights.astype(float)).sum() / weights.astype(float).sum())
