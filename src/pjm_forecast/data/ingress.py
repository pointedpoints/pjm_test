from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema, PreparedDataset
from pjm_forecast.weather import build_open_meteo_weather_frame

from .epftoolbox import download_dataset_if_needed
from .official_weather_ready import build_official_weather_ready_prepared_dataset
from .pjm_official import build_official_prepared_dataset


@dataclass(frozen=True)
class PreparedDataResult:
    prepared: PreparedDataset
    weather_df: pd.DataFrame | None = None


def prepare_dataset(
    config: ProjectConfig,
    raw_dir: Path,
    *,
    weather_builder: Callable[[ProjectConfig, pd.Series | pd.Index, Path], pd.DataFrame] = build_open_meteo_weather_frame,
) -> PreparedDataResult:
    source_config = config.without_weather_feature_contracts() if config.weather_enabled else config
    prepared = _prepare_base_dataset(source_config, raw_dir)

    if not config.weather_enabled:
        return PreparedDataResult(prepared=prepared, weather_df=None)

    weather_df = _normalize_weather_frame(
        weather_builder(config, prepared.panel_df["ds"], raw_dir),
        expected_ds=prepared.panel_df["ds"],
        output_columns=config.weather_output_columns(),
    )
    enriched_panel = prepared.panel_df.merge(weather_df, on="ds", how="left")
    enriched = PreparedDataset.from_panel_frame(config, enriched_panel, schema=FeatureSchema(config))
    return PreparedDataResult(prepared=enriched, weather_df=weather_df)


def _prepare_base_dataset(config: ProjectConfig, raw_dir: Path) -> PreparedDataset:
    if config.dataset_source_type == "epftoolbox":
        csv_path = download_dataset_if_needed(config, raw_dir)
        return PreparedDataset.from_source(config, csv_path)
    if config.dataset_source_type == "pjm_official":
        return build_official_prepared_dataset(config, raw_dir)
    if config.dataset_source_type == "official_weather_ready":
        return build_official_weather_ready_prepared_dataset(config, raw_dir)
    raise ValueError(f"Unsupported dataset.source_type={config.dataset_source_type!r}.")


def _normalize_weather_frame(
    weather_df: pd.DataFrame,
    *,
    expected_ds: pd.Series | pd.Index,
    output_columns: list[str],
) -> pd.DataFrame:
    required_columns = ["ds", *output_columns]
    missing_columns = [column for column in required_columns if column not in weather_df.columns]
    if missing_columns:
        raise ValueError(f"Weather frame is missing required columns: {missing_columns}")

    normalized = weather_df.loc[:, required_columns].copy()
    normalized["ds"] = pd.to_datetime(normalized["ds"], utc=False)
    if normalized["ds"].duplicated().any():
        raise ValueError("Weather frame contains duplicate ds timestamps.")
    if normalized[required_columns].isna().any().any():
        missing = normalized[required_columns].isna().sum()
        raise ValueError(f"Weather frame contains missing values: {missing[missing > 0].to_dict()}")

    expected_index = pd.Index(pd.to_datetime(expected_ds, utc=False))
    actual_index = pd.Index(normalized["ds"])
    if len(actual_index) != len(expected_index) or not actual_index.equals(expected_index):
        raise ValueError("Weather frame ds timestamps must align exactly to the base hourly panel.")
    return normalized
