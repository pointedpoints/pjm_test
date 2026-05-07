from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
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
    enriched_panel = _fill_load_forecast_nan_with_weather(enriched_panel, weather_df)
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


def _fill_load_forecast_nan_with_weather(
    panel: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fill remaining load forecast NaN using weather regression + ffill/bfill fallback.

    Strategy: for each day with NaN in the load forecast column, train a Ridge
    regression on same-day non-NaN hours → predict missing hours.  Falls back to
    similar historical days if same-day samples are insufficient.  Final fallback
    is cross-day ffill/bfill.
    """
    LOAD_COL = "zonal_load_forecast"
    WEATHER_COLS = [
        "weather_temp_mean", "weather_temp_spread",
        "weather_apparent_temp_mean", "weather_wind_speed_mean",
        "weather_cloud_cover_mean", "weather_precip_area_fraction",
    ]

    if LOAD_COL not in panel.columns:
        return panel
    if not panel[LOAD_COL].isna().any():
        return panel

    available_cols = [c for c in WEATHER_COLS if c in panel.columns]
    if len(available_cols) < 3:
        # Not enough weather features for regression — fall through to ffill/bfill
        panel[LOAD_COL] = panel[LOAD_COL].ffill().bfill()
        return panel

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    panel = panel.copy()
    panel["_date"] = pd.to_datetime(panel["ds"]).dt.normalize()
    nan_mask = panel[LOAD_COL].isna()

    for day in panel.loc[nan_mask, "_date"].drop_duplicates():
        day_mask = panel["_date"] == day
        missing_in_day = day_mask & nan_mask
        available_in_day = day_mask & ~nan_mask
        n_available = available_in_day.sum()

        if n_available < 6:
            # Not enough same-day samples — try historical days
            day_dt = pd.Timestamp(day)
            target_dow = day_dt.dayofweek
            target_month = day_dt.month
            historical = panel[
                (panel["_date"] < day_dt)
                & (panel["_date"].dt.dayofweek == target_dow)
                & (panel["_date"].dt.month == target_month)
                & ~panel[LOAD_COL].isna()
            ]
            if len(historical) < 12:
                # Not enough history either — leave NaN for ffill/bfill
                continue
            X_train = historical[available_cols].values
            y_train = historical[LOAD_COL].values
        else:
            X_train = panel.loc[available_in_day, available_cols].values
            y_train = panel.loc[available_in_day, LOAD_COL].values

        X_pred = panel.loc[missing_in_day, available_cols].values
        if X_pred.shape[0] == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_pred_s = scaler.transform(X_pred)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_pred_s)
        preds = np.clip(preds, 0, None)  # load cannot be negative

        panel.loc[missing_in_day, LOAD_COL] = preds

    # Final fallback: forward-fill only (no bfill — that's look-ahead leakage).
    # If NaN remains at the very beginning of the dataset (no prior value to ffill),
    # we raise — this means weather regression also failed, which needs investigation.
    panel[LOAD_COL] = panel[LOAD_COL].ffill()
    panel = panel.drop(columns=["_date"])

    if panel[LOAD_COL].isna().any():
        raise ValueError(
            "Load forecast still has NaN after weather regression + ffill. "
            "Check for NaN at the very beginning of the dataset."
        )
    return panel
