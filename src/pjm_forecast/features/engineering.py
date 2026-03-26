from __future__ import annotations

from math import pi
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from pjm_forecast.config import ProjectConfig


def _encode_cyclical(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    angle = 2 * pi * series.astype(float) / float(period)
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(angle),
            f"{prefix}_cos": np.cos(angle),
        },
        index=series.index,
    )


def build_feature_frame(config: ProjectConfig, panel_df: pd.DataFrame) -> pd.DataFrame:
    feature_cfg = config.features
    feature_df = panel_df.copy()
    feature_df["date"] = feature_df["ds"].dt.normalize()
    feature_df["hour"] = feature_df["ds"].dt.hour
    feature_df["day_of_week"] = feature_df["ds"].dt.dayofweek
    feature_df["day_of_year"] = feature_df["ds"].dt.dayofyear
    feature_df["month"] = feature_df["ds"].dt.month
    feature_df["is_weekend"] = feature_df["day_of_week"].isin([5, 6]).astype(int)

    country_holidays = holidays.country_holidays(feature_cfg["holiday_country"])
    feature_df["is_holiday"] = feature_df["date"].isin(country_holidays).astype(int)

    cyclical_columns = []
    for column_name, period in feature_cfg["cyclical"].items():
        encoded = _encode_cyclical(feature_df[column_name], period=period, prefix=column_name)
        feature_df = pd.concat([feature_df, encoded], axis=1)
        cyclical_columns.extend(encoded.columns.tolist())

    for lag in feature_cfg["price_lags"]:
        feature_df[f"price_lag_{lag}"] = feature_df["y"].shift(lag)
    for lag in feature_cfg["load_lags"]:
        feature_df[f"system_load_forecast_lag_{lag}"] = feature_df["system_load_forecast"].shift(lag)
        feature_df[f"zonal_load_forecast_lag_{lag}"] = feature_df["zonal_load_forecast"].shift(lag)

    ordered_columns = [
        "unique_id",
        "ds",
        "y",
        "system_load_forecast",
        "zonal_load_forecast",
        "is_weekend",
        "is_holiday",
        *cyclical_columns,
        *(f"price_lag_{lag}" for lag in feature_cfg["price_lags"]),
        *(f"system_load_forecast_lag_{lag}" for lag in feature_cfg["load_lags"]),
        *(f"zonal_load_forecast_lag_{lag}" for lag in feature_cfg["load_lags"]),
    ]
    return feature_df.loc[:, ordered_columns].copy()


def nbeatsx_futr_exog_columns(config: ProjectConfig) -> list[str]:
    feature_cfg = config.features
    cyclical_columns = []
    for column_name in feature_cfg["cyclical"]:
        cyclical_columns.extend([f"{column_name}_sin", f"{column_name}_cos"])

    calendar_columns = ["is_weekend", "is_holiday", *cyclical_columns]
    return [*feature_cfg["future_exog"], *calendar_columns]


def nbeatsx_hist_exog_columns(config: ProjectConfig) -> list[str]:
    feature_cfg = config.features
    lag_columns = [f"price_lag_{lag}" for lag in feature_cfg["price_lags"]]
    load_lag_columns = []
    for lag in feature_cfg["load_lags"]:
        load_lag_columns.extend(
            [
                f"system_load_forecast_lag_{lag}",
                f"zonal_load_forecast_lag_{lag}",
            ]
        )
    return [*lag_columns, *load_lag_columns]


def save_feature_frame(feature_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
