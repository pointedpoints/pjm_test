from __future__ import annotations

from dataclasses import dataclass
import json
from math import pi
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .config import ProjectConfig


PANEL_ID_COLUMNS = ["unique_id", "ds", "system_load_forecast", "zonal_load_forecast"]
PREDICTION_COLUMNS = ["ds", "y", "y_pred", "model", "split", "seed", "quantile", "metadata"]
EPF_ALIAS_MAP = {
    "y": "Price",
    "system_load_forecast": "Exogenous 1",
    "zonal_load_forecast": "Exogenous 2",
}
DEFAULT_RETRIEVAL_CALENDAR_BASES = ("day_of_week", "day_of_year", "month")


def prediction_metadata(forecast_day: pd.Timestamp) -> str:
    return json.dumps({"forecast_day": pd.Timestamp(forecast_day).isoformat()})


def forecast_day_from_prediction_frame(day_df: pd.DataFrame) -> pd.Timestamp:
    if "metadata" in day_df.columns and day_df["metadata"].notna().any():
        payload = day_df["metadata"].iloc[0]
        if isinstance(payload, str):
            value = json.loads(payload)["forecast_day"]
            return pd.Timestamp(value).normalize()
    return pd.Timestamp(day_df["ds"].iloc[0]).normalize()


def default_nbeatsx_protected_exog_columns() -> list[str]:
    return [
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]


@dataclass(frozen=True)
class FeatureSchema:
    config: ProjectConfig

    def raw_column_map(self) -> dict[str, str]:
        dataset_cfg = self.config.dataset
        return {
            dataset_cfg["timestamp_col"]: "ds",
            dataset_cfg["price_col"]: self.config.target_column,
            dataset_cfg["exogenous_columns"]["system_load_forecast"]: "system_load_forecast",
            dataset_cfg["exogenous_columns"]["zonal_load_forecast"]: "zonal_load_forecast",
        }

    def panel_columns(self) -> list[str]:
        return ["unique_id", "ds", self.config.target_column, *PANEL_ID_COLUMNS[2:]]

    def future_exog_columns(self) -> list[str]:
        return list(self.config.features["future_exog"])

    def cyclical_bases(self) -> list[str]:
        return list(self.config.features["cyclical"])

    def cyclical_columns(self) -> list[str]:
        columns: list[str] = []
        for column_name in self.cyclical_bases():
            columns.extend([f"{column_name}_sin", f"{column_name}_cos"])
        return columns

    def calendar_columns(self) -> list[str]:
        return ["is_weekend", "is_holiday", *self.cyclical_columns()]

    def price_lag_columns(self) -> list[str]:
        return [f"price_lag_{lag}" for lag in self.config.features["price_lags"]]

    def load_lag_columns(self) -> list[str]:
        columns: list[str] = []
        for lag in self.config.features["load_lags"]:
            columns.extend(
                [
                    f"system_load_forecast_lag_{lag}",
                    f"zonal_load_forecast_lag_{lag}",
                ]
            )
        return columns

    def feature_columns(self) -> list[str]:
        return [
            *self.panel_columns(),
            *self.calendar_columns(),
            *self.price_lag_columns(),
            *self.load_lag_columns(),
        ]

    def nbeatsx_futr_exog_columns(self) -> list[str]:
        return [*self.future_exog_columns(), *self.calendar_columns()]

    def nbeatsx_hist_exog_columns(self) -> list[str]:
        return [*self.price_lag_columns(), *self.load_lag_columns()]

    def nbeatsx_protected_exog_columns(self) -> list[str]:
        return [*default_nbeatsx_protected_exog_columns()]

    def retrieval_price_columns(self) -> list[str]:
        return [self.config.target_column]

    def retrieval_load_columns(self) -> list[str]:
        return self.future_exog_columns()

    def retrieval_calendar_columns(self) -> list[str]:
        columns = ["is_weekend", "is_holiday"]
        for column_name in self.cyclical_bases():
            if column_name in DEFAULT_RETRIEVAL_CALENDAR_BASES:
                columns.extend([f"{column_name}_sin", f"{column_name}_cos"])
        return columns

    def retrieval_feature_columns(self) -> list[str]:
        return [*self.retrieval_price_columns(), *self.retrieval_load_columns(), *self.retrieval_calendar_columns()]

    def epftoolbox_alias_map(self) -> dict[str, str]:
        return dict(EPF_ALIAS_MAP)

    def prediction_columns(self) -> list[str]:
        return list(PREDICTION_COLUMNS)

    def normalize_panel_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        renamed = raw_df.rename(columns=self.raw_column_map())
        missing = [
            column
            for column in ["ds", self.config.target_column, "system_load_forecast", "zonal_load_forecast"]
            if column not in renamed.columns
        ]
        if missing:
            raise ValueError(f"Dataset is missing required mapped columns: {missing}")

        panel_df = renamed.loc[:, ["ds", self.config.target_column, "system_load_forecast", "zonal_load_forecast"]].copy()
        panel_df["ds"] = pd.to_datetime(panel_df["ds"], utc=False)
        panel_df["unique_id"] = self.config.dataset["unique_id"]
        panel_df = panel_df.sort_values("ds").reset_index(drop=True)
        panel_df = panel_df.loc[:, self.panel_columns()]
        self.validate_panel_frame(panel_df)
        return panel_df

    def build_feature_frame(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        self.validate_panel_frame(panel_df)
        feature_df = panel_df.copy()
        feature_df["date"] = feature_df["ds"].dt.normalize()
        feature_df["hour"] = feature_df["ds"].dt.hour
        feature_df["day_of_week"] = feature_df["ds"].dt.dayofweek
        feature_df["day_of_year"] = feature_df["ds"].dt.dayofyear
        feature_df["month"] = feature_df["ds"].dt.month
        feature_df["is_weekend"] = feature_df["day_of_week"].isin([5, 6]).astype(int)

        country_holidays = holidays.country_holidays(self.config.features["holiday_country"])
        feature_df["is_holiday"] = feature_df["date"].isin(country_holidays).astype(int)

        for column_name, period in self.config.features["cyclical"].items():
            encoded = self._encode_cyclical(feature_df[column_name], period=period, prefix=column_name)
            feature_df = pd.concat([feature_df, encoded], axis=1)

        for lag in self.config.features["price_lags"]:
            feature_df[f"price_lag_{lag}"] = feature_df[self.config.target_column].shift(lag)
        for lag in self.config.features["load_lags"]:
            feature_df[f"system_load_forecast_lag_{lag}"] = feature_df["system_load_forecast"].shift(lag)
            feature_df[f"zonal_load_forecast_lag_{lag}"] = feature_df["zonal_load_forecast"].shift(lag)

        feature_df = feature_df.loc[:, self.feature_columns()].copy()
        self.validate_feature_frame(feature_df)
        return feature_df

    def validate_panel_frame(self, panel_df: pd.DataFrame) -> None:
        self._require_columns(panel_df, self.panel_columns(), "panel frame")
        self._validate_ds_series(panel_df["ds"], "panel frame")
        if panel_df[self.panel_columns()].isna().any().any():
            missing = panel_df[self.panel_columns()].isna().sum()
            raise ValueError(f"Panel frame contains missing values: {missing.to_dict()}")
        diffs = panel_df["ds"].diff().dropna()
        if not diffs.eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Panel frame must be contiguous hourly data with no duplicate or missing hours.")

    def validate_feature_frame(self, feature_df: pd.DataFrame) -> None:
        self._require_columns(feature_df, self.feature_columns(), "feature frame")
        self._validate_ds_series(feature_df["ds"], "feature frame")
        non_lag_columns = [*self.panel_columns(), *self.calendar_columns()]
        if feature_df[non_lag_columns].isna().any().any():
            missing = feature_df[non_lag_columns].isna().sum()
            raise ValueError(f"Feature frame contains missing non-lag values: {missing.to_dict()}")

    def validate_prediction_frame(self, prediction_df: pd.DataFrame, require_metadata: bool = True) -> None:
        self._require_columns(prediction_df, self.prediction_columns(), "prediction frame")
        self._validate_ds_series(prediction_df["ds"], "prediction frame")
        required_non_null = ["y", "y_pred", "model", "split", "seed"]
        if prediction_df[required_non_null].isna().any().any():
            missing = prediction_df[required_non_null].isna().sum()
            raise ValueError(f"Prediction frame contains missing required values: {missing.to_dict()}")
        if require_metadata and prediction_df["metadata"].isna().any():
            raise ValueError("Prediction frame metadata column must be populated for every row.")
        for column in ["model", "split", "seed"]:
            if prediction_df[column].nunique(dropna=False) > 1:
                raise ValueError(f"Prediction frame column '{column}' must be constant within a run.")

    def _require_columns(self, frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{frame_name} is missing required columns: {missing}")

    def _validate_ds_series(self, ds: pd.Series, frame_name: str) -> None:
        if not is_datetime64_any_dtype(ds):
            raise ValueError(f"{frame_name} requires a datetime-like 'ds' column.")
        if ds.dt.tz is not None:
            raise ValueError(f"{frame_name} must use timezone-naive local timestamps in 'ds'.")
        if not ds.is_monotonic_increasing:
            raise ValueError(f"{frame_name} must be sorted by 'ds'.")
        if ds.duplicated().any():
            raise ValueError(f"{frame_name} contains duplicate 'ds' timestamps.")

    def _encode_cyclical(self, series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
        angle = 2 * pi * series.astype(float) / float(period)
        return pd.DataFrame(
            {
                f"{prefix}_sin": np.sin(angle),
                f"{prefix}_cos": np.cos(angle),
            },
            index=series.index,
        )


@dataclass(frozen=True)
class PreparedDataset:
    config: ProjectConfig
    schema: FeatureSchema
    panel_df: pd.DataFrame
    feature_df: pd.DataFrame
    split_boundaries: dict[str, pd.Timestamp]

    @classmethod
    def from_source(cls, config: ProjectConfig, csv_path: Path) -> "PreparedDataset":
        raw_df = pd.read_csv(csv_path)
        raw_df.columns = [column.strip() for column in raw_df.columns]
        schema = FeatureSchema(config)
        panel_df = schema.normalize_panel_frame(raw_df)
        return cls.from_panel_frame(config, panel_df, schema=schema)

    @classmethod
    def from_panel_frame(
        cls,
        config: ProjectConfig,
        panel_df: pd.DataFrame,
        schema: FeatureSchema | None = None,
    ) -> "PreparedDataset":
        schema = schema or FeatureSchema(config)
        schema.validate_panel_frame(panel_df)
        feature_df = schema.build_feature_frame(panel_df)
        split_boundaries = cls.build_split_boundaries(config, panel_df)
        return cls(
            config=config,
            schema=schema,
            panel_df=panel_df.copy(),
            feature_df=feature_df,
            split_boundaries=split_boundaries,
        )

    @classmethod
    def from_artifacts(
        cls,
        config: ProjectConfig,
        *,
        panel_path: Path,
        feature_path: Path,
        split_boundaries_path: Path,
    ) -> "PreparedDataset":
        schema = FeatureSchema(config)
        panel_df = pd.read_parquet(panel_path)
        feature_df = pd.read_parquet(feature_path)
        split_boundaries = cls.load_split_boundaries(split_boundaries_path)
        schema.validate_panel_frame(panel_df)
        schema.validate_feature_frame(feature_df)
        return cls(
            config=config,
            schema=schema,
            panel_df=panel_df,
            feature_df=feature_df,
            split_boundaries=split_boundaries,
        )

    @staticmethod
    def build_split_boundaries(config: ProjectConfig, panel_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
        schema = FeatureSchema(config)
        schema.validate_panel_frame(panel_df)
        days = pd.Index(panel_df["ds"].dt.normalize().drop_duplicates().sort_values())
        years_test = config.backtest["years_test"]
        validation_days = config.backtest["validation_days"]
        test_days = years_test * 364

        if len(days) <= test_days + validation_days:
            raise ValueError("Not enough daily observations for requested validation/test split.")

        return {
            "train_end": days[-(test_days + validation_days + 1)],
            "validation_start": days[-(test_days + validation_days)],
            "validation_end": days[-(test_days + 1)],
            "test_start": days[-test_days],
            "test_end": days[-1],
        }

    @staticmethod
    def load_split_boundaries(path: Path) -> dict[str, pd.Timestamp]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: pd.Timestamp(value) for key, value in payload.items()}

    @staticmethod
    def save_split_boundaries(split_boundaries: dict[str, pd.Timestamp], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: pd.Timestamp(value).isoformat() for key, value in split_boundaries.items()}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save(self, *, panel_path: Path, feature_path: Path, split_boundaries_path: Path) -> None:
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        self.panel_df.to_parquet(panel_path, index=False)
        self.feature_df.to_parquet(feature_path, index=False)
        self.save_split_boundaries(self.split_boundaries, split_boundaries_path)

    def daily_index(self) -> pd.Index:
        return pd.Index(self.feature_df["ds"].dt.normalize().drop_duplicates().sort_values())

    def split_days(self, split_name: str) -> list[pd.Timestamp]:
        days = self.daily_index()
        if split_name == "validation":
            mask = (days >= self.split_boundaries["validation_start"]) & (days <= self.split_boundaries["validation_end"])
            return list(days[mask])
        if split_name == "test":
            mask = (days >= self.split_boundaries["test_start"]) & (days <= self.split_boundaries["test_end"])
            return list(days[mask])
        raise ValueError(f"Unsupported split name: {split_name}")

    def days_between(self, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
        days = self.daily_index()
        mask = (days >= start.normalize()) & (days <= end.normalize())
        return list(days[mask])

    def latest_history_window(self, window_days: int) -> pd.DataFrame:
        history_end = self.feature_df["ds"].max()
        window_start = history_end - pd.Timedelta(days=window_days) + pd.Timedelta(hours=1)
        history_df = self.feature_df.loc[(self.feature_df["ds"] >= window_start) & (self.feature_df["ds"] <= history_end)].copy()
        if history_df.empty:
            raise ValueError("No history rows found for export window.")
        self.schema.validate_feature_frame(history_df)
        return history_df
