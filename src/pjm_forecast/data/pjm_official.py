from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema, PreparedDataset


DEFAULT_API_BASE_URL = "https://api.pjm.com/api/v1"
MAX_API_RANGE_DAYS = 366
DEFAULT_HISTORY_YEARS = 5
DEFAULT_MIN_HISTORY_YEARS = 3

COMED_FUEL_COLUMNS = [
    "fuel_coal_share",
    "fuel_gas_share",
    "fuel_nuclear_share",
    "fuel_wind_share",
    "fuel_solar_share",
    "fuel_hydro_share",
    "fuel_other_share",
]


@dataclass(frozen=True)
class FeedSpec:
    name: str
    date_field: str
    filters: dict[str, str]


OFFICIAL_FEEDS: dict[str, FeedSpec] = {
    "da_hrl_lmps": FeedSpec("da_hrl_lmps", "datetime_beginning_ept", {"zone": "COMED", "type": "ZONE", "row_is_current": "TRUE"}),
    "load_frcstd_hist": FeedSpec("load_frcstd_hist", "forecast_hour_beginning_ept", {"forecast_area": "COMED"}),
    "hrl_load_estimated": FeedSpec("hrl_load_estimated", "datetime_beginning_ept", {"load_area": "COMED"}),
    "hrl_load_metered": FeedSpec("hrl_load_metered", "datetime_beginning_ept", {"zone": "CE", "is_verified": "TRUE"}),
    "ops_sum_frcst_peak_area": FeedSpec("ops_sum_frcst_peak_area", "projected_peak_datetime_ept", {"area": "COMED"}),
    "ops_sum_prev_period": FeedSpec("ops_sum_prev_period", "datetime_beginning_ept", {"area": "ComEd"}),
    "gen_by_fuel": FeedSpec("gen_by_fuel", "datetime_beginning_ept", {}),
    "hourly_wind_power_forecast": FeedSpec("hourly_wind_power_forecast", "datetime_beginning_ept", {}),
    "hourly_solar_power_forecast": FeedSpec("hourly_solar_power_forecast", "datetime_beginning_ept", {}),
    "frcstd_gen_outages": FeedSpec("frcstd_gen_outages", "forecast_date", {}),
}


def build_official_prepared_dataset(
    config: ProjectConfig,
    raw_dir: Path,
    *,
    request_fn: Callable[[str], bytes] | None = None,
) -> PreparedDataset:
    downloader = PjmOfficialDownloader(config=config, raw_dir=raw_dir, request_fn=request_fn)
    builder = PjmOfficialDatasetBuilder(config=config)
    return builder.build_from_feed_frames(downloader.download_required_feeds())


@dataclass
class PjmOfficialDownloader:
    config: ProjectConfig
    raw_dir: Path
    request_fn: Callable[[str], bytes] | None = None

    def download_required_feeds(self) -> dict[str, pd.DataFrame]:
        start_day, end_day = self._download_window()
        feed_frames: dict[str, pd.DataFrame] = {}
        for feed_name, spec in OFFICIAL_FEEDS.items():
            cache_path = self.raw_dir / f"{feed_name}_{start_day:%Y%m%d}_{end_day:%Y%m%d}.parquet"
            if cache_path.exists():
                feed_frames[feed_name] = pd.read_parquet(cache_path)
                continue
            frame = self._download_feed(spec, start_day=start_day, end_day=end_day)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
            feed_frames[feed_name] = frame
        return feed_frames

    def chunk_ranges(self, start_day: pd.Timestamp, end_day: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        current = start_day.normalize()
        final_day = end_day.normalize()
        while current <= final_day:
            chunk_end = min(current + pd.Timedelta(days=MAX_API_RANGE_DAYS - 1), final_day)
            ranges.append((current, chunk_end))
            current = chunk_end + pd.Timedelta(days=1)
        return ranges

    def _download_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        dataset_cfg = self.config.dataset
        end_day = pd.Timestamp(dataset_cfg.get("end_date") or pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).normalize()
        history_years = int(dataset_cfg.get("history_years", DEFAULT_HISTORY_YEARS))
        start_day = (end_day - pd.Timedelta(days=history_years * 365 - 1)).normalize()
        return start_day, end_day

    def _download_feed(self, spec: FeedSpec, *, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
        pages: list[pd.DataFrame] = []
        for chunk_start, chunk_end in self.chunk_ranges(start_day, end_day):
            start_row = 1
            while True:
                url = self._build_url(spec, chunk_start=chunk_start, chunk_end=chunk_end, start_row=start_row)
                payload = self._fetch(url)
                chunk_df = pd.read_csv(io.BytesIO(payload))
                if chunk_df.empty:
                    break
                pages.append(chunk_df)
                if len(chunk_df) < 50_000:
                    break
                start_row += len(chunk_df)
        if not pages:
            return pd.DataFrame()
        return pd.concat(pages, axis=0, ignore_index=True)

    def _build_url(self, spec: FeedSpec, *, chunk_start: pd.Timestamp, chunk_end: pd.Timestamp, start_row: int) -> str:
        params = {
            spec.date_field: f"{chunk_start:%m/%d/%Y} to {chunk_end:%m/%d/%Y}",
            "rowCount": 50000,
            "startRow": start_row,
            "download": "true",
            "format": "csv",
            "subscription-key": self._subscription_key(),
            **spec.filters,
        }
        return f"{self.config.dataset.get('api_base_url', DEFAULT_API_BASE_URL).rstrip('/')}/{spec.name}?{urlencode(params)}"

    def _subscription_key(self) -> str:
        env_name = str(self.config.dataset.get("subscription_key_env", "PJM_SUBSCRIPTION_KEY"))
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(
                f"Official PJM dataset download requires environment variable {env_name} to be set with a Data Miner subscription key."
            )
        return value

    def _fetch(self, url: str) -> bytes:
        if self.request_fn is not None:
            return self.request_fn(url)
        with urlopen(url) as response:
            return response.read()


@dataclass
class PjmOfficialDatasetBuilder:
    config: ProjectConfig

    def build_from_feed_frames(self, feed_frames: dict[str, pd.DataFrame]) -> PreparedDataset:
        panel_df = self._build_panel_frame(feed_frames)
        prepared = PreparedDataset.from_panel_frame(self.config, panel_df, schema=FeatureSchema(self.config))
        self._validate_history_span(prepared.panel_df)
        return prepared

    def _build_panel_frame(self, feed_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        target_df = self._normalize_lmp(feed_frames["da_hrl_lmps"])
        frames = [
            self._normalize_load_forecast(feed_frames["load_frcstd_hist"]),
            self._normalize_hourly_value(feed_frames["hrl_load_estimated"], "estimated_load_hourly", "comed_load_estimated"),
            self._normalize_hourly_value(feed_frames["hrl_load_metered"], "mw", "comed_load_metered"),
            self._normalize_peak_area(feed_frames["ops_sum_frcst_peak_area"]),
            self._normalize_prev_period(feed_frames["ops_sum_prev_period"]),
            self._normalize_wind_forecast(feed_frames["hourly_wind_power_forecast"]),
            self._normalize_solar_forecast(feed_frames["hourly_solar_power_forecast"]),
            self._normalize_gen_outages(feed_frames["frcstd_gen_outages"]),
            self._normalize_gen_by_fuel(feed_frames["gen_by_fuel"]),
        ]

        panel_df = target_df.copy()
        for frame in frames:
            panel_df = panel_df.merge(frame, on="ds", how="left")

        required_columns = [column for column in FeatureSchema(self.config).panel_columns() if column != "unique_id"]
        panel_df = panel_df.sort_values("ds").reset_index(drop=True)
        panel_df = panel_df.dropna(subset=required_columns).reset_index(drop=True)
        panel_df["unique_id"] = self.config.dataset["unique_id"]
        ordered = ["unique_id", *required_columns]
        return panel_df.loc[:, ordered]

    def _normalize_lmp(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if "row_is_current" in df.columns:
            df = df.loc[df["row_is_current"].astype(str).str.upper() == "TRUE"].copy()
        if "type" in df.columns:
            df = df.loc[df["type"].astype(str).str.upper() == "ZONE"].copy()
        if "pnode_name" in df.columns and (df["pnode_name"].astype(str).str.upper() == "COMED").any():
            df = df.loc[df["pnode_name"].astype(str).str.upper() == "COMED"].copy()
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        df["y"] = df["total_lmp_da"].astype(float)
        if df["ds"].duplicated().any():
            raise ValueError("Official da_hrl_lmps normalization produced duplicate ds rows for COMED.")
        return df.loc[:, ["ds", "y"]].sort_values("ds").reset_index(drop=True)

    def _normalize_load_forecast(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["issue_ts"] = pd.to_datetime(df["evaluated_at_ept"], utc=False)
        df["ds"] = pd.to_datetime(df["forecast_hour_beginning_ept"], utc=False)
        df["forecast_day"] = df["ds"].dt.normalize()
        df = df.loc[df["issue_ts"] < df["forecast_day"]].copy()
        df = self._latest_issue_by_target(df, issue_column="issue_ts", target_column="ds")
        df["comed_load_forecast"] = df["forecast_load_mw"].astype(float)
        return df.loc[:, ["ds", "comed_load_forecast"]].sort_values("ds").reset_index(drop=True)

    def _normalize_hourly_value(self, frame: pd.DataFrame, value_column: str, output_column: str) -> pd.DataFrame:
        df = frame.copy()
        if "generated_at_ept" in df.columns:
            df["generated_at_ept"] = pd.to_datetime(df["generated_at_ept"], utc=False)
            df = df.sort_values(["datetime_beginning_ept", "generated_at_ept"]).drop_duplicates("datetime_beginning_ept", keep="last")
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        df[output_column] = df[value_column].astype(float)
        return df.loc[:, ["ds", output_column]].sort_values("ds").reset_index(drop=True)

    def _normalize_peak_area(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["issue_ts"] = pd.to_datetime(df["generated_at_ept"], utc=False)
        df["forecast_day"] = pd.to_datetime(df["projected_peak_datetime_ept"], utc=False).dt.normalize()
        df = df.loc[df["issue_ts"] < df["forecast_day"]].copy()
        df = self._latest_issue_by_target(df, issue_column="issue_ts", target_column="forecast_day")
        df = self._expand_daily_frame(
            df.loc[
                :,
                [
                    "forecast_day",
                    "pjm_load_forecast",
                    "internal_scheduled_capacity",
                    "unscheduled_steam_capacity",
                ],
            ].rename(
                columns={
                    "forecast_day": "day",
                    "pjm_load_forecast": "comed_peak_load_forecast",
                    "internal_scheduled_capacity": "comed_internal_scheduled_capacity",
                    "unscheduled_steam_capacity": "comed_unscheduled_steam_capacity",
                }
            )
        )
        return df

    def _normalize_prev_period(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if "generated_at_ept" in df.columns:
            df["generated_at_ept"] = pd.to_datetime(df["generated_at_ept"], utc=False)
            df = df.sort_values(["datetime_beginning_ept", "generated_at_ept"]).drop_duplicates("datetime_beginning_ept", keep="last")
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        return df.loc[
            :,
            ["ds", "area_load_forecast", "actual_load", "dispatch_rate"],
        ].rename(
            columns={
                "area_load_forecast": "comed_prev_period_area_load_forecast",
                "actual_load": "comed_prev_period_actual_load",
                "dispatch_rate": "comed_prev_period_dispatch_rate",
            }
        ).sort_values("ds").reset_index(drop=True)

    def _normalize_wind_forecast(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["issue_ts"] = pd.to_datetime(df["evaluated_at_ept"], utc=False)
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        df["forecast_day"] = df["ds"].dt.normalize()
        df = df.loc[df["issue_ts"] < df["forecast_day"]].copy()
        df = self._latest_issue_by_target(df, issue_column="issue_ts", target_column="ds")
        df["wind_forecast_mwh"] = df["wind_forecast_mwh"].astype(float)
        return df.loc[:, ["ds", "wind_forecast_mwh"]].sort_values("ds").reset_index(drop=True)

    def _normalize_solar_forecast(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["issue_ts"] = pd.to_datetime(df["evaluated_at_ept"], utc=False)
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        df["forecast_day"] = df["ds"].dt.normalize()
        df = df.loc[df["issue_ts"] < df["forecast_day"]].copy()
        df = self._latest_issue_by_target(df, issue_column="issue_ts", target_column="ds")
        df["solar_forecast_mwh"] = df["solar_forecast_mwh"].astype(float)
        return df.loc[:, ["ds", "solar_forecast_mwh"]].sort_values("ds").reset_index(drop=True)

    def _normalize_gen_outages(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["issue_ts"] = pd.to_datetime(df["forecast_execution_date_ept"], utc=False)
        df["forecast_day"] = pd.to_datetime(df["forecast_date"], utc=False).dt.normalize()
        df = df.loc[df["issue_ts"] < df["forecast_day"]].copy()
        df = self._latest_issue_by_target(df, issue_column="issue_ts", target_column="forecast_day")
        daily = df.loc[
            :,
            [
                "forecast_day",
                "forecast_gen_outage_mw_rto",
                "forecast_gen_outage_mw_west",
                "forecast_gen_outage_mw_other",
            ],
        ].rename(columns={"forecast_day": "day"})
        return self._expand_daily_frame(daily)

    def _normalize_gen_by_fuel(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["ds"] = pd.to_datetime(df["datetime_beginning_ept"], utc=False)
        df["fuel_group"] = df["fuel_type"].astype(str).map(_fuel_group).fillna("other")
        df["share"] = df["fuel_percentage_of_total"].astype(float)
        grouped = (
            df.groupby(["ds", "fuel_group"], as_index=False)["share"]
            .sum()
            .pivot(index="ds", columns="fuel_group", values="share")
            .reset_index()
            .fillna(0.0)
        )
        grouped.columns.name = None
        renamed = grouped.rename(columns={column: f"fuel_{column}_share" for column in grouped.columns if column != "ds"})
        for column in COMED_FUEL_COLUMNS:
            if column not in renamed.columns:
                renamed[column] = 0.0
        return renamed.loc[:, ["ds", *COMED_FUEL_COLUMNS]].sort_values("ds").reset_index(drop=True)

    def _expand_daily_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in frame.to_dict(orient="records"):
            day = pd.Timestamp(record.pop("day")).normalize()
            for hour in range(24):
                rows.append({"ds": day + pd.Timedelta(hours=hour), **record})
        return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    def _latest_issue_by_target(self, frame: pd.DataFrame, *, issue_column: str, target_column: str) -> pd.DataFrame:
        return frame.sort_values([target_column, issue_column]).drop_duplicates(target_column, keep="last").reset_index(drop=True)

    def _validate_history_span(self, panel_df: pd.DataFrame) -> None:
        min_years = int(self.config.dataset.get("min_history_years", DEFAULT_MIN_HISTORY_YEARS))
        daily_index = pd.Index(panel_df["ds"].dt.normalize().drop_duplicates().sort_values())
        if len(daily_index) < min_years * 364:
            raise ValueError(
                f"Official PJM dataset only produced {len(daily_index)} daily observations; expected at least {min_years * 364}."
            )


def _fuel_group(value: str) -> str:
    normalized = value.strip().lower()
    if "coal" in normalized:
        return "coal"
    if "gas" in normalized:
        return "gas"
    if "nuclear" in normalized:
        return "nuclear"
    if "wind" in normalized:
        return "wind"
    if "solar" in normalized:
        return "solar"
    if "hydro" in normalized or "water" in normalized:
        return "hydro"
    return "other"
