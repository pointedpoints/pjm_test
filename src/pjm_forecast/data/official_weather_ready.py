from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import PreparedDataset


DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True)
class WeatherReadyDataset:
    frame: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp


def build_comed_weather_ready_dataset(
    raw_dir: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> WeatherReadyDataset:
    lmp = _load_da_lmp(raw_dir)
    load_forecast = _load_strict_dayahead_load_forecast(raw_dir)

    start = pd.Timestamp(start or min(lmp["Date"].min(), load_forecast["Date"].min()))
    end = pd.Timestamp(end or max(lmp["Date"].max(), load_forecast["Date"].max()))
    expected = pd.date_range(start, end, freq="h")

    lmp = _normalize_lmp_to_epf_clock(lmp, expected)
    load_forecast = _normalize_load_forecast(load_forecast, expected)

    frame = lmp.merge(load_forecast, on="Date", how="inner")
    frame["System load forecast"] = frame["Zonal COMED load foecast"]
    ordered = ["Date", "Zonal COMED price", "System load forecast", "Zonal COMED load foecast"]
    return WeatherReadyDataset(frame=frame.loc[:, ordered], start=start, end=end)


def save_weather_ready_dataset(dataset: WeatherReadyDataset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(output_path, index=False)
    return output_path


def build_weather_ready_csv_if_needed(config: ProjectConfig, raw_dir: Path) -> Path:
    dataset_cfg = config.dataset
    local_csv_path = dataset_cfg.get("local_csv_path")
    if local_csv_path:
        output_path = config.resolve_path(str(local_csv_path)) if not Path(local_csv_path).is_absolute() else Path(local_csv_path)
    else:
        output_path = raw_dir / dataset_cfg.get("source_filename", "PJM_COMED_weather_ready.csv")

    if output_path.exists():
        return output_path

    dataset = build_comed_weather_ready_dataset(
        raw_dir,
        start=dataset_cfg.get("start_date"),
        end=dataset_cfg.get("end_date"),
    )
    return save_weather_ready_dataset(dataset, output_path)


def build_official_weather_ready_prepared_dataset(config: ProjectConfig, raw_dir: Path) -> PreparedDataset:
    csv_path = build_weather_ready_csv_if_needed(config, raw_dir)
    return PreparedDataset.from_source(config, csv_path)


def _load_da_lmp(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("rt_da_monthly_lmps*.csv")):
        frame = pd.read_csv(
            path,
            usecols=["datetime_beginning_ept", "pnode_name", "type", "total_lmp_da"],
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No rt_da_monthly_lmps*.csv files found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        (combined["pnode_name"].astype(str).str.upper() == "COMED")
        & (combined["type"].astype(str).str.upper() == "ZONE")
    ].copy()
    combined["Date"] = pd.to_datetime(combined["datetime_beginning_ept"], format=DATETIME_FORMAT)
    combined["Zonal COMED price"] = combined["total_lmp_da"].astype(float)
    grouped = (
        combined.groupby("Date", as_index=False)["Zonal COMED price"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return grouped


def _load_strict_dayahead_load_forecast(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("load_frcstd_hist*.csv")):
        frame = pd.read_csv(
            path,
            usecols=["evaluated_at_ept", "forecast_hour_beginning_ept", "forecast_area", "forecast_load_mw"],
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No load_frcstd_hist*.csv files found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["forecast_area"].astype(str).str.upper() == "COMED"].copy()
    combined["issue_ts"] = pd.to_datetime(combined["evaluated_at_ept"], format=DATETIME_FORMAT)
    combined["Date"] = pd.to_datetime(combined["forecast_hour_beginning_ept"], format=DATETIME_FORMAT)
    combined["forecast_day"] = combined["Date"].dt.normalize()
    strict = combined.loc[combined["issue_ts"] < combined["forecast_day"]].copy()
    latest = strict.sort_values(["Date", "issue_ts"]).drop_duplicates("Date", keep="last")
    latest["Zonal COMED load foecast"] = latest["forecast_load_mw"].astype(float)
    return latest.loc[:, ["Date", "Zonal COMED load foecast"]].sort_values("Date").reset_index(drop=True)


def _normalize_lmp_to_epf_clock(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> pd.DataFrame:
    normalized = frame.set_index("Date").reindex(expected).rename_axis("Date").reset_index()
    normalized["Zonal COMED price"] = normalized["Zonal COMED price"].interpolate(method="linear", limit_direction="both")
    if normalized["Zonal COMED price"].isna().any():
        raise ValueError("DA LMP normalization left missing values after interpolation.")
    return normalized


def _normalize_load_forecast(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> pd.DataFrame:
    normalized = frame.set_index("Date").reindex(expected).rename_axis("Date").reset_index()
    values = normalized["Zonal COMED load foecast"]
    for lag in (24, 48, 168):
        values = values.combine_first(values.shift(lag))
    values = values.interpolate(method="linear", limit_direction="both")
    values = values.ffill().bfill()
    normalized["Zonal COMED load foecast"] = values
    if normalized["Zonal COMED load foecast"].isna().any():
        raise ValueError("Load forecast normalization left missing values after simple fills.")
    return normalized
