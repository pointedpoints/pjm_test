from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.data.official_weather_ready import build_comed_weather_ready_dataset


def test_build_comed_weather_ready_dataset_normalizes_dst_and_missing_forecast(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    pd.DataFrame(
        [
            {"datetime_beginning_ept": "03/14/2021 12:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 10.0},
            {"datetime_beginning_ept": "03/14/2021 01:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 12.0},
            {"datetime_beginning_ept": "03/14/2021 03:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 16.0},
            {"datetime_beginning_ept": "03/14/2021 04:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 18.0},
            {"datetime_beginning_ept": "11/07/2021 01:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 20.0},
            {"datetime_beginning_ept": "11/07/2021 01:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 24.0},
            {"datetime_beginning_ept": "11/07/2021 02:00:00 AM", "pnode_name": "COMED", "type": "ZONE", "total_lmp_da": 26.0},
        ]
    ).to_csv(raw_dir / "rt_da_monthly_lmps.csv", index=False)

    pd.DataFrame(
        [
            {
                "evaluated_at_ept": "03/13/2021 05:45:00 AM",
                "forecast_hour_beginning_ept": "03/14/2021 12:00:00 AM",
                "forecast_area": "COMED",
                "forecast_load_mw": 100.0,
            },
            {
                "evaluated_at_ept": "03/13/2021 05:45:00 AM",
                "forecast_hour_beginning_ept": "03/14/2021 01:00:00 AM",
                "forecast_area": "COMED",
                "forecast_load_mw": 110.0,
            },
            {
                "evaluated_at_ept": "03/13/2021 05:45:00 AM",
                "forecast_hour_beginning_ept": "03/14/2021 03:00:00 AM",
                "forecast_area": "COMED",
                "forecast_load_mw": 130.0,
            },
            {
                "evaluated_at_ept": "03/13/2021 05:45:00 AM",
                "forecast_hour_beginning_ept": "03/14/2021 04:00:00 AM",
                "forecast_area": "COMED",
                "forecast_load_mw": 140.0,
            },
        ]
    ).to_csv(raw_dir / "load_frcstd_hist.csv", index=False)

    dataset = build_comed_weather_ready_dataset(
        raw_dir,
        start=pd.Timestamp("2021-03-14 00:00:00"),
        end=pd.Timestamp("2021-03-14 04:00:00"),
    )

    frame = dataset.frame
    assert len(frame) == 5
    assert list(frame["Date"]) == list(pd.date_range("2021-03-14 00:00:00", periods=5, freq="h"))
    assert frame["Zonal COMED price"].isna().sum() == 0
    assert frame["Zonal COMED load foecast"].isna().sum() == 0
    assert frame.loc[frame["Date"] == pd.Timestamp("2021-03-14 02:00:00"), "Zonal COMED price"].iloc[0] == 14.0
    assert frame.loc[frame["Date"] == pd.Timestamp("2021-03-14 02:00:00"), "Zonal COMED load foecast"].iloc[0] == 120.0
