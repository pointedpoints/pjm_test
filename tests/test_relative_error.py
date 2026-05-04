import numpy as np
import pandas as pd

from pjm_forecast.evaluation.relative_error import compute_relative_error_diagnostics


def _frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 01:00",
            "2026-02-01 00:00",
            "2026-02-01 01:00",
        ]
    )
    y_values = [20.0, 40.0, 80.0, 200.0]
    q50_values = [25.0, 30.0, 100.0, 140.0]
    rows = []
    for ds, y, q50 in zip(timestamps, y_values, q50_values, strict=True):
        rows.append({"ds": ds, "y": y, "y_pred": q50, "quantile": 0.50, "model": "m", "split": "test", "seed": 7})
        rows.append(
            {"ds": ds, "y": y, "y_pred": q50 + 10.0, "quantile": 0.90, "model": "m", "split": "test", "seed": 7}
        )
    return pd.DataFrame(rows)


def test_relative_error_reports_price_bins() -> None:
    result = compute_relative_error_diagnostics(_frame())
    price_bins = result.loc[result["slice_type"].eq("actual_price_bin")]

    row = price_bins.loc[price_bins["slice"].eq("20-30")].iloc[0]
    assert row["n_hours"] == 1
    assert row["q50_mae"] == 5.0
    assert np.isclose(row["wape"], 0.25)
    assert np.isclose(row["median_ape"], 0.25)


def test_relative_error_reports_month_and_hour_slices() -> None:
    result = compute_relative_error_diagnostics(_frame())
    january = result.loc[(result["slice_type"].eq("month")) & (result["slice"].eq("2026-01"))].iloc[0]
    hour_zero = result.loc[(result["slice_type"].eq("hour")) & (result["slice"].eq("0"))].iloc[0]

    assert january["n_hours"] == 2
    assert np.isclose(january["q50_mae"], 7.5)
    assert hour_zero["n_hours"] == 2
