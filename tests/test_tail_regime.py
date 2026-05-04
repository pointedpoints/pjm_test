import numpy as np
import pandas as pd

from pjm_forecast.evaluation.tail_regime import compute_daily_peak_tail_gap, compute_tail_regime_diagnostics


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h")
    y_values = [10.0, 20.0, 30.0, 40.0, 100.0, 120.0, 200.0, 300.0]
    rows = []
    for ds, y in zip(timestamps, y_values, strict=True):
        rows.append({"ds": ds, "y": y, "y_pred": y - 1.0, "quantile": 0.50, "model": "m", "split": "test", "seed": 7})
        rows.append(
            {"ds": ds, "y": y, "y_pred": y + 10.0, "quantile": 0.99, "model": "m", "split": "test", "seed": 7}
        )
        rows.append(
            {"ds": ds, "y": y, "y_pred": y + 20.0, "quantile": 0.995, "model": "m", "split": "test", "seed": 7}
        )
    return pd.DataFrame(rows)


def test_tail_regime_reports_actual_price_segments() -> None:
    result = compute_tail_regime_diagnostics(_frame())
    assert set(result["regime"]) == {
        "all",
        "actual_le_p50",
        "actual_p50_p80",
        "actual_p80_p90",
        "actual_p90_p95",
        "actual_p95_p99",
        "actual_gt_p99",
    }
    all_row = result.loc[result["regime"].eq("all")].iloc[0]
    assert all_row["n_hours"] == 8
    assert np.isclose(all_row["q99_upper_coverage"], 1.0)


def test_daily_peak_tail_gap_reports_peak_hour_gap() -> None:
    result = compute_daily_peak_tail_gap(_frame())
    assert len(result) == 1
    row = result.iloc[0]
    assert row["actual_max"] == 300.0
    assert row["actual_peak_hour"] == 7
    assert row["peak_q99_gap"] == -10.0
