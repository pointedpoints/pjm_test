from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.evaluation import compute_normal_day_diagnostics


def _quantile_prediction_frame(*, include_spike_score: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = [
        ("2026-01-01 00:00:00", 20.0, 18.0, 200.0, 0.20),
        ("2026-01-01 01:00:00", 20.0, 22.0, 220.0, 0.20),
        ("2026-01-02 00:00:00", 22.0, 24.0, 30.0, 0.40),
        ("2026-01-02 01:00:00", 24.0, 20.0, 32.0, 0.40),
        ("2026-01-03 00:00:00", 100.0, 80.0, 130.0, 0.90),
        ("2026-01-03 01:00:00", 120.0, 90.0, 150.0, 0.90),
    ]
    for ds, y, q50, q99, spike_score in values:
        for quantile, y_pred in [(0.50, q50), (0.99, q99)]:
            row = {
                "ds": pd.Timestamp(ds),
                "y": y,
                "y_pred": y_pred,
                "model": "nbeatsx",
                "split": "test",
                "seed": 7,
                "quantile": quantile,
                "metadata": "{}",
            }
            if include_spike_score:
                row["spike_score"] = spike_score
            rows.append(row)
    return pd.DataFrame(rows)


def test_normal_day_diagnostics_summarize_q50_relative_error_segments() -> None:
    diagnostics = compute_normal_day_diagnostics(
        _quantile_prediction_frame(),
        actual_daily_max_quantile=0.95,
        low_risk_threshold=0.50,
    )

    expected_columns = [
        "segment",
        "n_days",
        "n_hours",
        "actual_daily_max_threshold",
        "low_risk_score_column",
        "low_risk_score_threshold",
        "actual_mean",
        "actual_median",
        "q50_mae",
        "q50_bias_mean",
        "q50_wape",
        "smape",
        "median_ape",
        "p75_ape",
        "p90_ape",
    ]
    assert diagnostics.columns.tolist() == expected_columns

    rows = diagnostics.set_index("segment")
    assert rows.index.tolist() == [
        "all",
        "actual_normal_day",
        "actual_spike_day",
        "forecast_low_risk_day",
        "forecast_high_risk_day",
    ]

    threshold = float(pd.Series([20.0, 24.0, 120.0]).quantile(0.95))
    assert np.isclose(rows.loc["all", "actual_daily_max_threshold"], threshold)
    assert rows.loc["all", "low_risk_score_column"] == "spike_score"
    assert rows.loc["all", "low_risk_score_threshold"] == 0.50

    normal = rows.loc["actual_normal_day"]
    spike = rows.loc["actual_spike_day"]
    low_risk = rows.loc["forecast_low_risk_day"]
    high_risk = rows.loc["forecast_high_risk_day"]

    assert normal["n_days"] == 2
    assert normal["n_hours"] == 4
    assert np.isclose(normal["q50_wape"], 10.0 / 86.0)
    assert np.isclose(normal["median_ape"], 0.10)
    assert np.isclose(normal["p75_ape"], 0.11666666666666667)
    assert np.isclose(normal["p90_ape"], 0.14666666666666667)
    assert np.isclose(normal["smape"], 0.11731898917253608)
    assert np.isclose(normal["q50_mae"], 2.5)
    assert np.isclose(normal["q50_bias_mean"], 0.5)

    assert spike["n_days"] == 1
    assert spike["n_hours"] == 2
    assert np.isclose(spike["q50_wape"], 50.0 / 220.0)

    assert low_risk["n_days"] == 2
    assert low_risk["n_hours"] == 4
    assert np.isclose(low_risk["q50_wape"], normal["q50_wape"])
    assert high_risk["n_days"] == 1
    assert high_risk["n_hours"] == 2


def test_normal_day_diagnostics_handles_missing_low_risk_score_column() -> None:
    diagnostics = compute_normal_day_diagnostics(_quantile_prediction_frame(include_spike_score=False))
    rows = diagnostics.set_index("segment")

    for segment in ["forecast_low_risk_day", "forecast_high_risk_day"]:
        assert rows.loc[segment, "n_days"] == 0
        assert rows.loc[segment, "n_hours"] == 0
        assert np.isnan(rows.loc[segment, "q50_wape"])
        assert rows.loc[segment, "low_risk_score_column"] == "spike_score"


def test_normal_day_diagnostics_validates_configuration() -> None:
    frame = _quantile_prediction_frame()

    with pytest.raises(ValueError, match="actual_daily_max_quantile"):
        compute_normal_day_diagnostics(frame, actual_daily_max_quantile=1.0)

    with pytest.raises(ValueError, match="low_risk_aggregation"):
        compute_normal_day_diagnostics(frame, low_risk_aggregation="median")
