from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.evaluation.regime_metrics import compute_regime_metrics


def test_compute_regime_metrics_splits_point_and_tail_diagnostics_by_price_regime() -> None:
    ds = pd.date_range("2026-01-01 00:00:00", periods=10, freq="h")
    y_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 30.0, 50.0]
    q50_values = [9.0, 11.0, 13.0, 12.0, 14.0, 14.0, 17.0, 17.0, 25.0, 40.0]
    q01_values = [value - 3.0 for value in q50_values]
    q05_values = [value - 2.0 for value in q50_values]
    q10_values = [value - 1.0 for value in q50_values]
    q90_values = [value + 2.0 for value in q50_values]
    q95_values = [value + 3.0 for value in q50_values]
    q99_values = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 23.0, 28.0, 45.0]

    rows = []
    for index, ts in enumerate(ds):
        for quantile, predictions in [
            (0.01, q01_values),
            (0.05, q05_values),
            (0.10, q10_values),
            (0.50, q50_values),
            (0.90, q90_values),
            (0.95, q95_values),
            (0.99, q99_values),
        ]:
            rows.append(
                {
                    "ds": ts,
                    "y": y_values[index],
                    "y_pred": predictions[index],
                    "model": "quantile_dummy",
                    "split": "test",
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    predictions = pd.DataFrame(rows)

    metrics = compute_regime_metrics(predictions)

    assert set(metrics["regime"]) == {"all", "normal", "high", "spike", "extreme", "daily_max"}
    all_row = metrics.loc[metrics["regime"] == "all"].iloc[0]
    extreme_row = metrics.loc[metrics["regime"] == "extreme"].iloc[0]
    daily_max_row = metrics.loc[metrics["regime"] == "daily_max"].iloc[0]
    assert all_row["n"] == 10
    assert np.isclose(all_row["p50_mae"], 2.1)
    assert np.isclose(all_row["q99_exceedance_rate"], 0.2)
    assert np.isclose(extreme_row["p50_underprediction_mean"], 10.0)
    assert np.isclose(extreme_row["worst_q99_underprediction"], 5.0)
    assert daily_max_row["n"] == 1
    assert np.isclose(daily_max_row["daily_max_q99_gap_max"], 5.0)
