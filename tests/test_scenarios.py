from __future__ import annotations

import pandas as pd

from pjm_forecast.evaluation.scenarios import compute_scenario_diagnostics
from pjm_forecast.prepared_data import prediction_metadata


def _prediction_frame(split: str) -> pd.DataFrame:
    rows = []
    forecast_days = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    templates = [
        [(8.0, 10.0, 12.0), (18.0, 20.0, 22.0)],
        [(9.0, 11.0, 13.0), (16.0, 18.0, 20.0)],
        [(7.0, 9.0, 11.0), (17.0, 19.0, 21.0)],
    ]
    observations = [[10.0, 20.0], [11.0, 18.0], [9.0, 19.0]]
    for forecast_day, template, observed in zip(forecast_days, templates, observations, strict=True):
        metadata = prediction_metadata(forecast_day)
        for hour_index, ((q10, q50, q90), y_true) in enumerate(zip(template, observed, strict=True)):
            ds = forecast_day + pd.Timedelta(hours=hour_index)
            for quantile, value in [(0.1, q10), (0.5, q50), (0.9, q90)]:
                rows.append(
                    {
                        "ds": ds,
                        "y": y_true,
                        "y_pred": value,
                        "model": "nbeatsx",
                        "split": split,
                        "seed": 7,
                        "quantile": quantile,
                        "metadata": metadata,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_scenario_diagnostics_returns_scores_for_quantile_predictions() -> None:
    diagnostics = compute_scenario_diagnostics(
        _prediction_frame("validation"),
        _prediction_frame("test"),
        family="student_t",
        n_samples=32,
        dof_grid=[3.0, 5.0],
        random_seed=7,
        tail_policy="linear",
    )
    assert diagnostics["has_scenarios"] is True
    assert diagnostics["family"] == "student_t"
    assert diagnostics["tail_policy"] == "linear"
    assert diagnostics["energy_score"] >= 0.0
    assert diagnostics["variogram_score"] >= 0.0
    assert diagnostics["path_mean_mae"] >= 0.0
    assert diagnostics["train_days"] == 3.0
    assert diagnostics["eval_days"] == 3.0


def test_compute_scenario_diagnostics_returns_na_for_point_predictions() -> None:
    point_frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "y": [1.0, 2.0],
            "y_pred": [1.5, 2.5],
            "model": ["seasonal_naive"] * 2,
            "split": ["test"] * 2,
            "seed": [7] * 2,
            "quantile": [pd.NA] * 2,
            "metadata": ["{}"] * 2,
        }
    )
    diagnostics = compute_scenario_diagnostics(point_frame, point_frame)
    assert diagnostics["has_scenarios"] is False
    assert pd.isna(diagnostics["energy_score"])
