from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.copula import (
    GaussianCopula,
    StudentTCopula,
    build_quantile_surface_panel,
    energy_score,
    fit_copula_from_predictions,
    sample_copula_scenarios,
    variogram_score,
)
from pjm_forecast.prepared_data import prediction_metadata


def _prediction_frame() -> pd.DataFrame:
    rows = []
    forecast_days = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    hourly_templates = [
        [(8.0, 10.0, 12.0), (18.0, 20.0, 22.0)],
        [(9.0, 11.0, 13.0), (16.0, 18.0, 20.0)],
        [(7.0, 9.0, 11.0), (17.0, 19.0, 21.0)],
    ]
    observations = [[10.0, 20.0], [11.0, 18.0], [9.0, 19.0]]
    for forecast_day, template, observed in zip(forecast_days, hourly_templates, observations, strict=True):
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
                        "split": "validation",
                        "seed": 7,
                        "quantile": quantile,
                        "metadata": metadata,
                    }
                )
    return pd.DataFrame(rows)


def test_build_quantile_surface_panel_extracts_days_and_pseudo_observations() -> None:
    panel = build_quantile_surface_panel(_prediction_frame())
    assert panel.horizon == 2
    assert len(panel.forecast_days) == 3
    assert panel.pseudo_observations().shape == (3, 2)


def test_gaussian_copula_fit_and_sample() -> None:
    uniforms = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.7],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    copula = GaussianCopula.fit(uniforms)
    samples = copula.sample(16, random_state=7)
    assert samples.shape == (16, 2)
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)
    assert copula.log_likelihood(uniforms) == copula.log_likelihood(uniforms)


def test_student_t_copula_fit_and_sample() -> None:
    uniforms = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.7],
            [0.4, 0.5],
            [0.6, 0.65],
        ],
        dtype=float,
    )
    copula = StudentTCopula.fit(uniforms, dof_grid=[3.0, 5.0, 10.0])
    samples = copula.sample(12, random_state=7)
    assert samples.shape == (12, 2)
    assert copula.degrees_of_freedom in {3.0, 5.0, 10.0}
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)


def test_fit_copula_from_predictions_and_sample_scenarios() -> None:
    copula, panel = fit_copula_from_predictions(_prediction_frame(), family="student_t", dof_grid=[3.0, 5.0])
    marginals = panel.marginals_for_day("2026-01-03")
    scenarios = sample_copula_scenarios(copula, marginals, 8, random_state=7)
    assert scenarios.shape == (8, 2)
    assert np.all(np.isfinite(scenarios))


def test_energy_and_variogram_score_are_non_negative() -> None:
    observation = np.array([10.0, 20.0], dtype=float)
    scenarios = np.array([[9.0, 19.0], [10.5, 20.5], [11.0, 21.0]], dtype=float)
    assert energy_score(observation, scenarios) >= 0.0
    assert variogram_score(observation, scenarios) >= 0.0
