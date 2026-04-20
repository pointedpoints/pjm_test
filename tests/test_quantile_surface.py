from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_forecast.quantile_surface import (
    QuantileSurface,
    mean_crps_from_quantile_predictions,
    pit_values_from_quantile_predictions,
    quantile_surfaces_from_frame,
    summarize_pit,
)


def _frame() -> pd.DataFrame:
    rows = []
    for ts, y_true, q10, q50, q90 in [
        (pd.Timestamp("2026-01-01 00:00:00"), 10.0, 8.0, 10.0, 12.0),
        (pd.Timestamp("2026-01-01 01:00:00"), 20.0, 18.0, 20.0, 22.0),
    ]:
        for quantile, value in [(0.1, q10), (0.5, q50), (0.9, q90)]:
            rows.append(
                {
                    "ds": ts,
                    "y": y_true,
                    "y_pred": value,
                    "model": "nbeatsx",
                    "split": "validation",
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_quantile_surface_interpolates_cdf_and_ppf() -> None:
    surface = QuantileSurface.from_quantiles([0.1, 0.5, 0.9], [8.0, 10.0, 12.0])
    assert surface.ppf(0.5) == 10.0
    assert surface.ppf(0.3) == 9.0
    assert surface.cdf(10.0) == 0.5
    assert surface.cdf(9.0) == pytest.approx(0.3)


def test_quantile_surface_interval_and_sampling() -> None:
    surface = QuantileSurface.from_quantiles([0.1, 0.5, 0.9], [8.0, 10.0, 12.0])
    lower, upper = surface.interval(0.8)
    assert lower == 8.0
    assert upper == 12.0
    samples = surface.sample(5, random_state=7)
    assert samples.shape == (5,)
    assert np.all(samples >= 8.0)
    assert np.all(samples <= 12.0)


def test_quantile_surface_crps_and_pit_are_available() -> None:
    surface = QuantileSurface.from_quantiles([0.1, 0.5, 0.9], [8.0, 10.0, 12.0])
    assert surface.pit(10.0) == 0.5
    assert surface.crps(10.0) >= 0.0


def test_quantile_surface_helpers_build_per_timestamp_surfaces() -> None:
    frame = _frame()
    surfaces = quantile_surfaces_from_frame(frame)
    assert list(surfaces) == [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 01:00:00")]
    assert surfaces[pd.Timestamp("2026-01-01 00:00:00")].ppf(0.5) == 10.0


def test_quantile_surface_helpers_compute_crps_and_pit_summary() -> None:
    frame = _frame()
    pit_values = pit_values_from_quantile_predictions(frame)
    assert pit_values.tolist() == [0.5, 0.5]
    assert mean_crps_from_quantile_predictions(frame) >= 0.0
    summary = summarize_pit(frame)
    assert summary["pit_mean"] == 0.5
    assert summary["pit_variance"] == 0.0
