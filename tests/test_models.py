from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecast.config import load_config
from pjm_forecast.models.epftoolbox_wrappers import _dnn_trials_filename
from pjm_forecast.models.nbeatsx import AsinhQuantileScaler, NBEATSxModel, ZScoreScaler
from pjm_forecast.models.registry import build_model
from pjm_forecast.models.seasonal_naive import SeasonalNaiveModel


def test_seasonal_naive_uses_requested_lag() -> None:
    model = SeasonalNaiveModel(seasonal_lag_hours=24)
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": list(range(48)),
        }
    )
    future = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    model.fit(history)
    predictions = model.predict(history_df=history, future_df=future)
    assert predictions["y_pred"].iloc[0] == 24


def test_dnn_trials_filename_matches_epftoolbox_convention() -> None:
    filename = _dnn_trials_filename(
        experiment_id=1,
        nlayers=2,
        dataset="PJM",
        years_test=2,
        shuffle_train=True,
        data_augmentation=False,
        calibration_window_years=2,
    )
    assert filename == "DNN_hyperparameters_nl2_datPJM_YT2_SF_CW2_1"


def test_asinh_quantile_scaler_round_trip() -> None:
    scaler = AsinhQuantileScaler().fit(pd.Series([10.0, 20.0, 30.0, 40.0]))
    transformed = scaler.transform_series(pd.Series([15.0, 25.0, 35.0]))
    restored = scaler.inverse_transform_array(transformed.to_numpy())
    assert np.allclose(restored, np.array([15.0, 25.0, 35.0]))


def test_asinh_quantile_scaler_inverse_is_finite_for_large_values() -> None:
    scaler = AsinhQuantileScaler().fit(pd.Series([10.0, 20.0, 30.0, 40.0]))
    restored = scaler.inverse_transform_array(np.array([1000.0, -1000.0]))
    assert np.isfinite(restored).all()


def test_zscore_scaler_skips_time_logic_and_scales_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "system_load_forecast": [100.0, 110.0, 120.0],
            "zonal_load_forecast_lag_24": [10.0, 20.0, 30.0],
        }
    )
    scaler = ZScoreScaler().fit(frame, ["system_load_forecast", "zonal_load_forecast_lag_24"])
    transformed = scaler.transform_frame(frame, ["system_load_forecast", "zonal_load_forecast_lag_24"])
    assert np.allclose(transformed.mean().to_numpy(), np.array([0.0, 0.0]), atol=1e-7)


def test_nbeatsx_resolves_ensemble_members_without_mutating_config() -> None:
    model = NBEATSxModel(
        h=24,
        freq="h",
        input_size=336,
        max_steps=10,
        learning_rate=0.001,
        batch_size=16,
        dropout_prob_theta=0.0,
        scaler_type="identity",
        stack_types=["trend", "seasonality", "identity"],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        futr_exog_list=["system_load_forecast"],
        hist_exog_list=["price_lag_168"],
        ensemble_members=[
            {"seed_offset": 0},
            {"seed_offset": 5, "input_size": 168},
        ],
        random_seed=7,
    )
    resolved = model._resolved_member_kwargs()
    assert [member["random_seed"] for member in resolved] == [7, 12]
    assert [member["input_size"] for member in resolved] == [336, 168]
    assert model.ensemble_members == [{"seed_offset": 0}, {"seed_offset": 5, "input_size": 168}]


def test_build_model_can_disable_nbeatsx_ensemble() -> None:
    config = load_config("configs/pjm_day_ahead_v1.yaml")
    default_model = build_model(config, "nbeatsx", seed=7)
    single_model = build_model(config, "nbeatsx", seed=7, disable_ensemble=True)
    assert len(default_model.ensemble_members) >= 2
    assert single_model.ensemble_members == []
