from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pjm_forecast.config import load_config
from pjm_forecast.models.epftoolbox_wrappers import _dnn_trials_filename
from pjm_forecast.models.nhits import NHITSModel
from pjm_forecast.models.nbeatsx import AsinhQuantileScaler, NBEATSxModel, ZScoreScaler
from pjm_forecast.models.registry import build_model
from pjm_forecast.prepared_data import FeatureSchema
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
    config = load_config("configs/pjm_day_ahead_current_processed.yaml")
    schema = FeatureSchema(config)
    default_model = build_model(config, "nbeatsx", seed=7)
    single_model = build_model(config, "nbeatsx", seed=7, disable_ensemble=True)
    assert len(default_model.ensemble_members) == 2
    assert single_model.ensemble_members == []
    assert default_model.h == config.prediction_horizon
    assert default_model.freq == config.prediction_freq
    assert default_model.target_transform == "asinh_q95"
    assert default_model.exog_scaler == "zscore"
    assert default_model.loss_name == "huber_mqloss"
    assert default_model.loss_delta == 0.75
    assert 0.5 in default_model.quantiles
    assert default_model.futr_exog_list == schema.nbeatsx_futr_exog_columns()
    assert default_model.hist_exog_list == schema.nbeatsx_hist_exog_columns()
    assert default_model.protected_exog_columns == schema.nbeatsx_protected_exog_columns()


def test_build_model_supports_named_nhits_config(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_current_processed.yaml").read_text(encoding="utf-8"))
    payload["models"]["nhits_quantile"] = {
        "type": "nhits",
        "h": 24,
        "input_size": 336,
        "max_steps": 10,
        "learning_rate": 0.001,
        "batch_size": 16,
        "dropout_prob_theta": 0.0,
        "scaler_type": "identity",
        "stack_types": ["identity", "identity", "identity"],
        "mlp_units": [[256, 256], [256, 256], [256, 256]],
        "loss_name": "huber_mqloss",
        "loss_delta": 0.75,
        "quantiles": [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99],
        "quantile_weights": [1.0, 1.5, 2.0, 1.0, 3.0, 4.0, 5.0],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    model = build_model(config, "nhits_quantile", seed=7)
    assert isinstance(model, NHITSModel)
    assert model.quantile_weights == [1.0, 1.5, 2.0, 1.0, 3.0, 4.0, 5.0]


def test_nbeatsx_accepts_huber_mqloss_configuration() -> None:
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
        futr_exog_list=["zonal_load_forecast"],
        hist_exog_list=["price_lag_24"],
        loss_name="huber_mqloss",
        loss_delta=0.75,
        quantiles=[0.1, 0.5, 0.9],
    )
    assert model.loss_name == "huber_mqloss"
    assert model.loss_delta == 0.75
    assert model.quantiles == [0.1, 0.5, 0.9]


def test_nbeatsx_accepts_quantile_weights_configuration() -> None:
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
        futr_exog_list=["zonal_load_forecast"],
        hist_exog_list=["price_lag_24"],
        loss_name="huber_mqloss",
        loss_delta=0.75,
        quantiles=[0.1, 0.5, 0.9],
        quantile_weights=[1.0, 1.0, 3.0],
    )
    assert model.quantile_weights == [1.0, 1.0, 3.0]


def test_nbeatsx_rejects_non_positive_huber_delta() -> None:
    with pytest.raises(ValueError, match="loss_delta"):
        NBEATSxModel(
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
            futr_exog_list=["zonal_load_forecast"],
            hist_exog_list=["price_lag_24"],
            loss_name="huber_mqloss",
            loss_delta=0.0,
            quantiles=[0.1, 0.5, 0.9],
        )


def test_nbeatsx_snapshot_metadata_round_trip(tmp_path: Path) -> None:
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
        protected_exog_columns=["is_weekend", "is_holiday"],
        target_transform="asinh_q95",
        exog_scaler="zscore",
        loss_name="huber_mqloss",
        loss_delta=0.5,
        quantiles=[0.1, 0.5, 0.9],
        ensemble_members=[{"seed_offset": 0}],
        random_seed=7,
    )
    snapshot_dir = tmp_path / "snapshot"
    model.save(snapshot_dir)
    metadata = json.loads((snapshot_dir / "metadata.json").read_text(encoding="utf-8"))
    loaded = NBEATSxModel.load(snapshot_dir)
    assert metadata["model_config"]["target_transform"] == "asinh_q95"
    assert metadata["model_config"]["protected_exog_columns"] == ["is_weekend", "is_holiday"]
    assert metadata["model_config"]["loss_name"] == "huber_mqloss"
    assert metadata["model_config"]["loss_delta"] == 0.5
    assert loaded.target_transform == "asinh_q95"
    assert loaded.protected_exog_columns == ["is_weekend", "is_holiday"]
    assert loaded.loss_name == "huber_mqloss"
    assert loaded.loss_delta == 0.5
    assert loaded.ensemble_members == [{"seed_offset": 0}]


def test_nhits_accepts_quantile_configuration() -> None:
    model = NHITSModel(
        h=24,
        freq="h",
        input_size=336,
        max_steps=10,
        learning_rate=0.001,
        batch_size=16,
        dropout_prob_theta=0.0,
        scaler_type="identity",
        stack_types=["identity", "identity", "identity"],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        futr_exog_list=["zonal_load_forecast"],
        hist_exog_list=["price_lag_24"],
        loss_name="huber_mqloss",
        loss_delta=0.75,
        quantiles=[0.1, 0.5, 0.9],
        quantile_weights=[1.0, 1.0, 3.0],
    )
    assert model.loss_name == "huber_mqloss"
    assert model.quantiles == [0.1, 0.5, 0.9]
    assert model.quantile_weights == [1.0, 1.0, 3.0]


def test_nbeatsx_spike_stack_requires_spike_hours() -> None:
    with pytest.raises(ValueError, match="spike_hours"):
        NBEATSxModel(
            h=24,
            freq="h",
            input_size=336,
            max_steps=10,
            learning_rate=0.001,
            batch_size=16,
            dropout_prob_theta=0.0,
            scaler_type="identity",
            stack_types=["trend", "seasonality", "identity", "spike"],
            mlp_units=[[256, 256], [256, 256], [256, 256], [256, 256]],
            futr_exog_list=["zonal_load_forecast"],
            hist_exog_list=["price_lag_24"],
        )


def test_nbeatsx_spike_stack_metadata_round_trip(tmp_path: Path) -> None:
    model = NBEATSxModel(
        h=24,
        freq="h",
        input_size=168,
        max_steps=10,
        learning_rate=0.001,
        batch_size=16,
        dropout_prob_theta=0.0,
        scaler_type="identity",
        stack_types=["trend", "seasonality", "identity", "spike"],
        mlp_units=[[256, 256], [256, 256], [256, 256], [256, 256]],
        n_blocks=[1, 1, 1, 1],
        spike_hours=[7, 17, 18, 19, 20],
        spike_kernel="triangle",
        spike_radius=1,
        futr_exog_list=["zonal_load_forecast"],
        hist_exog_list=["price_lag_24"],
        ensemble_members=[],
        random_seed=7,
    )
    snapshot_dir = tmp_path / "spike_snapshot"
    model.save(snapshot_dir)
    loaded = NBEATSxModel.load(snapshot_dir)
    assert loaded.stack_types == ["trend", "seasonality", "identity", "spike"]
    assert loaded.n_blocks == [1, 1, 1, 1]
    assert loaded.spike_hours == [7, 17, 18, 19, 20]
    assert loaded.spike_kernel == "triangle"
    assert loaded.spike_radius == 1


def test_nbeatsx_spike_stack_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="spike_radius"):
        NBEATSxModel(
            h=24,
            freq="h",
            input_size=168,
            max_steps=10,
            learning_rate=0.001,
            batch_size=16,
            dropout_prob_theta=0.0,
            scaler_type="identity",
            stack_types=["trend", "seasonality", "identity", "spike"],
            mlp_units=[[256, 256], [256, 256], [256, 256], [256, 256]],
            n_blocks=[1, 1, 1, 1],
            spike_hours=[7, 17, 18, 19, 20],
            spike_radius=-1,
            futr_exog_list=["zonal_load_forecast"],
            hist_exog_list=["price_lag_24"],
        )
