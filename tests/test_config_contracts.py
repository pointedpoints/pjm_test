from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pjm_forecast.config import load_config
from pjm_forecast.prepared_data import FeatureSchema


def _write_temp_config(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    mutate(payload)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_runtime_contract_matches_across_v1_and_kaggle_configs() -> None:
    for path in [
        Path("configs/pjm_day_ahead_v1.yaml"),
        Path("configs/pjm_day_ahead_kaggle.yaml"),
    ]:
        config = load_config(path)
        runtime_cfg = config.nbeatsx_runtime_config()
        assert config.target_column == "y"
        assert config.prediction_horizon == 24
        assert config.prediction_freq == "h"
        assert config.resolved_neuralforecast_scaler_strategy("nbeatsx") == "robust"
        assert runtime_cfg["h"] == config.prediction_horizon
        assert runtime_cfg["freq"] == config.prediction_freq
        assert runtime_cfg["target_transform"] == "asinh_q95"
        assert runtime_cfg["exog_scaler"] == "zscore"
        assert config.retrieval_base_model_name == "nbeatsx"
        assert config.retrieval_output_model_name == "nbeatsx_rag"

    for path, model_name in [
        (Path("configs/pjm_day_ahead_current_processed.yaml"), "nhits_tail_grid_weighted_main"),
    ]:
        config = load_config(path)
        runtime_cfg = config.runtime_model_config(model_name)
        assert config.target_column == "y"
        assert config.prediction_horizon == 24
        assert config.prediction_freq == "h"
        assert config.resolved_neuralforecast_scaler_strategy(model_name) == "robust"
        assert runtime_cfg["h"] == config.prediction_horizon
        assert runtime_cfg["freq"] == config.prediction_freq
        assert runtime_cfg["target_transform"] == "asinh_q95"
        assert runtime_cfg["exog_scaler"] == "zscore"
        assert config.retrieval_base_model_name == "nbeatsx"
        assert config.retrieval_output_model_name == "nbeatsx_rag"


def test_current_processed_config_uses_nhits_tail_grid_contract() -> None:
    config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    runtime_cfg = config.runtime_model_config("nhits_tail_grid_weighted_main")

    assert config.backtest["benchmark_models"] == ["nhits_tail_grid_weighted_main"]
    assert config.tuning["model_name"] == "nhits_tail_grid_weighted_main"
    assert runtime_cfg["type"] == "nhits"
    assert runtime_cfg["loss_name"] == "huber_mqloss"
    assert runtime_cfg["loss_delta"] == 0.75
    assert runtime_cfg["quantiles"][0] == 0.01
    assert runtime_cfg["quantiles"][-3:] == [0.975, 0.99, 0.995]
    assert 0.5 in runtime_cfg["quantiles"]
    assert len(runtime_cfg["quantile_weights"]) == len(runtime_cfg["quantiles"])
    assert len(runtime_cfg["quantile_deltas"]) == len(runtime_cfg["quantiles"])
    assert runtime_cfg["monotonicity_penalty"] == 0.03
    assert runtime_cfg["ensemble_members"] == [{"seed_offset": 0}]
    assert config.tuning["metric"] == "pinball"
    assert config.features["price_lags"] == [24]
    assert "spike_score" not in config.features["future_exog"]
    assert any(item.get("name") == "spike_score" for item in config.features["derived_features"])
    postprocess_cfg = config.report["quantile_postprocess"]
    assert postprocess_cfg["monotonic"] is True
    assert postprocess_cfg["median_bias"]["enabled"] is False
    assert postprocess_cfg["median_bias"]["source_split"] == "validation"
    assert postprocess_cfg["median_bias"]["group_by"] == "hour"
    assert postprocess_cfg["median_bias"]["regime_score_column"] == "spike_score"
    assert postprocess_cfg["median_bias"]["regime_threshold"] == 0.50
    assert postprocess_cfg["median_bias"]["max_abs_adjustment"] == 20.0
    assert postprocess_cfg["calibration"]["enabled"] is True
    assert postprocess_cfg["calibration"]["source_split"] == "validation"
    assert postprocess_cfg["calibration"]["method"] == "cqr_asymmetric"
    assert postprocess_cfg["calibration"]["group_by"] == "hour"
    assert postprocess_cfg["calibration"]["regime_score_column"] == "spike_score"
    assert postprocess_cfg["calibration"]["regime_threshold"] == 0.50
    assert postprocess_cfg["calibration"]["min_group_size"] == 24
    assert postprocess_cfg["calibration"]["interval_coverage_floors"]["0.01-0.99"] == 0.95
    scenario_cfg = config.report["scenario_evaluation"]
    assert scenario_cfg["enabled"] is True
    assert scenario_cfg["copula_family"] == "student_t"
    assert scenario_cfg["tail_policy"] == "linear"
    assert scenario_cfg["n_samples"] == 256


def test_tail_grid_phase1_config_uses_raw_nhits_upper_tail_contract() -> None:
    config = load_config(Path("configs/experiments/pjm_current_validation_nhits_tail_grid_phase1.yaml"))
    baseline_cfg = config.runtime_model_config("nhits_baseline")
    tail_cfg = config.runtime_model_config("nhits_tail_grid_weighted")

    assert config.backtest["benchmark_models"] == ["nhits_baseline", "nhits_tail_grid", "nhits_tail_grid_weighted"]
    assert config.report["quantile_postprocess"]["monotonic"] is False
    assert config.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert config.report["scenario_evaluation"]["enabled"] is False
    assert baseline_cfg["quantiles"][-1] == 0.99
    assert tail_cfg["type"] == "nhits"
    assert tail_cfg["quantiles"][-3:] == [0.975, 0.99, 0.995]
    assert len(tail_cfg["quantile_weights"]) == len(tail_cfg["quantiles"])
    assert len(tail_cfg["quantile_deltas"]) == len(tail_cfg["quantiles"])
    assert tail_cfg["quantile_weights"][-1] > tail_cfg["quantile_weights"][-2] > tail_cfg["quantile_weights"][-3]
    assert tail_cfg["monotonicity_penalty"] == 0.03


def test_q50_weight_grid_config_only_changes_median_quantile_weight() -> None:
    config = load_config(Path("configs/experiments/pjm_current_validation_nhits_q50_weight_grid.yaml"))
    base_cfg = config.runtime_model_config("nhits_q50w100")
    q50w125_cfg = config.runtime_model_config("nhits_q50w125")
    q50w150_cfg = config.runtime_model_config("nhits_q50w150")
    median_index = base_cfg["quantiles"].index(0.5)

    assert config.backtest["benchmark_models"] == ["nhits_q50w100", "nhits_q50w125", "nhits_q50w150"]
    assert config.report["quantile_postprocess"]["monotonic"] is True
    assert config.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert config.report["scenario_evaluation"]["enabled"] is False
    assert base_cfg["quantile_weights"][median_index] == 1.0
    assert q50w125_cfg["quantile_weights"][median_index] == 1.25
    assert q50w150_cfg["quantile_weights"][median_index] == 1.5

    for left, right in [(base_cfg, q50w125_cfg), (base_cfg, q50w150_cfg)]:
        assert left["quantiles"] == right["quantiles"]
        assert left["quantile_deltas"] == right["quantile_deltas"]
        assert left["monotonicity_penalty"] == right["monotonicity_penalty"]
        left_weights = list(left["quantile_weights"])
        right_weights = list(right["quantile_weights"])
        left_weights.pop(median_index)
        right_weights.pop(median_index)
        assert left_weights == right_weights


def test_q50w150_test_candidate_uses_canonical_cqr_contract() -> None:
    config = load_config(Path("configs/experiments/pjm_current_test_nhits_q50w150.yaml"))
    runtime_cfg = config.runtime_model_config("nhits_q50w150")
    median_index = runtime_cfg["quantiles"].index(0.5)
    postprocess_cfg = config.report["quantile_postprocess"]

    assert config.backtest["benchmark_models"] == ["nhits_q50w150"]
    assert runtime_cfg["type"] == "nhits"
    assert runtime_cfg["quantile_weights"][median_index] == 1.5
    assert postprocess_cfg["monotonic"] is True
    assert postprocess_cfg["median_bias"]["enabled"] is False
    assert postprocess_cfg["calibration"]["enabled"] is True
    assert postprocess_cfg["calibration"]["method"] == "cqr_asymmetric"
    assert postprocess_cfg["calibration"]["group_by"] == "hour_x_regime"
    assert postprocess_cfg["calibration"]["regime_score_column"] == "spike_score"
    assert postprocess_cfg["calibration"]["regime_threshold"] == 0.50
    assert config.report["scenario_evaluation"]["enabled"] is True
    assert config.report["scenario_evaluation"]["tail_policy"] == "linear"


def test_phase1_benchmark_floor_config_restores_p50_feature_contract() -> None:
    config = load_config(Path("configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml"))

    assert config.backtest["benchmark_models"] == ["seasonal_naive", "lear", "lightgbm_q", "xgboost_q"]
    assert config.features["price_lags"] == [24, 168]
    assert config.features["source_lags"] == [24, 168]
    assert "system_load_forecast" in config.features["future_exog"]
    assert any(item.get("name") == "prior_day_price_max" for item in config.features["derived_features"])
    assert any(item.get("name") == "prior_day_price_max_ramp" for item in config.features["derived_features"])
    assert config.models["lightgbm_q"]["quantiles"] == [0.10, 0.50, 0.90]
    assert config.models["xgboost_q"]["quantiles"] == [0.10, 0.50, 0.90]
    assert config.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert config.report["scenario_evaluation"]["enabled"] is False


def test_phase1_p50_friendly_neural_config_uses_restored_features_and_moderate_quantile_grid() -> None:
    config = load_config(Path("configs/experiments/pjm_current_validation_phase1_p50_friendly_neural.yaml"))
    nhits_cfg = config.runtime_model_config("nhits_p50_friendly")
    nbeatsx_cfg = config.runtime_model_config("nbeatsx_p50_friendly")

    assert config.backtest["benchmark_models"] == ["nhits_p50_friendly", "nbeatsx_p50_friendly"]
    assert "system_load_forecast" in config.features["future_exog"]
    assert "prior_day_price_max" in config.features["future_exog"]
    assert "prior_day_price_max_ramp" in config.features["future_exog"]
    assert config.features["price_lags"] == [24, 168]
    assert config.features["source_lags"] == [24, 168]
    assert nhits_cfg["quantiles"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    assert nhits_cfg["quantile_weights"][4] == 1.10
    assert nhits_cfg["monotonicity_penalty"] == 0.01
    assert nbeatsx_cfg["quantile_weights"] == nhits_cfg["quantile_weights"]
    assert nbeatsx_cfg["quantile_deltas"] == nhits_cfg["quantile_deltas"]
    assert config.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert config.report["scenario_evaluation"]["enabled"] is False


def test_p50_future_price_lag_experiment_configs_use_horizon_aligned_lags() -> None:
    lag168 = load_config(Path("configs/experiments/pjm_current_p50_futr_lag168.yaml"))
    lag168_336 = load_config(Path("configs/experiments/pjm_current_p50_futr_lag168_336.yaml"))

    lag168_schema = FeatureSchema(lag168)
    lag168_336_schema = FeatureSchema(lag168_336)

    assert lag168.backtest["benchmark_models"] == ["nhits_p50_futr_lag168"]
    assert lag168_336.backtest["benchmark_models"] == ["nhits_p50_futr_lag168_336"]
    assert "future_price_lag_168" in lag168_schema.nbeatsx_futr_exog_columns()
    assert "future_price_lag_168" not in lag168_schema.nbeatsx_hist_exog_columns()
    assert "future_price_lag_336" not in lag168_schema.nbeatsx_futr_exog_columns()
    assert "future_price_lag_168" in lag168_336_schema.nbeatsx_futr_exog_columns()
    assert "future_price_lag_336" in lag168_336_schema.nbeatsx_futr_exog_columns()
    assert "future_price_lag_336" not in lag168_336_schema.nbeatsx_hist_exog_columns()
    assert any(item.get("kind") == "future_known_lag" for item in lag168.features["derived_features"])
    assert lag168.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert lag168.report["scenario_evaluation"]["enabled"] is False


def test_p50_price_state_experiment_config_uses_prior_day_state_as_model_features() -> None:
    config = load_config(Path("configs/experiments/pjm_current_p50_price_state.yaml"))
    schema = FeatureSchema(config)

    assert config.backtest["benchmark_models"] == ["nhits_p50_price_state"]
    for column in ["prior_day_price_max", "prior_day_price_spread", "prior_day_price_max_ramp"]:
        assert column in schema.nbeatsx_futr_exog_columns()
        assert column not in schema.nbeatsx_hist_exog_columns()
    stats = {item["name"]: item["stat"] for item in config.features["derived_features"] if item.get("kind") == "prior_day_price_stat"}
    assert stats == {
        "prior_day_price_max": "max",
        "prior_day_price_spread": "spread",
        "prior_day_price_max_ramp": "max_ramp",
    }
    assert config.report["quantile_postprocess"]["calibration"]["enabled"] is False
    assert config.report["scenario_evaluation"]["enabled"] is False


def test_phase1_nbeatsx_test_raw_config_carries_spike_score_context_without_enabling_calibration() -> None:
    config = load_config(Path("configs/experiments/pjm_current_test_phase1_nbeatsx_p50_raw.yaml"))

    assert config.backtest["benchmark_models"] == ["nbeatsx_p50_friendly"]
    assert "spike_score" in config.features["future_exog"]

    calibration = config.report["quantile_postprocess"]["calibration"]
    assert calibration["enabled"] is False
    assert calibration["group_by"] == "hour_x_regime"
    assert calibration["regime_score_column"] == "spike_score"


def test_phase1_nbeatsx_calibration_compare_configs_split_hour_and_hour_x_regime() -> None:
    hour = load_config(Path("configs/experiments/pjm_current_test_phase1_nbeatsx_p50_hour.yaml"))
    hour_x_regime = load_config(Path("configs/experiments/pjm_current_test_phase1_nbeatsx_p50_hour_x_regime.yaml"))

    hour_calibration = hour.report["quantile_postprocess"]["calibration"]
    regime_calibration = hour_x_regime.report["quantile_postprocess"]["calibration"]

    assert hour_calibration["enabled"] is True
    assert hour_calibration["group_by"] == "hour"
    assert "regime_score_column" not in hour_calibration

    assert regime_calibration["enabled"] is True
    assert regime_calibration["group_by"] == "hour_x_regime"
    assert regime_calibration["regime_score_column"] == "spike_score"
    assert hour.project["directories"]["prediction_dir"] == hour_x_regime.project["directories"]["prediction_dir"]


def test_load_config_rejects_nbeatsx_horizon_drift(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path, lambda payload: payload["models"]["nbeatsx"].__setitem__("h", 12))
    with pytest.raises(ValueError, match="must match backtest.horizon"):
        load_config(config_path)


def test_load_config_rejects_unsupported_target_column(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path, lambda payload: payload["features"].__setitem__("target_col", "price"))
    with pytest.raises(ValueError, match="target_col"):
        load_config(config_path)


def test_load_config_rejects_scaler_strategy_not_listed_in_candidates(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["features"]["scaler"].__setitem__("strategy_candidates", ["none", "standard"]),
    )
    with pytest.raises(ValueError, match="strategy_candidates"):
        load_config(config_path)


def test_load_config_rejects_derived_feature_with_missing_dependency(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["features"].__setitem__(
            "derived_features",
            [
                {
                    "kind": "multiply",
                    "left": "missing_feature",
                    "right": "is_weekend",
                    "name": "bad_interaction",
                }
            ],
        ),
    )
    with pytest.raises(ValueError, match="multiply inputs are unavailable"):
        load_config(config_path)


def test_load_config_allows_hidden_weather_dependency_for_derived_feature(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.__setitem__("weather", payload.get("weather", {})),
            payload["weather"].__setitem__("enabled", True),
            payload["weather"].__setitem__("provider", "open_meteo_historical_forecast"),
            payload["weather"].__setitem__("output_columns", ["weather_apparent_temp_mean"]),
            payload["weather"].__setitem__(
                "points",
                [{"name": "demo", "latitude": 1.0, "longitude": 2.0, "weight": 1.0}],
            ),
            payload["features"].__setitem__("future_exog", ["zonal_load_forecast", "heating_degree_hidden"]),
            payload["features"].__setitem__("lag_sources", ["zonal_load_forecast"]),
            payload["features"].__setitem__(
                "derived_features",
                [
                    {
                        "kind": "degree_day",
                        "source": "weather_apparent_temp_mean",
                        "mode": "heating",
                        "base": 18.0,
                        "name": "heating_degree_hidden",
                    }
                ],
            ),
        ),
    )
    load_config(config_path)


def test_load_config_rejects_invalid_hour_indicator(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["features"].__setitem__(
            "derived_features",
            [{"kind": "hour_indicator", "hour": 25, "name": "bad_hour"}],
        ),
    )
    with pytest.raises(ValueError, match="hour_indicator"):
        load_config(config_path)


def test_load_config_rejects_invalid_prior_day_price_stat(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["features"].__setitem__(
            "derived_features",
            [{"kind": "prior_day_price_stat", "source": "y", "stat": "median", "name": "prior_day_price_median"}],
        ),
    )
    with pytest.raises(ValueError, match="prior_day_price_stat"):
        load_config(config_path)


def test_load_config_rejects_invalid_pre_holiday_window(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["features"].__setitem__(
            "derived_features",
            [{"kind": "pre_holiday_window", "max_days": 0, "name": "bad_pre_holiday"}],
        ),
    )
    with pytest.raises(ValueError, match="max_days >= 1"):
        load_config(config_path)


def test_load_config_rejects_unsupported_quantile_calibration_method(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "calibration": {
                        "enabled": True,
                        "source_split": "validation",
                        "method": "unsupported_method",
                    },
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported report.quantile_postprocess.calibration.method"):
        load_config(config_path)


def test_load_config_rejects_quantile_contract_without_median(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.9]),
        ),
    )
    with pytest.raises(ValueError, match="include 0.5"):
        load_config(config_path)


def test_load_config_allows_huber_mqloss_for_quantile_training(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "huber_mqloss"),
            payload["models"]["nbeatsx"].__setitem__("loss_delta", 0.75),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
        ),
    )
    config = load_config(config_path)
    assert config.nbeatsx_runtime_config()["loss_name"] == "huber_mqloss"
    assert config.nbeatsx_runtime_config()["loss_delta"] == 0.75


def test_load_config_allows_quantile_weights_for_neuralforecast_models(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "huber_mqloss"),
            payload["models"]["nbeatsx"].__setitem__("loss_delta", 0.75),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["models"]["nbeatsx"].__setitem__("quantile_weights", [1.0, 1.0, 3.0]),
        ),
    )
    config = load_config(config_path)
    assert config.nbeatsx_runtime_config()["quantile_weights"] == [1.0, 1.0, 3.0]


def test_load_config_allows_quantile_deltas_and_monotonicity_penalty(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "huber_mqloss"),
            payload["models"]["nbeatsx"].__setitem__("loss_delta", 0.75),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["models"]["nbeatsx"].__setitem__("quantile_deltas", [1.25, 0.75, 1.25]),
            payload["models"]["nbeatsx"].__setitem__("monotonicity_penalty", 0.05),
        ),
    )
    runtime_cfg = load_config(config_path).nbeatsx_runtime_config()
    assert runtime_cfg["quantile_deltas"] == [1.25, 0.75, 1.25]
    assert runtime_cfg["monotonicity_penalty"] == 0.05


def test_load_config_rejects_invalid_quantile_weights_length(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["models"]["nbeatsx"].__setitem__("quantile_weights", [1.0, 3.0]),
        ),
    )
    with pytest.raises(ValueError, match="quantile_weights"):
        load_config(config_path)


def test_load_config_rejects_invalid_quantile_deltas_length(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "huber_mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["models"]["nbeatsx"].__setitem__("quantile_deltas", [0.75, 1.25]),
        ),
    )
    with pytest.raises(ValueError, match="quantile_deltas"):
        load_config(config_path)


def test_load_config_rejects_negative_monotonicity_penalty(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "huber_mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["models"]["nbeatsx"].__setitem__("monotonicity_penalty", -0.01),
        ),
    )
    with pytest.raises(ValueError, match="monotonicity_penalty"):
        load_config(config_path)


def test_runtime_model_config_supports_named_nhits_models(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: payload["models"].__setitem__(
            "nhits_quantile",
            {
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
                "quantiles": [0.1, 0.5, 0.9],
                "quantile_weights": [1.0, 1.0, 3.0],
                "quantile_deltas": [1.25, 0.75, 1.25],
                "monotonicity_penalty": 0.05,
            },
        ),
    )
    config = load_config(config_path)
    runtime_cfg = config.runtime_model_config("nhits_quantile")
    assert runtime_cfg["type"] == "nhits"
    assert runtime_cfg["loss_name"] == "huber_mqloss"
    assert runtime_cfg["quantile_weights"] == [1.0, 1.0, 3.0]
    assert runtime_cfg["quantile_deltas"] == [1.25, 0.75, 1.25]
    assert runtime_cfg["monotonicity_penalty"] == 0.05


def test_load_config_rejects_cqr_quantiles_without_symmetric_pairs(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "calibration": {
                        "enabled": True,
                        "source_split": "validation",
                        "method": "cqr",
                    },
                },
            ),
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.2, 0.5, 0.9]),
        ),
    )
    with pytest.raises(ValueError, match="requires symmetric quantile pairs"):
        load_config(config_path)


def test_load_config_allows_asymmetric_cqr_with_hour_grouping_and_coverage_floors(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "calibration": {
                        "enabled": True,
                        "source_split": "validation",
                        "method": "cqr_asymmetric",
                        "group_by": "hour",
                        "min_group_size": 24,
                        "interval_coverage_floors": {"0.10-0.90": 0.76},
                    },
                },
            ),
        ),
    )
    config = load_config(config_path)
    calibration = config.report["quantile_postprocess"]["calibration"]
    assert calibration["method"] == "cqr_asymmetric"
    assert calibration["group_by"] == "hour"


def test_load_config_allows_hour_x_regime_calibration(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload["features"]["future_exog"].append("spike_score"),
            payload["features"].setdefault("derived_features", []).append(
                {
                    "kind": "spike_score",
                    "name": "spike_score",
                    "inputs": [
                        {"source": "zonal_load_forecast", "weight": 0.7},
                        {"source": "system_load_forecast", "weight": 0.3},
                    ],
                }
            ),
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "calibration": {
                        "enabled": True,
                        "source_split": "validation",
                        "method": "cqr_asymmetric",
                        "group_by": "hour_x_regime",
                        "regime_score_column": "spike_score",
                        "regime_threshold": 0.67,
                        "min_group_size": 24,
                    },
                },
            ),
        ),
    )
    calibration = load_config(config_path).report["quantile_postprocess"]["calibration"]
    assert calibration["group_by"] == "hour_x_regime"
    assert calibration["regime_score_column"] == "spike_score"


def test_load_config_allows_median_bias_calibration(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "median_bias": {
                        "enabled": True,
                        "source_split": "validation",
                        "group_by": "hour",
                        "min_group_size": 24,
                        "max_abs_adjustment": 10.0,
                    },
                },
            ),
        ),
    )
    median_bias = load_config(config_path).report["quantile_postprocess"]["median_bias"]
    assert median_bias["enabled"] is True
    assert median_bias["group_by"] == "hour"


def test_load_config_rejects_invalid_median_bias_contract(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "median_bias": {
                        "enabled": True,
                        "source_split": "test",
                    },
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="median_bias.source_split"):
        load_config(config_path)


def test_load_config_rejects_invalid_quantile_coverage_floor_key(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload["models"]["nbeatsx"].__setitem__("loss_name", "mqloss"),
            payload["models"]["nbeatsx"].__setitem__("quantiles", [0.1, 0.5, 0.9]),
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "quantile_postprocess",
                {
                    "monotonic": True,
                    "calibration": {
                        "enabled": True,
                        "source_split": "validation",
                        "method": "cqr_asymmetric",
                        "interval_coverage_floors": {"0.20-0.80": 0.76},
                    },
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="unsupported pair"):
        load_config(config_path)


def test_load_config_rejects_unsupported_scenario_copula_family(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "scenario_evaluation",
                {
                    "enabled": True,
                    "source_split": "validation",
                    "copula_family": "vine",
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported report.scenario_evaluation.copula_family"):
        load_config(config_path)


def test_load_config_rejects_unsupported_scenario_tail_policy(tmp_path: Path) -> None:
    config_path = _write_temp_config(
        tmp_path,
        lambda payload: (
            payload.setdefault("report", {}),
            payload["report"].__setitem__(
                "scenario_evaluation",
                {
                    "enabled": True,
                    "source_split": "validation",
                    "copula_family": "student_t",
                    "tail_policy": "gpd",
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported report.scenario_evaluation.tail_policy"):
        load_config(config_path)
