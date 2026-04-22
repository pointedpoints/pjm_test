from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pjm_forecast.config import load_config


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
        Path("configs/pjm_day_ahead_current_processed.yaml"),
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


def test_current_processed_config_uses_quantile_nbeatsx_contract() -> None:
    config = load_config(Path("configs/pjm_day_ahead_current_processed.yaml"))
    runtime_cfg = config.nbeatsx_runtime_config()

    assert runtime_cfg["loss_name"] == "huber_mqloss"
    assert runtime_cfg["loss_delta"] == 0.75
    assert runtime_cfg["quantiles"][0] == 0.01
    assert runtime_cfg["quantiles"][-1] == 0.99
    assert 0.5 in runtime_cfg["quantiles"]
    assert runtime_cfg["ensemble_members"] == [{"seed_offset": 0}, {"seed_offset": 11}]
    assert config.tuning["metric"] == "pinball"
    assert config.features["price_lags"] == [24]
    postprocess_cfg = config.report["quantile_postprocess"]
    assert postprocess_cfg["monotonic"] is True
    assert postprocess_cfg["calibration"]["enabled"] is True
    assert postprocess_cfg["calibration"]["source_split"] == "validation"
    assert postprocess_cfg["calibration"]["method"] == "cqr_asymmetric"
    assert postprocess_cfg["calibration"]["group_by"] == "hour"
    assert postprocess_cfg["calibration"]["min_group_size"] == 24
    assert postprocess_cfg["calibration"]["interval_coverage_floors"]["0.01-0.99"] == 0.95
    scenario_cfg = config.report["scenario_evaluation"]
    assert scenario_cfg["enabled"] is True
    assert scenario_cfg["copula_family"] == "student_t"
    assert scenario_cfg["n_samples"] == 256


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
