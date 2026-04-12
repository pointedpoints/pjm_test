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
    for path in [Path("configs/pjm_day_ahead_v1.yaml"), Path("configs/pjm_day_ahead_kaggle.yaml")]:
        config = load_config(path)
        runtime_cfg = config.nbeatsx_runtime_config()
        assert config.target_column == "y"
        assert config.prediction_horizon == 24
        assert config.prediction_freq == "h"
        assert config.resolved_nbeatsx_scaler_strategy() == "robust"
        assert runtime_cfg["h"] == config.prediction_horizon
        assert runtime_cfg["freq"] == config.prediction_freq
        assert runtime_cfg["target_transform"] == "asinh_q95"
        assert runtime_cfg["exog_scaler"] == "zscore"
        assert config.retrieval_base_model_name == "nbeatsx"
        assert config.retrieval_output_model_name == "nbeatsx_rag"


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
