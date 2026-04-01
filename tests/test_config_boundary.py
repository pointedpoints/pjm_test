from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pjm_forecast.config import load_config


def _write_variant(tmp_path: Path, *, mutate) -> Path:
    payload = yaml.safe_load(Path("configs/pjm_day_ahead_v1.yaml").read_text(encoding="utf-8"))
    mutate(payload)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_shipped_configs_resolve_nbeatsx_runtime_contract() -> None:
    for config_path in [Path("configs/pjm_day_ahead_v1.yaml"), Path("configs/pjm_day_ahead_kaggle.yaml")]:
        config = load_config(config_path)
        runtime = config.nbeatsx_runtime_config()
        assert config.target_column == "y"
        assert runtime["h"] == config.prediction_horizon
        assert runtime["freq"] == config.prediction_freq
        assert runtime["target_transform"] == "asinh_q95"
        assert runtime["exog_scaler"] == "zscore"


def test_load_config_rejects_invalid_target_column(tmp_path: Path) -> None:
    config_path = _write_variant(tmp_path, mutate=lambda payload: payload["features"].__setitem__("target_col", "price"))
    with pytest.raises(ValueError, match="target_col"):
        load_config(config_path)


def test_load_config_rejects_nbeatsx_horizon_drift(tmp_path: Path) -> None:
    config_path = _write_variant(tmp_path, mutate=lambda payload: payload["models"]["nbeatsx"].__setitem__("h", 48))
    with pytest.raises(ValueError, match="must match backtest.horizon"):
        load_config(config_path)


def test_load_config_rejects_incompatible_scaler_strategy_candidates(tmp_path: Path) -> None:
    config_path = _write_variant(
        tmp_path,
        mutate=lambda payload: payload["features"]["scaler"].__setitem__("strategy_candidates", ["none"]),
    )
    with pytest.raises(ValueError, match="must be listed in features.scaler.strategy_candidates"):
        load_config(config_path)
