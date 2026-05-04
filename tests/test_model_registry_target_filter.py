from __future__ import annotations

from pathlib import Path

import yaml

from pjm_forecast.config import ProjectConfig
from pjm_forecast.models.registry import build_model
from pjm_forecast.models.target_filter import SpikeFilteredTargetModel


def _config_with_filtered_lightgbm() -> ProjectConfig:
    raw = yaml.safe_load(Path("configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml").read_text(encoding="utf-8"))
    raw["models"]["lightgbm_q_filtered"] = dict(raw["models"]["lightgbm_q"])
    raw["models"]["lightgbm_q_filtered"]["target_filter"] = {
        "enabled": True,
        "window_observations": 365,
        "min_history": 60,
        "quantile": 0.95,
        "fallback_quantile": 0.975,
        "iqr_multiplier": 3.0,
    }
    return ProjectConfig(
        raw=raw,
        path=Path("configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml").resolve(),
    )


def test_registry_wraps_model_when_target_filter_is_enabled() -> None:
    config = _config_with_filtered_lightgbm()

    model = build_model(config, "lightgbm_q_filtered", seed=7)

    assert isinstance(model, SpikeFilteredTargetModel)
    assert model.filter_config.min_history == 60
    assert model.filter_config.window_observations == 365


def test_registry_leaves_model_unwrapped_when_target_filter_is_absent() -> None:
    config = _config_with_filtered_lightgbm()

    model = build_model(config, "lightgbm_q", seed=7)

    assert not isinstance(model, SpikeFilteredTargetModel)
