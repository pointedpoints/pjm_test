from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


NBEATSX_SCALER_STRATEGIES: dict[str, tuple[str, str]] = {
    "none": ("identity", "identity"),
    "standard": ("identity", "zscore"),
    "robust": ("asinh_q95", "zscore"),
}


@dataclass(slots=True)
class ProjectConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def tuning(self) -> dict[str, Any]:
        return self.raw["tuning"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def report(self) -> dict[str, Any]:
        return self.raw["report"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw.get("retrieval", {})

    @property
    def target_column(self) -> str:
        return str(self.features.get("target_col", "y"))

    @property
    def prediction_horizon(self) -> int:
        return int(self.backtest["horizon"])

    @property
    def prediction_freq(self) -> str:
        return str(self.backtest["freq"])

    @property
    def retrieval_base_model_name(self) -> str:
        return str(self.retrieval.get("base_model_name", "nbeatsx"))

    @property
    def retrieval_output_model_name(self) -> str:
        return str(self.retrieval.get("output_model_name", "nbeatsx_rag"))

    def scaler_candidates(self) -> list[str]:
        scaler_cfg = self.features.get("scaler", {})
        return [str(value) for value in scaler_cfg.get("strategy_candidates", [])]

    def resolved_nbeatsx_scaler_strategy(self) -> str:
        model_cfg = self.models["nbeatsx"]
        pair = (
            str(model_cfg.get("target_transform", "identity")),
            str(model_cfg.get("exog_scaler", "identity")),
        )
        for strategy_name, strategy_pair in NBEATSX_SCALER_STRATEGIES.items():
            if pair == strategy_pair:
                return strategy_name
        raise ValueError(f"Unsupported NBEATSx scaler contract: target_transform={pair[0]!r}, exog_scaler={pair[1]!r}.")

    def nbeatsx_runtime_config(self) -> dict[str, Any]:
        model_cfg = dict(self.models["nbeatsx"])
        configured_h = int(model_cfg.get("h", self.prediction_horizon))
        if configured_h != self.prediction_horizon:
            raise ValueError(
                f"models.nbeatsx.h={configured_h} must match backtest.horizon={self.prediction_horizon} in v1."
            )

        configured_freq = str(model_cfg.get("freq", self.prediction_freq))
        if configured_freq != self.prediction_freq:
            raise ValueError(
                f"models.nbeatsx.freq={configured_freq!r} must match backtest.freq={self.prediction_freq!r} in v1."
            )

        strategy_name = self.resolved_nbeatsx_scaler_strategy()
        strategy_candidates = self.scaler_candidates()
        scaler_enabled = bool(self.features.get("scaler", {}).get("enabled", False))
        if scaler_enabled and strategy_name not in strategy_candidates:
            raise ValueError(
                f"NBEATSx scaler strategy {strategy_name!r} must be listed in features.scaler.strategy_candidates."
            )
        if not scaler_enabled and strategy_name != "none":
            raise ValueError("features.scaler.enabled=false requires NBEATSx to use the 'none' scaler strategy.")

        target_transform, exog_scaler = NBEATSX_SCALER_STRATEGIES[strategy_name]
        model_cfg["h"] = self.prediction_horizon
        model_cfg["freq"] = self.prediction_freq
        model_cfg["target_transform"] = target_transform
        model_cfg["exog_scaler"] = exog_scaler
        return model_cfg

    def validate_runtime_contracts(self) -> None:
        if self.target_column != "y":
            raise ValueError(f"features.target_col={self.target_column!r} is unsupported; v1 requires the canonical target column 'y'.")
        invalid_candidates = [value for value in self.scaler_candidates() if value not in NBEATSX_SCALER_STRATEGIES]
        if invalid_candidates:
            raise ValueError(f"Unsupported features.scaler.strategy_candidates: {invalid_candidates}")
        if "nbeatsx" in self.models:
            self.nbeatsx_runtime_config()

    def resolve_path(self, relative_path: str) -> Path:
        override = os.environ.get("PJM_PROJECT_ROOT_OVERRIDE") or self.project.get("root_override")
        base_path = Path(override).resolve() if override else self.path.parent.parent
        return (base_path / relative_path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ProjectConfig(raw=raw, path=config_path)
    config.validate_runtime_contracts()
    return config
