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
    def dataset_source_type(self) -> str:
        return str(self.dataset.get("source_type", "epftoolbox"))

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
    def weather(self) -> dict[str, Any]:
        return self.raw.get("weather", {})

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

    @property
    def weather_enabled(self) -> bool:
        return bool(self.weather.get("enabled", False))

    def weather_output_columns(self) -> list[str]:
        return [str(value) for value in self.weather.get("output_columns", [])]

    def without_weather_feature_contracts(self) -> "ProjectConfig":
        raw_copy = yaml.safe_load(yaml.safe_dump(self.raw, sort_keys=False))
        weather_columns = set(self.weather_output_columns())
        raw_copy["features"]["future_exog"] = [
            column for column in raw_copy["features"].get("future_exog", [])
            if str(column) not in weather_columns
        ]
        if "lag_sources" in raw_copy["features"]:
            raw_copy["features"]["lag_sources"] = [
                column for column in raw_copy["features"]["lag_sources"]
                if str(column) not in weather_columns
            ]
        raw_copy["weather"] = dict(raw_copy.get("weather", {}))
        raw_copy["weather"]["enabled"] = False
        return ProjectConfig(raw=raw_copy, path=self.path)

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
        if self.dataset_source_type not in {"epftoolbox", "pjm_official", "official_weather_ready"}:
            raise ValueError(f"Unsupported dataset.source_type={self.dataset_source_type!r}.")
        if self.weather_enabled:
            self.validate_weather_contracts()
        if "nbeatsx" in self.models:
            self.nbeatsx_runtime_config()

    def validate_weather_contracts(self) -> None:
        weather_cfg = self.weather
        provider = str(weather_cfg.get("provider", ""))
        if provider not in {"open_meteo_historical_forecast"}:
            raise ValueError(f"Unsupported weather.provider={provider!r}.")

        points = weather_cfg.get("points", [])
        if not points:
            raise ValueError("weather.enabled=true requires at least one configured weather.points entry.")
        for point in points:
            missing = [key for key in ["name", "latitude", "longitude", "weight"] if key not in point]
            if missing:
                raise ValueError(f"weather.points entries are missing required keys: {missing}")

        output_columns = self.weather_output_columns()
        if not output_columns:
            raise ValueError("weather.enabled=true requires weather.output_columns to be configured.")

        future_exog = {str(column) for column in self.features.get("future_exog", [])}
        missing_outputs = [column for column in output_columns if column not in future_exog]
        if missing_outputs:
            raise ValueError(
                "weather.output_columns must also be listed in features.future_exog; "
                f"missing: {missing_outputs}"
            )

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
