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
        removed_derived_names = set(weather_columns)

        def _derived_dependencies(item: dict[str, Any]) -> set[str]:
            kind = str(item["kind"])
            if kind == "degree_day":
                return {str(item["source"])}
            if kind == "multiply":
                return {str(item["left"]), str(item["right"])}
            if kind == "sum":
                return {str(value) for value in item.get("inputs", [])}
            if kind == "hour_indicator":
                return set()
            return set()

        changed = True
        while changed:
            changed = False
            for item in raw_copy["features"].get("derived_ramps", []):
                source = str(item["source"])
                name = str(item.get("name", f"{source}_delta_{int(item.get('lag', 24))}"))
                if source in removed_derived_names and name not in removed_derived_names:
                    removed_derived_names.add(name)
                    changed = True
            for item in raw_copy["features"].get("derived_features", []):
                name = str(item["name"])
                if _derived_dependencies(item) & removed_derived_names and name not in removed_derived_names:
                    removed_derived_names.add(name)
                    changed = True

        raw_copy["features"]["future_exog"] = [
            column for column in raw_copy["features"].get("future_exog", [])
            if str(column) not in weather_columns and str(column) not in removed_derived_names
        ]
        if "lag_sources" in raw_copy["features"]:
            raw_copy["features"]["lag_sources"] = [
                column for column in raw_copy["features"]["lag_sources"]
                if str(column) not in weather_columns and str(column) not in removed_derived_names
            ]
        if "derived_ramps" in raw_copy["features"]:
            raw_copy["features"]["derived_ramps"] = [
                item for item in raw_copy["features"]["derived_ramps"]
                if str(item["source"]) not in weather_columns
            ]
        if "derived_features" in raw_copy["features"]:
            raw_copy["features"]["derived_features"] = [
                item for item in raw_copy["features"]["derived_features"]
                if str(item["name"]) not in removed_derived_names
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
        self.validate_derived_feature_contracts()
        if self.weather_enabled:
            self.validate_weather_contracts()
        if "nbeatsx" in self.models:
            self.nbeatsx_runtime_config()

    def validate_derived_feature_contracts(self) -> None:
        derived_names: set[str] = set()
        available_names = {
            *self.features.get("future_exog", []),
            *self.features.get("lag_sources", []),
            *self.dataset.get("exogenous_columns", {}).keys(),
            *self.weather_output_columns(),
            *["is_weekend", "is_holiday"],
        }
        for item in self.features.get("derived_ramps", []):
            source = str(item["source"])
            if source not in available_names:
                raise ValueError(f"derived_ramps source {source!r} must be present in future_exog or lag_sources.")
            derived_names.add(str(item.get("name", f"{source}_delta_{int(item.get('lag', 24))}")))

        available_feature_names = available_names | derived_names
        for item in self.features.get("derived_features", []):
            kind = str(item.get("kind", ""))
            name = str(item.get("name", ""))
            if not name:
                raise ValueError("derived_features entries require a non-empty name.")
            if kind == "degree_day":
                source = str(item.get("source", ""))
                mode = str(item.get("mode", ""))
                if source not in available_feature_names:
                    raise ValueError(f"derived_features source {source!r} must already exist before deriving {name!r}.")
                if mode not in {"heating", "cooling"}:
                    raise ValueError(f"derived_features mode={mode!r} is unsupported for {name!r}.")
                if "base" not in item:
                    raise ValueError(f"derived_features {name!r} requires a base threshold.")
            elif kind == "multiply":
                left = str(item.get("left", ""))
                right = str(item.get("right", ""))
                missing = [value for value in [left, right] if value not in available_feature_names]
                if missing:
                    raise ValueError(f"derived_features multiply inputs are unavailable for {name!r}: {missing}")
            elif kind == "sum":
                inputs = [str(value) for value in item.get("inputs", [])]
                if not inputs:
                    raise ValueError(f"derived_features sum requires at least one input for {name!r}.")
                missing = [value for value in inputs if value not in available_feature_names]
                if missing:
                    raise ValueError(f"derived_features sum inputs are unavailable for {name!r}: {missing}")
            elif kind == "hour_indicator":
                hour = int(item.get("hour", -1))
                if hour < 0 or hour > 23:
                    raise ValueError(f"derived_features hour_indicator hour={hour!r} is unsupported for {name!r}.")
            else:
                raise ValueError(f"Unsupported derived_features kind={kind!r}.")
            available_feature_names.add(name)

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

        required_weather_outputs = {
            str(column) for column in self.features.get("future_exog", []) if str(column).startswith("weather_")
        }
        for item in self.features.get("derived_ramps", []):
            source = str(item["source"])
            if source.startswith("weather_"):
                required_weather_outputs.add(source)
        for item in self.features.get("derived_features", []):
            kind = str(item.get("kind", ""))
            if kind == "degree_day":
                source = str(item.get("source", ""))
                if source.startswith("weather_"):
                    required_weather_outputs.add(source)
            elif kind == "multiply":
                for side in [str(item.get("left", "")), str(item.get("right", ""))]:
                    if side.startswith("weather_"):
                        required_weather_outputs.add(side)

        missing_outputs = [column for column in required_weather_outputs if column not in output_columns]
        if missing_outputs:
            raise ValueError(
                "weather.output_columns must cover all weather future_exog columns and derived weather dependencies; "
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
