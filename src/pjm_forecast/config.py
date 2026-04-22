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
NEURALFORECAST_LOSS_NAMES = {"mae", "mqloss", "huber_mqloss"}
NEURALFORECAST_MODEL_TYPES = {"nbeatsx", "nhits"}
TUNING_METRICS = {"mae", "rmse", "smape", "pinball"}
QUANTILE_CALIBRATION_METHODS = {"cqr", "cqr_asymmetric"}
QUANTILE_CALIBRATION_GROUP_BY = {"hour"}
SCENARIO_COPULA_FAMILIES = {"gaussian", "student_t"}


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
            if kind in {"days_to_next_holiday", "days_since_prev_holiday", "days_to_year_end", "year_end_window", "pre_holiday_window"}:
                return set()
            if kind == "hour_indicator":
                return set()
            if kind == "prior_day_price_stat":
                return {str(item.get("source", self.target_column))}
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

    def expected_prediction_quantiles(self, model_name: str) -> list[float]:
        model_cfg = self.models.get(model_name)
        if not isinstance(model_cfg, dict):
            return []

        loss_name = str(model_cfg.get("loss_name", "mae")).lower()
        if loss_name not in {"mqloss", "huber_mqloss"}:
            return []

        quantiles = model_cfg.get("quantiles", [])
        if not quantiles:
            return []
        return sorted({float(value) for value in quantiles})

    def resolved_neuralforecast_scaler_strategy(self, model_name: str) -> str:
        model_cfg = self.models[model_name]
        pair = (
            str(model_cfg.get("target_transform", "identity")),
            str(model_cfg.get("exog_scaler", "identity")),
        )
        for strategy_name, strategy_pair in NBEATSX_SCALER_STRATEGIES.items():
            if pair == strategy_pair:
                return strategy_name
        raise ValueError(
            f"Unsupported {model_name} scaler contract: target_transform={pair[0]!r}, exog_scaler={pair[1]!r}."
        )

    def runtime_model_config(self, model_name: str) -> dict[str, Any]:
        model_cfg = dict(self.models[model_name])
        model_type = str(model_cfg.get("type", "")).lower()
        if model_type not in NEURALFORECAST_MODEL_TYPES:
            return model_cfg

        configured_h = int(model_cfg.get("h", self.prediction_horizon))
        if configured_h != self.prediction_horizon:
            raise ValueError(
                f"models.{model_name}.h={configured_h} must match backtest.horizon={self.prediction_horizon} in v1."
            )

        configured_freq = str(model_cfg.get("freq", self.prediction_freq))
        if configured_freq != self.prediction_freq:
            raise ValueError(
                f"models.{model_name}.freq={configured_freq!r} must match backtest.freq={self.prediction_freq!r} in v1."
            )

        strategy_name = self.resolved_neuralforecast_scaler_strategy(model_name)
        strategy_candidates = self.scaler_candidates()
        scaler_enabled = bool(self.features.get("scaler", {}).get("enabled", False))
        if scaler_enabled and strategy_name not in strategy_candidates:
            raise ValueError(
                f"{model_name} scaler strategy {strategy_name!r} must be listed in features.scaler.strategy_candidates."
            )
        if not scaler_enabled and strategy_name != "none":
            raise ValueError(f"features.scaler.enabled=false requires {model_name} to use the 'none' scaler strategy.")

        target_transform, exog_scaler = NBEATSX_SCALER_STRATEGIES[strategy_name]
        loss_name = str(model_cfg.get("loss_name", "mae")).lower()
        if loss_name not in NEURALFORECAST_LOSS_NAMES:
            raise ValueError(f"Unsupported {model_name} loss_name={loss_name!r}.")
        loss_delta = float(model_cfg.get("loss_delta", 1.0))
        if loss_delta <= 0.0:
            raise ValueError(f"models.{model_name}.loss_delta must be > 0.")
        quantiles = model_cfg.get("quantiles", [])
        if loss_name in {"mqloss", "huber_mqloss"}:
            if not quantiles:
                raise ValueError(f"models.{model_name}.quantiles must be configured when loss_name is quantile-based.")
            normalized_quantiles = sorted({float(value) for value in quantiles})
            if normalized_quantiles != [float(value) for value in quantiles]:
                raise ValueError(f"models.{model_name}.quantiles must be unique and sorted ascending.")
            invalid_quantiles = [value for value in normalized_quantiles if value <= 0.0 or value >= 1.0]
            if invalid_quantiles:
                raise ValueError(f"models.{model_name}.quantiles must be within (0, 1): {invalid_quantiles}")
            if not any(abs(value - 0.5) <= 1e-9 for value in normalized_quantiles):
                raise ValueError(f"models.{model_name}.quantiles must include 0.5 for p50-compatible evaluation.")
            quantile_weights = model_cfg.get("quantile_weights", [])
            if quantile_weights:
                normalized_weights = [float(value) for value in quantile_weights]
                if len(normalized_weights) != len(normalized_quantiles):
                    raise ValueError(f"models.{model_name}.quantile_weights must match quantiles length.")
                if any(value <= 0.0 for value in normalized_weights):
                    raise ValueError(f"models.{model_name}.quantile_weights must be strictly positive.")
                model_cfg["quantile_weights"] = normalized_weights
            else:
                model_cfg["quantile_weights"] = []
            quantile_deltas = model_cfg.get("quantile_deltas", [])
            if quantile_deltas:
                normalized_deltas = [float(value) for value in quantile_deltas]
                if len(normalized_deltas) != len(normalized_quantiles):
                    raise ValueError(f"models.{model_name}.quantile_deltas must match quantiles length.")
                if any(value <= 0.0 for value in normalized_deltas):
                    raise ValueError(f"models.{model_name}.quantile_deltas must be strictly positive.")
                model_cfg["quantile_deltas"] = normalized_deltas
            else:
                model_cfg["quantile_deltas"] = []
            monotonicity_penalty = float(model_cfg.get("monotonicity_penalty", 0.0))
            if monotonicity_penalty < 0.0:
                raise ValueError(f"models.{model_name}.monotonicity_penalty must be >= 0.")
            model_cfg["monotonicity_penalty"] = monotonicity_penalty
            model_cfg["quantiles"] = normalized_quantiles
        else:
            model_cfg["quantiles"] = []
            model_cfg["quantile_weights"] = []
            model_cfg["quantile_deltas"] = []
            model_cfg["monotonicity_penalty"] = 0.0
        model_cfg["loss_name"] = loss_name
        model_cfg["loss_delta"] = loss_delta
        model_cfg["h"] = self.prediction_horizon
        model_cfg["freq"] = self.prediction_freq
        model_cfg["target_transform"] = target_transform
        model_cfg["exog_scaler"] = exog_scaler
        return model_cfg

    def nbeatsx_runtime_config(self) -> dict[str, Any]:
        return self.runtime_model_config("nbeatsx")

    def nhits_runtime_config(self) -> dict[str, Any]:
        return self.runtime_model_config("nhits")

    def validate_runtime_contracts(self) -> None:
        if self.target_column != "y":
            raise ValueError(f"features.target_col={self.target_column!r} is unsupported; v1 requires the canonical target column 'y'.")
        invalid_candidates = [value for value in self.scaler_candidates() if value not in NBEATSX_SCALER_STRATEGIES]
        if invalid_candidates:
            raise ValueError(f"Unsupported features.scaler.strategy_candidates: {invalid_candidates}")
        tuning_metric = str(self.tuning.get("metric", "mae")).lower()
        if tuning_metric not in TUNING_METRICS:
            raise ValueError(f"Unsupported tuning.metric={tuning_metric!r}.")
        if self.dataset_source_type not in {"epftoolbox", "pjm_official", "official_weather_ready"}:
            raise ValueError(f"Unsupported dataset.source_type={self.dataset_source_type!r}.")
        self.validate_derived_feature_contracts()
        if self.weather_enabled:
            self.validate_weather_contracts()
        self.validate_quantile_postprocess_contracts()
        self.validate_scenario_evaluation_contracts()
        for model_name, model_cfg in self.models.items():
            if str(model_cfg.get("type", "")).lower() in NEURALFORECAST_MODEL_TYPES:
                self.runtime_model_config(model_name)

    def validate_derived_feature_contracts(self) -> None:
        derived_names: set[str] = set()
        available_names = {
            *self.features.get("future_exog", []),
            *self.features.get("lag_sources", []),
            *self.dataset.get("exogenous_columns", {}).keys(),
            *self.weather_output_columns(),
            *["is_weekend", "is_holiday"],
            self.target_column,
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
            elif kind == "days_to_next_holiday":
                pass
            elif kind == "days_since_prev_holiday":
                pass
            elif kind == "days_to_year_end":
                pass
            elif kind == "year_end_window":
                pass
            elif kind == "pre_holiday_window":
                max_days = int(item.get("max_days", 0))
                if max_days <= 0:
                    raise ValueError(f"derived_features pre_holiday_window requires max_days >= 1 for {name!r}.")
            elif kind == "hour_indicator":
                hour = int(item.get("hour", -1))
                if hour < 0 or hour > 23:
                    raise ValueError(f"derived_features hour_indicator hour={hour!r} is unsupported for {name!r}.")
            elif kind == "prior_day_price_stat":
                source = str(item.get("source", self.target_column))
                stat = str(item.get("stat", ""))
                if source not in available_feature_names:
                    raise ValueError(f"derived_features source {source!r} must already exist before deriving {name!r}.")
                if stat not in {"spread", "max_ramp", "max", "min", "mean"}:
                    raise ValueError(f"derived_features prior_day_price_stat stat={stat!r} is unsupported for {name!r}.")
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

    def validate_quantile_postprocess_contracts(self) -> None:
        postprocess = self.report.get("quantile_postprocess", {})
        if not postprocess:
            return
        monotonic = postprocess.get("monotonic")
        if monotonic is not None and not isinstance(monotonic, bool):
            raise ValueError("report.quantile_postprocess.monotonic must be a boolean when configured.")
        calibration = postprocess.get("calibration", {})
        if not calibration:
            return
        enabled = calibration.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("report.quantile_postprocess.calibration.enabled must be a boolean when configured.")
        source_split = calibration.get("source_split", "validation")
        if source_split not in {"validation"}:
            raise ValueError("report.quantile_postprocess.calibration.source_split currently only supports 'validation'.")
        method = calibration.get("method", "cqr")
        if method not in QUANTILE_CALIBRATION_METHODS:
            raise ValueError(f"Unsupported report.quantile_postprocess.calibration.method={method!r}.")
        group_by = calibration.get("group_by")
        if group_by is not None and group_by not in QUANTILE_CALIBRATION_GROUP_BY:
            raise ValueError(
                "report.quantile_postprocess.calibration.group_by must be one of "
                f"{sorted(QUANTILE_CALIBRATION_GROUP_BY)} when configured."
            )
        min_group_size = calibration.get("min_group_size")
        if min_group_size is not None and (not isinstance(min_group_size, int) or int(min_group_size) <= 0):
            raise ValueError("report.quantile_postprocess.calibration.min_group_size must be a positive integer.")

        interval_coverage_floors = calibration.get("interval_coverage_floors", {})
        if interval_coverage_floors and not isinstance(interval_coverage_floors, dict):
            raise ValueError("report.quantile_postprocess.calibration.interval_coverage_floors must be a mapping.")

        for model_name in self.models:
            quantiles = self.expected_prediction_quantiles(model_name)
            if not quantiles:
                continue
            if not _supports_cqr_quantile_pairs(quantiles):
                raise ValueError(
                    "report.quantile_postprocess.calibration requires symmetric quantile pairs for "
                    f"model {model_name!r}; received {quantiles}"
                )
            if not interval_coverage_floors:
                continue
            supported_pairs = {(lower, upper) for lower, upper in _quantile_pairs(quantiles)}
            for key, value in interval_coverage_floors.items():
                try:
                    lower_text, upper_text = str(key).split("-", maxsplit=1)
                    pair = (float(lower_text), float(upper_text))
                except Exception as exc:
                    raise ValueError(
                        "report.quantile_postprocess.calibration.interval_coverage_floors keys must look like "
                        "'0.10-0.90'."
                    ) from exc
                if pair not in supported_pairs:
                    raise ValueError(
                        "report.quantile_postprocess.calibration.interval_coverage_floors includes unsupported pair "
                        f"{key!r} for model {model_name!r}."
                    )
                numeric_value = float(value)
                if not 0.0 < numeric_value < 1.0:
                    raise ValueError(
                        "report.quantile_postprocess.calibration.interval_coverage_floors values must be in (0, 1)."
                    )

    def validate_scenario_evaluation_contracts(self) -> None:
        scenario_cfg = self.report.get("scenario_evaluation", {})
        if not scenario_cfg:
            return
        enabled = scenario_cfg.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("report.scenario_evaluation.enabled must be a boolean when configured.")
        source_split = scenario_cfg.get("source_split", "validation")
        if source_split not in {"validation"}:
            raise ValueError("report.scenario_evaluation.source_split currently only supports 'validation'.")
        family = str(scenario_cfg.get("copula_family", "student_t"))
        if family not in SCENARIO_COPULA_FAMILIES:
            raise ValueError(f"Unsupported report.scenario_evaluation.copula_family={family!r}.")
        n_samples = scenario_cfg.get("n_samples", 256)
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("report.scenario_evaluation.n_samples must be a positive integer.")
        random_seed = scenario_cfg.get("random_seed", 7)
        if not isinstance(random_seed, int):
            raise ValueError("report.scenario_evaluation.random_seed must be an integer.")
        dof_grid = scenario_cfg.get("dof_grid")
        if dof_grid is not None:
            if not isinstance(dof_grid, list) or not dof_grid:
                raise ValueError("report.scenario_evaluation.dof_grid must be a non-empty list when configured.")
            for value in dof_grid:
                numeric_value = float(value)
                if numeric_value <= 2.0:
                    raise ValueError("report.scenario_evaluation.dof_grid values must be > 2.")

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


def _supports_cqr_quantile_pairs(quantiles: list[float]) -> bool:
    normalized = sorted(float(value) for value in quantiles)
    for quantile in normalized:
        if quantile >= 0.5:
            continue
        if not any(abs(other - (1.0 - quantile)) <= 1e-9 for other in normalized):
            return False
    return True


def _quantile_pairs(quantiles: list[float]) -> list[tuple[float, float]]:
    normalized = sorted(float(value) for value in quantiles)
    pairs: list[tuple[float, float]] = []
    for quantile in normalized:
        if quantile >= 0.5:
            continue
        partner = 1.0 - quantile
        if any(abs(other - partner) <= 1e-9 for other in normalized):
            pairs.append((quantile, partner))
    return pairs
