from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema
from pjm_forecast.spike_filter import SpikeFilterConfig

from .epftoolbox_wrappers import DNNModel, LEARModel
from .nhits import NHITSModel
from .nbeatsx import NBEATSxModel
from .patchtst import PatchTSTModel
from .seasonal_naive import SeasonalNaiveModel
from .target_filter import SpikeFilteredTargetModel
from .tide import TiDEModel
from .tree_quantile import LightGBMQuantileModel, XGBoostQuantileModel


def build_model(
    config: ProjectConfig,
    model_name: str,
    seed: int | None = None,
    hyperparameter_dir: Path | None = None,
    disable_ensemble: bool = False,
):
    model_type = deepcopy(config.models[model_name]).get("type")
    if model_type == "nbeatsx":
        model_cfg = config.runtime_model_config(model_name)
    elif model_type == "nhits":
        model_cfg = config.runtime_model_config(model_name)
    else:
        model_cfg = deepcopy(config.models[model_name])
    target_filter_cfg = model_cfg.pop("target_filter", {}) or {}
    model_type = model_cfg.pop("type")
    schema = FeatureSchema(config)

    if model_type == "seasonal_naive":
        model = SeasonalNaiveModel(seasonal_lag_hours=config.backtest["seasonal_naive_lag_hours"])
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "lear":
        model = LEARModel(calibration_window_days=model_cfg["calibration_window_days"])
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "dnn":
        model = DNNModel(
            experiment_id=model_cfg["experiment_id"],
            hyperparameter_dir=str(hyperparameter_dir or config.resolve_path(config.project["directories"]["hyperparameter_dir"])),
            dataset_dir=str(config.resolve_path(config.project["directories"]["raw_data_dir"])),
            nlayers=model_cfg["nlayers"],
            dataset=config.dataset["dataset_name"],
            years_test=config.backtest["years_test"],
            shuffle_train=model_cfg["shuffle_train"],
            data_augmentation=model_cfg["data_augmentation"],
            calibration_window_years=model_cfg["calibration_window_years"],
            auto_generate_hyperparameters=model_cfg["auto_generate_hyperparameters"],
            hyperopt_max_evals=model_cfg["hyperopt_max_evals"],
        )
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "lightgbm_quantile":
        model = LightGBMQuantileModel(
            feature_columns=[column for column in schema.feature_columns() if column not in {"unique_id", "ds", config.target_column}],
            quantiles=[float(value) for value in model_cfg.get("quantiles", [])],
            random_seed=seed if seed is not None else config.project["benchmark_seed"],
            model_params={key: value for key, value in model_cfg.items() if key not in {"type", "loss_name", "quantiles"}},
        )
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "xgboost_quantile":
        model = XGBoostQuantileModel(
            feature_columns=[column for column in schema.feature_columns() if column not in {"unique_id", "ds", config.target_column}],
            quantiles=[float(value) for value in model_cfg.get("quantiles", [])],
            random_seed=seed if seed is not None else config.project["benchmark_seed"],
            model_params={key: value for key, value in model_cfg.items() if key not in {"type", "loss_name", "quantiles"}},
        )
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "nbeatsx":
        contract = schema.nbeatsx_exogenous_contract()
        if seed is not None:
            model_cfg["random_seed"] = seed
        model_cfg["futr_exog_list"] = contract.futr_exog_columns
        model_cfg["hist_exog_list"] = contract.hist_exog_columns
        model_cfg["protected_exog_columns"] = contract.protected_exog_columns
        if disable_ensemble:
            model_cfg["ensemble_members"] = []
        model = NBEATSxModel(**model_cfg)
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "nhits":
        contract = schema.nbeatsx_exogenous_contract()
        if seed is not None:
            model_cfg["random_seed"] = seed
        model_cfg["futr_exog_list"] = contract.futr_exog_columns
        model_cfg["hist_exog_list"] = contract.hist_exog_columns
        model_cfg["protected_exog_columns"] = contract.protected_exog_columns
        if disable_ensemble:
            model_cfg["ensemble_members"] = []
        model = NHITSModel(**model_cfg)
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "tide":
        contract = schema.nbeatsx_exogenous_contract()
        if seed is not None:
            model_cfg["random_seed"] = seed
        model_cfg["futr_exog_list"] = contract.futr_exog_columns
        model_cfg["hist_exog_list"] = contract.hist_exog_columns
        model_cfg["protected_exog_columns"] = contract.protected_exog_columns
        if disable_ensemble:
            model_cfg["ensemble_members"] = []
        model = TiDEModel(**model_cfg)
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    if model_type == "patchtst":
        contract = schema.nbeatsx_exogenous_contract()
        if seed is not None:
            model_cfg["random_seed"] = seed
        model_cfg["futr_exog_list"] = contract.futr_exog_columns
        model_cfg["hist_exog_list"] = contract.hist_exog_columns
        model_cfg["protected_exog_columns"] = contract.protected_exog_columns
        if disable_ensemble:
            model_cfg["ensemble_members"] = []
        model = PatchTSTModel(**model_cfg)
        return _maybe_wrap_target_filter(model, target_filter_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")


def _maybe_wrap_target_filter(model, target_filter_cfg: dict):
    if not bool(target_filter_cfg.get("enabled", False)):
        return model
    filter_config = SpikeFilterConfig(
        window_observations=int(target_filter_cfg.get("window_observations", 365)),
        min_history=int(target_filter_cfg.get("min_history", 60)),
        quantile=float(target_filter_cfg.get("quantile", 0.95)),
        fallback_quantile=float(target_filter_cfg.get("fallback_quantile", 0.975)),
        iqr_multiplier=float(target_filter_cfg.get("iqr_multiplier", 3.0)),
    )
    return SpikeFilteredTargetModel(base_model=model, filter_config=filter_config)
