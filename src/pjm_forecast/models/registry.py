from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pjm_forecast.config import ProjectConfig
from pjm_forecast.features import nbeatsx_futr_exog_columns, nbeatsx_hist_exog_columns

from .epftoolbox_wrappers import DNNModel, LEARModel
from .nbeatsx import NBEATSxModel
from .seasonal_naive import SeasonalNaiveModel


def build_model(
    config: ProjectConfig,
    model_name: str,
    seed: int | None = None,
    hyperparameter_dir: Path | None = None,
    disable_ensemble: bool = False,
):
    model_cfg = deepcopy(config.models[model_name])
    model_type = model_cfg.pop("type")

    if model_type == "seasonal_naive":
        return SeasonalNaiveModel(seasonal_lag_hours=config.backtest["seasonal_naive_lag_hours"])
    if model_type == "lear":
        return LEARModel(calibration_window_days=model_cfg["calibration_window_days"])
    if model_type == "dnn":
        return DNNModel(
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
    if model_type == "nbeatsx":
        if seed is not None:
            model_cfg["random_seed"] = seed
        model_cfg["freq"] = config.backtest["freq"]
        model_cfg["futr_exog_list"] = nbeatsx_futr_exog_columns(config)
        model_cfg["hist_exog_list"] = nbeatsx_hist_exog_columns(config)
        if disable_ensemble:
            model_cfg["ensemble_members"] = []
        return NBEATSxModel(**model_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")
