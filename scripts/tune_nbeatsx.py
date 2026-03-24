from __future__ import annotations

import argparse
import json

import optuna
import pandas as pd

from pjm_forecast.backtest import get_daily_split_days, run_rolling_backtest
from pjm_forecast.backtest.splits import load_split_boundaries
from pjm_forecast.config import load_config
from pjm_forecast.evaluation.metrics import compute_metrics
from pjm_forecast.models import build_model
from pjm_forecast.paths import ensure_project_directories


MLP_UNIT_SEARCH_SPACE = {
    "256x256": [[256, 256], [256, 256]],
    "512x512": [[512, 512], [512, 512]],
    "512x256": [[512, 256], [256, 256]],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    directories = ensure_project_directories(config)
    feature_df = pd.read_parquet(directories["processed_data_dir"] / "feature_store.parquet")
    split_boundaries = load_split_boundaries(directories["processed_data_dir"] / "split_boundaries.json")
    validation_days = get_daily_split_days(feature_df, split_boundaries, split_name="validation")
    tuning_cfg = config.tuning

    def objective(trial: optuna.Trial) -> float:
        config.models["nbeatsx"]["input_size"] = trial.suggest_categorical(
            "input_size",
            tuning_cfg["search_space"]["input_size"],
        )
        config.models["nbeatsx"]["learning_rate"] = trial.suggest_float(
            "learning_rate",
            tuning_cfg["search_space"]["learning_rate"][0],
            tuning_cfg["search_space"]["learning_rate"][1],
            log=True,
        )
        config.models["nbeatsx"]["batch_size"] = trial.suggest_categorical(
            "batch_size",
            tuning_cfg["search_space"]["batch_size"],
        )
        config.models["nbeatsx"]["max_steps"] = trial.suggest_int(
            "max_steps",
            tuning_cfg["search_space"]["max_steps"][0],
            tuning_cfg["search_space"]["max_steps"][1],
        )
        config.models["nbeatsx"]["dropout_prob_theta"] = trial.suggest_float(
            "dropout_prob_theta",
            tuning_cfg["search_space"]["dropout"][0],
            tuning_cfg["search_space"]["dropout"][1],
        )
        mlp_units_key = trial.suggest_categorical("mlp_units", list(MLP_UNIT_SEARCH_SPACE))
        config.models["nbeatsx"]["mlp_units"] = MLP_UNIT_SEARCH_SPACE[mlp_units_key]

        predictions = run_rolling_backtest(
            config=config,
            feature_df=feature_df,
            split_name="validation",
            forecast_days=validation_days,
            model_builder=lambda: build_model(config, "nbeatsx", seed=config.project["benchmark_seed"]),
            model_name="nbeatsx",
            seed=config.project["benchmark_seed"],
        )
        return compute_metrics(predictions)["mae"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=tuning_cfg["n_trials"])

    output_path = directories["hyperparameter_dir"] / "nbeatsx_best_params.json"
    output_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    print(f"Saved best params to {output_path}")


if __name__ == "__main__":
    main()
