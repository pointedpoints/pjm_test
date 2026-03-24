from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pjm_forecast.backtest import get_daily_split_days, run_rolling_backtest
from pjm_forecast.backtest.splits import load_split_boundaries
from pjm_forecast.config import load_config
from pjm_forecast.models import build_model
from pjm_forecast.paths import ensure_project_directories


def _load_best_params(config, hyperparameter_dir: Path) -> None:
    best_params_path = hyperparameter_dir / "nbeatsx_best_params.json"
    if not best_params_path.exists():
        return
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    config.models["nbeatsx"].update(best_params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    directories = ensure_project_directories(config)
    _load_best_params(config, directories["hyperparameter_dir"])

    feature_df = pd.read_parquet(directories["processed_data_dir"] / "feature_store.parquet")
    split_boundaries = load_split_boundaries(directories["processed_data_dir"] / "split_boundaries.json")
    forecast_days = get_daily_split_days(feature_df, split_boundaries, split_name=args.split)

    for model_name in config.backtest["benchmark_models"]:
        seeds = config.project["random_seeds"] if model_name == "nbeatsx" else [config.project["benchmark_seed"]]
        for seed in seeds:
            output_path = directories["prediction_dir"] / f"{model_name}_{args.split}_seed{seed}.parquet"
            predictions = run_rolling_backtest(
                config=config,
                feature_df=feature_df,
                split_name=args.split,
                forecast_days=forecast_days,
                model_builder=lambda model_name=model_name, seed=seed: build_model(
                    config,
                    model_name,
                    seed=seed,
                    hyperparameter_dir=directories["hyperparameter_dir"],
                ),
                model_name=model_name,
                seed=seed,
                output_path=output_path,
            )
            print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()

