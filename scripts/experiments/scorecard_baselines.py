from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


DEFAULT_BASELINE_MODELS = [
    "seasonal_naive",
    "lear",
    "lightgbm_quantile",
    "xgboost_quantile",
    "nhits_tail_grid_weighted_main",
]


def baseline_model_names(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_BASELINE_MODELS)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_scorecard_baselines(config_path: str, split: str, models: list[str], run_backtest: bool) -> None:
    workspace = Workspace.open(config_path)
    configured_models = [model for model in models if model in workspace.config.models]
    missing_models = [model for model in models if model not in workspace.config.models]
    if missing_models:
        print(f"Skipping models not present in config: {', '.join(missing_models)}")

    if run_backtest:
        if not configured_models:
            raise ValueError("No requested baseline models are present in the config.")
        workspace.backtest(split=split, model_names=configured_models)
    workspace.evaluate(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--models", default=None, help="Comma-separated model names. Defaults to COMED scoreboard baselines.")
    parser.add_argument("--run-backtest", action="store_true", help="Run rolling backtest before evaluating existing predictions.")
    args = parser.parse_args()
    run_scorecard_baselines(
        config_path=args.config,
        split=args.split,
        models=baseline_model_names(args.models),
        run_backtest=bool(args.run_backtest),
    )


if __name__ == "__main__":
    main()
