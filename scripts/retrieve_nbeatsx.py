from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pjm_forecast.backtest import run_rolling_backtest
from pjm_forecast.backtest.splits import get_daily_split_days, load_split_boundaries
from pjm_forecast.config import load_config
from pjm_forecast.models import build_model
from pjm_forecast.paths import ensure_project_directories
from pjm_forecast.retrieval import RetrievalConfig, apply_residual_retrieval, tune_retrieval_params


def _prediction_path(prediction_dir: Path, model_name: str, split_name: str, seed: int) -> Path:
    return prediction_dir / f"{model_name}_{split_name}_seed{seed}.parquet"


def _load_or_backtest_prediction(
    config,
    directories: dict[str, Path],
    feature_df: pd.DataFrame,
    forecast_days: list[pd.Timestamp],
    split_name: str,
    seed: int,
    model_name: str = "nbeatsx",
    output_suffix: str | None = None,
) -> pd.DataFrame:
    output_name = f"{model_name}_{split_name}_seed{seed}.parquet" if output_suffix is None else output_suffix
    output_path = directories["prediction_dir"] / output_name
    if output_path.exists():
        return pd.read_parquet(output_path)

    predictions = run_rolling_backtest(
        config=config,
        feature_df=feature_df,
        split_name=split_name,
        forecast_days=forecast_days,
        model_builder=lambda: build_model(config, model_name, seed=seed),
        model_name=model_name,
        seed=seed,
        output_path=output_path,
    )
    return predictions


def _custom_daily_days(feature_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.Index(feature_df["ds"].dt.normalize().drop_duplicates().sort_values())
    mask = (days >= start.normalize()) & (days <= end.normalize())
    return list(days[mask])


def _retrieval_config(config) -> RetrievalConfig:
    retrieval = config.raw["retrieval"]
    return RetrievalConfig(
        history_days=retrieval["history_days"],
        price_weight=float(retrieval["weights"]["price"]),
        load_weight=float(retrieval["weights"]["load"]),
        calendar_weight=float(retrieval["weights"]["calendar"]),
        top_k=int(retrieval["top_k"]),
        min_gap_days=int(retrieval["min_gap_days"]),
        residual_clip_quantile=float(retrieval["residual_clip_quantile"]),
    )


def run_retrieve_nbeatsx(config_path: str, split: str = "test") -> None:
    config = load_config(config_path)
    directories = ensure_project_directories(config)
    feature_df = pd.read_parquet(directories["processed_data_dir"] / "feature_store.parquet")
    split_boundaries = load_split_boundaries(directories["processed_data_dir"] / "split_boundaries.json")
    benchmark_seed = config.project["benchmark_seed"]
    retrieval = config.raw["retrieval"]

    validation_days = get_daily_split_days(feature_df, split_boundaries, split_name="validation")
    validation_predictions = _load_or_backtest_prediction(
        config=config,
        directories=directories,
        feature_df=feature_df,
        forecast_days=validation_days,
        split_name="validation",
        seed=benchmark_seed,
    )

    warmup_end = split_boundaries["validation_start"] - pd.Timedelta(days=1)
    warmup_start = split_boundaries["validation_start"] - pd.Timedelta(days=retrieval["warmup_days"])
    warmup_days = _custom_daily_days(feature_df, start=warmup_start, end=warmup_end)
    warmup_predictions = _load_or_backtest_prediction(
        config=config,
        directories=directories,
        feature_df=feature_df,
        forecast_days=warmup_days,
        split_name="retrieval_warmup",
        seed=benchmark_seed,
        output_suffix=f"nbeatsx_retrieval_warmup_seed{benchmark_seed}.parquet",
    )

    retrieval_cfg = _retrieval_config(config)
    best_params_path = directories["hyperparameter_dir"] / "nbeatsx_rag_best_params.json"
    best_params, tuning_scores = tune_retrieval_params(
        feature_df=feature_df,
        validation_predictions=validation_predictions,
        initial_memory_predictions=warmup_predictions,
        config=retrieval_cfg,
        alpha_grid=[float(value) for value in retrieval["alpha_grid"]],
        tau_grid=[float(value) for value in retrieval["tau_grid"]],
        volatility_quantile_grid=[
            None if value is None else float(value) for value in retrieval["volatility_quantile_grid"]
        ],
    )
    best_params_path.write_text(
        json.dumps(
            {
                "alpha": best_params.alpha,
                "tau": best_params.tau,
                "predicted_volatility_threshold": best_params.predicted_volatility_threshold,
                "scores": tuning_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_rag = apply_residual_retrieval(
        feature_df=feature_df,
        base_predictions=validation_predictions,
        initial_memory_predictions=warmup_predictions,
        config=retrieval_cfg,
        params=best_params,
    )
    validation_rag.to_parquet(_prediction_path(directories["prediction_dir"], "nbeatsx_rag", "validation", benchmark_seed), index=False)

    if split == "validation":
        return

    test_days = get_daily_split_days(feature_df, split_boundaries, split_name="test")
    test_predictions = _load_or_backtest_prediction(
        config=config,
        directories=directories,
        feature_df=feature_df,
        forecast_days=test_days,
        split_name="test",
        seed=benchmark_seed,
    )

    initial_test_memory = pd.concat([warmup_predictions, validation_predictions], axis=0, ignore_index=True)
    test_rag = apply_residual_retrieval(
        feature_df=feature_df,
        base_predictions=test_predictions,
        initial_memory_predictions=initial_test_memory,
        config=retrieval_cfg,
        params=best_params,
    )
    test_rag.to_parquet(_prediction_path(directories["prediction_dir"], "nbeatsx_rag", "test", benchmark_seed), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_retrieve_nbeatsx(args.config, split=args.split)


if __name__ == "__main__":
    main()
