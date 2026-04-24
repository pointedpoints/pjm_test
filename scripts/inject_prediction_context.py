from __future__ import annotations

import argparse

from pjm_forecast.config import load_config
from pjm_forecast.prediction_context import inject_prediction_context_dir


def run_inject_prediction_context(
    config_path: str,
    *,
    source_prediction_dir: str,
    output_prediction_dir: str,
    context_columns: list[str],
    splits: list[str],
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    overwrite: bool = False,
    replace_existing_context: bool = False,
) -> None:
    config = load_config(config_path)
    results = inject_prediction_context_dir(
        config,
        source_prediction_dir=source_prediction_dir,
        output_prediction_dir=output_prediction_dir,
        context_columns=context_columns,
        splits=splits,
        models=models,
        seeds=seeds,
        overwrite=overwrite,
        replace_existing_context=replace_existing_context,
    )
    for result in results:
        columns = ", ".join(result.context_columns)
        print(
            f"Injected [{columns}] into {result.split} {result.model} seed={result.seed}: "
            f"{result.rows} rows -> {result.output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-prediction-dir", required=True)
    parser.add_argument("--output-prediction-dir", required=True)
    parser.add_argument("--context-columns", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["validation", "test"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-existing-context", action="store_true")
    args = parser.parse_args()
    run_inject_prediction_context(
        args.config,
        source_prediction_dir=args.source_prediction_dir,
        output_prediction_dir=args.output_prediction_dir,
        context_columns=args.context_columns,
        splits=args.splits,
        models=args.models,
        seeds=args.seeds,
        overwrite=args.overwrite,
        replace_existing_context=args.replace_existing_context,
    )


if __name__ == "__main__":
    main()
