from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecast.evaluation.median_bias_grid import evaluate_median_bias_grid


def _load_prediction(path: Path, *, score_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Prediction frame is empty: {path}")
    if score_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing {score_column!r}: {path}")
    return frame


def _parse_floats(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    if not parsed:
        raise ValueError("At least one numeric value is required.")
    return parsed


def _parse_interval_coverage_floors(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    floors: dict[str, float] = {}
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            key, raw_floor = item.split("=", maxsplit=1)
            floors[key.strip()] = float(raw_floor)
    return floors


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bounded median-bias postprocess grid on existing predictions.")
    parser.add_argument("--validation-prediction", required=True, help="Validation prediction parquet path.")
    parser.add_argument("--test-prediction", help="Optional test prediction parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument(
        "--max-abs-adjustments",
        nargs="+",
        default=["5", "10", "20"],
        help="Median-bias caps as separate values or comma-separated lists.",
    )
    parser.add_argument("--validation-holdout-days", type=int, default=91)
    parser.add_argument("--min-group-size", type=int, default=24)
    parser.add_argument("--group-by", default="hour", choices=["hour", "hour_x_regime"])
    parser.add_argument("--regime-score-column", default="spike_score")
    parser.add_argument("--regime-threshold", type=float, default=0.50)
    parser.add_argument(
        "--interval-coverage-floor",
        nargs="*",
        help="Optional CQR interval floors, e.g. 0.10-0.90=0.76 0.05-0.95=0.86.",
    )
    args = parser.parse_args()

    score_column = str(args.regime_score_column)
    validation_prediction = _load_prediction(Path(args.validation_prediction), score_column=score_column)
    test_prediction = None
    if args.test_prediction:
        test_prediction = _load_prediction(Path(args.test_prediction), score_column=score_column)

    result = evaluate_median_bias_grid(
        validation_prediction,
        test_frame=test_prediction,
        max_abs_adjustments=_parse_floats(list(args.max_abs_adjustments)),
        validation_holdout_days=int(args.validation_holdout_days),
        min_group_size=int(args.min_group_size),
        group_by=str(args.group_by),
        regime_score_column=score_column,
        regime_threshold=float(args.regime_threshold),
        interval_coverage_floors=_parse_interval_coverage_floors(args.interval_coverage_floor),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation_summary.to_csv(output_dir / "validation_holdout_summary.csv", index=False)
    if not result.test_summary.empty:
        result.test_summary.to_csv(output_dir / "test_summary.csv", index=False)


if __name__ == "__main__":
    main()
