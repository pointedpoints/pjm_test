from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecast.evaluation.hour_x_regime_grid import evaluate_hour_x_regime_threshold_grid


def _load_prediction(path: Path, *, score_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Prediction frame is empty: {path}")
    if score_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing {score_column!r}: {path}")
    return frame


def _parse_thresholds(values: list[str]) -> list[float]:
    thresholds: list[float] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hour_x_regime CQR threshold grid on existing predictions.")
    parser.add_argument("--validation-prediction", required=True, help="Validation prediction parquet path.")
    parser.add_argument("--test-prediction", help="Optional test prediction parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=["0.50", "0.67"],
        help="Regime thresholds as separate values or comma-separated lists.",
    )
    parser.add_argument("--validation-holdout-days", type=int, default=91)
    parser.add_argument("--min-group-size", type=int, default=24)
    parser.add_argument("--regime-score-column", default="spike_score")
    args = parser.parse_args()

    score_column = str(args.regime_score_column)
    validation_prediction = _load_prediction(Path(args.validation_prediction), score_column=score_column)
    test_prediction = None
    if args.test_prediction:
        test_prediction = _load_prediction(Path(args.test_prediction), score_column=score_column)

    result = evaluate_hour_x_regime_threshold_grid(
        validation_prediction,
        test_frame=test_prediction,
        thresholds=_parse_thresholds(list(args.thresholds)),
        validation_holdout_days=int(args.validation_holdout_days),
        min_group_size=int(args.min_group_size),
        regime_score_column=score_column,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation_summary.to_csv(output_dir / "validation_holdout_summary.csv", index=False)
    if not result.test_summary.empty:
        result.test_summary.to_csv(output_dir / "test_summary.csv", index=False)


if __name__ == "__main__":
    main()
