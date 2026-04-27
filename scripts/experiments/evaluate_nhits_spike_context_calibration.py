from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Prediction frame is empty: {path}")
    if "spike_score" not in frame.columns:
        raise ValueError(f"Prediction frame is missing spike_score: {path}")
    return frame


def _split_validation_holdout(frame: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_day_series = pd.to_datetime(frame["ds"]).dt.floor("D")
    forecast_days = pd.Index(forecast_day_series.unique()).sort_values()
    if holdout_days <= 0 or holdout_days >= len(forecast_days):
        raise ValueError(f"holdout_days must be in [1, {len(forecast_days) - 1}]")

    calibration_days = set(forecast_days[:-holdout_days])
    evaluation_days = set(forecast_days[-holdout_days:])
    calibration_frame = frame.loc[forecast_day_series.isin(calibration_days)].copy()
    evaluation_frame = frame.loc[forecast_day_series.isin(evaluation_days)].copy()
    return calibration_frame, evaluation_frame


def _variant_definitions() -> list[tuple[str, dict[str, object]]]:
    return [
        ("raw_monotonic", {"monotonic": True, "calibration_frame": None}),
        (
            "hour_cqr",
            {
                "monotonic": True,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour",
                "calibration_min_group_size": 24,
            },
        ),
        (
            "hour_regime_cqr_t50",
            {
                "monotonic": True,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour_x_regime",
                "calibration_regime_score_column": "spike_score",
                "calibration_regime_threshold": 0.50,
                "calibration_min_group_size": 24,
            },
        ),
        (
            "hour_regime_cqr_t67",
            {
                "monotonic": True,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour_x_regime",
                "calibration_regime_score_column": "spike_score",
                "calibration_regime_threshold": 0.67,
                "calibration_min_group_size": 24,
            },
        ),
    ]


def _evaluate_variants(
    *,
    eval_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, params in _variant_definitions():
        run_params = dict(params)
        run_params.setdefault("calibration_frame", calibration_frame)
        processed = postprocess_quantile_predictions(eval_frame, **run_params)
        metrics = compute_metrics(processed)
        diagnostics = compute_quantile_diagnostics(processed)
        row: dict[str, object] = {"mode": mode, "variant": variant_name}
        row.update(metrics)
        row.update(diagnostics)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NHITS spike-context calibration variants.")
    parser.add_argument("--validation-prediction", required=True, help="Validation prediction parquet path.")
    parser.add_argument("--test-prediction", required=True, help="Test prediction parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument(
        "--validation-holdout-days",
        type=int,
        default=91,
        help="Number of validation forecast days reserved for evaluation after fitting calibration on the earlier days.",
    )
    args = parser.parse_args()

    validation_prediction = _load_prediction(Path(args.validation_prediction))
    test_prediction = _load_prediction(Path(args.test_prediction))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_calibration, validation_eval = _split_validation_holdout(
        validation_prediction,
        holdout_days=int(args.validation_holdout_days),
    )

    validation_summary = _evaluate_variants(
        eval_frame=validation_eval,
        calibration_frame=validation_calibration,
        mode="validation_holdout",
    ).sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)
    validation_summary.to_csv(output_dir / "validation_holdout_summary.csv", index=False)

    test_summary = _evaluate_variants(
        eval_frame=test_prediction,
        calibration_frame=validation_prediction,
        mode="test",
    ).sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)
    test_summary.to_csv(output_dir / "test_summary.csv", index=False)


if __name__ == "__main__":
    main()
