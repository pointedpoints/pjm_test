from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from pjm_forecast.evaluation.event_risk_tail_overlay import (
    build_event_risk_tail_overlay_audit_artifacts,
    evaluate_event_risk_tail_overlay_grid,
)


def _load_prediction(path: Path, *, risk_score_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Prediction frame is empty: {path}")
    if risk_score_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing {risk_score_column!r}: {path}")
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


def _parse_strings(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    if not parsed:
        raise ValueError("At least one string value is required.")
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


def _spike_score_input_columns(config_path: str | None, risk_score_column: str) -> list[str]:
    if not config_path:
        return []
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    for feature in config.get("features", {}).get("derived_features", []) or []:
        if feature.get("kind") != "spike_score" or feature.get("name") != risk_score_column:
            continue
        return [str(item.get("source")) for item in feature.get("inputs", []) if item.get("source")]
    return []


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event-risk tail overlay on existing quantile predictions.")
    parser.add_argument("--config", help="Optional pipeline config used to audit spike_score input columns.")
    parser.add_argument("--validation-prediction", required=True, help="Validation prediction parquet path.")
    parser.add_argument("--test-prediction", help="Optional test prediction parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument("--risk-score-column", default="spike_score")
    parser.add_argument("--risk-aggregations", nargs="+", default=["mean"], help="Daily risk aggregations.")
    parser.add_argument("--risk-threshold-quantiles", nargs="+", default=["0.85", "0.90", "0.95"])
    parser.add_argument("--residual-quantiles", nargs="+", default=["0.50", "0.75", "0.90"])
    parser.add_argument("--max-uplifts", nargs="+", default=["25", "50", "100"])
    parser.add_argument("--target-quantiles", nargs="+", default=["0.99", "0.995"])
    parser.add_argument("--active-hour-sets", nargs="+", default=["all"], help="Overlay application hour sets.")
    parser.add_argument("--validation-holdout-days", type=int, default=91)
    parser.add_argument("--calibration-min-group-size", type=int, default=24)
    parser.add_argument("--regime-threshold", type=float, default=0.50)
    parser.add_argument("--selected-risk-aggregation", default="mean")
    parser.add_argument("--selected-risk-threshold-quantile", type=float, default=0.90)
    parser.add_argument("--selected-residual-quantile", type=float, default=1.00)
    parser.add_argument("--selected-max-uplift", type=float, default=50.0)
    parser.add_argument("--selected-active-hour-set", default="all")
    parser.add_argument(
        "--interval-coverage-floor",
        nargs="*",
        help="Optional CQR interval floors, e.g. 0.10-0.90=0.76 0.05-0.95=0.86.",
    )
    args = parser.parse_args()

    risk_score_column = str(args.risk_score_column)
    validation_prediction = _load_prediction(Path(args.validation_prediction), risk_score_column=risk_score_column)
    test_prediction = None
    if args.test_prediction:
        test_prediction = _load_prediction(Path(args.test_prediction), risk_score_column=risk_score_column)

    result = evaluate_event_risk_tail_overlay_grid(
        validation_prediction,
        test_frame=test_prediction,
        validation_holdout_days=int(args.validation_holdout_days),
        risk_score_column=risk_score_column,
        risk_aggregations=_parse_strings(list(args.risk_aggregations)),
        risk_threshold_quantiles=_parse_floats(list(args.risk_threshold_quantiles)),
        residual_quantiles=_parse_floats(list(args.residual_quantiles)),
        max_uplifts=_parse_floats(list(args.max_uplifts)),
        target_quantiles=_parse_floats(list(args.target_quantiles)),
        calibration_min_group_size=int(args.calibration_min_group_size),
        interval_coverage_floors=_parse_interval_coverage_floors(args.interval_coverage_floor),
        regime_threshold=float(args.regime_threshold),
        active_hour_sets=_parse_strings(list(args.active_hour_sets)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation_summary.to_csv(output_dir / "validation_holdout_summary.csv", index=False)
    if not result.test_summary.empty:
        result.test_summary.to_csv(output_dir / "test_summary.csv", index=False)

    audit = build_event_risk_tail_overlay_audit_artifacts(
        validation_prediction,
        test_frame=test_prediction,
        validation_holdout_days=int(args.validation_holdout_days),
        risk_score_column=risk_score_column,
        risk_aggregation=str(args.selected_risk_aggregation),
        risk_threshold_quantile=float(args.selected_risk_threshold_quantile),
        residual_quantile=float(args.selected_residual_quantile),
        max_uplift=float(args.selected_max_uplift),
        target_quantiles=_parse_floats(list(args.target_quantiles)),
        calibration_min_group_size=int(args.calibration_min_group_size),
        interval_coverage_floors=_parse_interval_coverage_floors(args.interval_coverage_floor),
        regime_threshold=float(args.regime_threshold),
        risk_score_input_columns=_spike_score_input_columns(args.config, risk_score_column),
        active_hour_set=str(args.selected_active_hour_set),
        conservative_active_hour_sets=_parse_strings(list(args.active_hour_sets)),
    )
    _write_json(output_dir / "overlay_implementation_audit.json", audit.implementation_audit)
    _write_json(output_dir / "spike_score_audit.json", audit.spike_score_audit)
    audit.active_day_diagnostics.to_csv(output_dir / "active_day_diagnostics.csv", index=False)
    audit.active_days_by_month.to_csv(output_dir / "active_days_by_month.csv", index=False)
    audit.width_by_regime.to_csv(output_dir / "width_by_regime.csv", index=False)
    audit.pinball_by_quantile.to_csv(output_dir / "pinball_by_quantile.csv", index=False)
    audit.conservative_variant_grid.to_csv(output_dir / "conservative_variant_grid.csv", index=False)
    audit.daily_max_gap_detail.to_csv(output_dir / "daily_max_gap_detail.csv", index=False)
    audit.event_day_before_after.to_csv(output_dir / "event_day_before_after.csv", index=False)


if __name__ == "__main__":
    main()
