from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pjm_forecast.evaluation.quality_gate import QualityDecision, QualityMetrics, evaluate_quality_gate, reference_result


@dataclass(frozen=True)
class RegistrySpec:
    label: str
    config_path: str
    metrics_dir: Path
    splits: tuple[str, ...]
    decision_override: QualityDecision | None = None
    reason_override: str | None = None


DEFAULT_SPECS = [
    RegistrySpec(
        label="nhits_tail_grid_weighted_long",
        config_path="configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_long.yaml",
        metrics_dir=Path("artifacts_tmp/nhits_tail_grid_weighted_long/metrics"),
        splits=("validation",),
        decision_override=QualityDecision.REFERENCE,
        reason_override="validation reference baseline",
    ),
    RegistrySpec(
        label="linear_tail",
        config_path="configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_linear_tail.yaml",
        metrics_dir=Path("artifacts_tmp/nhits_tail_grid_weighted_long_linear_tail/metrics"),
        splits=("validation", "test"),
        decision_override=QualityDecision.REFERENCE,
        reason_override="conservative tail reference",
    ),
    RegistrySpec(
        label="spike_context",
        config_path="configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_spike_context_hour_regime.yaml",
        metrics_dir=Path("artifacts_tmp/nhits_tail_grid_weighted_long_spike_context/metrics"),
        splits=("test",),
        decision_override=QualityDecision.TAIL_ONLY,
        reason_override="tail candidate with strong q99 gains but wide intervals",
    ),
    RegistrySpec(
        label="nhits_q50w150",
        config_path="configs/experiments/pjm_current_test_nhits_q50w150.yaml",
        metrics_dir=Path("artifacts_tmp/nhits_q50_weight_grid/metrics"),
        splits=("test",),
        decision_override=QualityDecision.TAIL_ONLY,
        reason_override="tail candidate with q99 gains but wide intervals",
    ),
    RegistrySpec(
        label="future_price_lag_168",
        config_path="configs/experiments/pjm_current_p50_futr_lag168.yaml",
        metrics_dir=Path("artifacts_phase2/p50_futr_lag168/metrics"),
        splits=("validation",),
        decision_override=QualityDecision.CONTEXT_ONLY,
        reason_override="mixed validation result; keep as context candidate only",
    ),
    RegistrySpec(
        label="future_price_lag_168_336",
        config_path="configs/experiments/pjm_current_p50_futr_lag168_336.yaml",
        metrics_dir=Path("artifacts_phase2/p50_futr_lag168_336/metrics"),
        splits=("validation",),
        decision_override=QualityDecision.REJECT,
        reason_override="weekly lag expansion failed validation",
    ),
    RegistrySpec(
        label="prior_day_price_state",
        config_path="configs/experiments/pjm_current_p50_price_state.yaml",
        metrics_dir=Path("artifacts_phase2/p50_price_state/metrics"),
        splits=("validation", "test"),
        decision_override=QualityDecision.CONTEXT_ONLY,
        reason_override="validation gain failed to generalize to test; context only",
    ),
]


def build_quality_flow_registry(root: Path, specs: list[RegistrySpec] | None = None) -> pd.DataFrame:
    specs = specs or DEFAULT_SPECS
    rows = []
    split_baselines: dict[str, QualityMetrics] = {}
    staged_rows: list[tuple[RegistrySpec, str, dict[str, object]]] = []
    for spec in specs:
        for split in spec.splits:
            row = _load_registry_row(root=root, spec=spec, split=split)
            if row is None:
                continue
            staged_rows.append((spec, split, row))
            if spec.decision_override == QualityDecision.REFERENCE and split not in split_baselines:
                split_baselines[split] = _quality_metrics_from_row(row)

    for spec, split, row in staged_rows:
        baseline = split_baselines.get(split)
        if spec.decision_override is not None:
            gate = reference_result(spec.reason_override or "fixed decision") if spec.decision_override == QualityDecision.REFERENCE else None
            if gate is None and baseline is not None:
                gate = evaluate_quality_gate(
                    baseline=baseline,
                    candidate=_quality_metrics_from_row(row),
                    validation_direction_consistent=spec.label != "prior_day_price_state" or split != "test",
                    main_model_candidate=spec.decision_override == QualityDecision.PROMOTE,
                )
            row["decision"] = spec.decision_override.value
            row["reason"] = spec.reason_override or (gate.reason if gate is not None else "fixed decision")
        elif baseline is not None:
            gate = evaluate_quality_gate(baseline=baseline, candidate=_quality_metrics_from_row(row))
            row["decision"] = gate.decision.value
            row["reason"] = gate.reason
        else:
            gate = reference_result("no baseline available for split")
            row["decision"] = gate.decision.value
            row["reason"] = gate.reason

        if baseline is not None:
            tradeoff = evaluate_quality_gate(
                baseline=baseline,
                candidate=_quality_metrics_from_row(row),
                main_model_candidate=False,
            )
            row["width98_ratio"] = tradeoff.width98_ratio
            row["tail_gain"] = tradeoff.tail_gain
            row["tail_gain_per_width"] = tradeoff.tail_gain_per_width
            row["pinball_delta"] = tradeoff.pinball_delta
            row["mae_delta"] = tradeoff.mae_delta
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No benchmark metrics were found for the configured quality flow registry specs.")
    return pd.DataFrame(rows).sort_values(["split", "decision", "run_name"]).reset_index(drop=True)


def _load_registry_row(*, root: Path, spec: RegistrySpec, split: str) -> dict[str, object] | None:
    metrics_dir = root / spec.metrics_dir
    metrics_path = metrics_dir / f"{split}_metrics.csv"
    diagnostics_path = metrics_dir / f"{split}_quantile_diagnostics.csv"
    regime_path = metrics_dir / f"{split}_regime_metrics.csv"
    if not metrics_path.exists() or not diagnostics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    diagnostics = pd.read_csv(diagnostics_path).iloc[0].to_dict()
    row: dict[str, object] = {
        "run_name": spec.label,
        "artifact_run": metrics.get("run"),
        "config_path": spec.config_path,
        "split": split,
        "mae": _coalesce(metrics.get("mae"), diagnostics.get("post_q50_mae")),
        "pinball": _coalesce(metrics.get("pinball"), diagnostics.get("post_pinball")),
        "q99_exceedance_rate": diagnostics.get("post_q99_exceedance_rate"),
        "q99_excess_mean": diagnostics.get("post_q99_excess_mean"),
        "worst_q99_underprediction": diagnostics.get("post_worst_q99_underprediction"),
        "width_98": diagnostics.get("post_width_98"),
        "daily_max_q99_gap": diagnostics.get("post_daily_max_q99_gap_max"),
    }
    row.update(_load_regime_columns(regime_path))
    return row


def _load_regime_columns(path: Path) -> dict[str, float]:
    result = {
        "normal_p50_mae": float("nan"),
        "extreme_p50_mae": float("nan"),
        "daily_max_q99_gap": float("nan"),
    }
    if not path.exists():
        return result
    frame = pd.read_csv(path)
    normal = frame.loc[frame["regime"].eq("normal")]
    extreme = frame.loc[frame["regime"].eq("extreme")]
    daily_max = frame.loc[frame["regime"].eq("daily_max")]
    if not normal.empty:
        result["normal_p50_mae"] = float(normal.iloc[0].get("p50_mae", float("nan")))
    if not extreme.empty:
        result["extreme_p50_mae"] = float(extreme.iloc[0].get("p50_mae", float("nan")))
    if not daily_max.empty:
        result["daily_max_q99_gap"] = float(daily_max.iloc[0].get("daily_max_q99_gap_max", float("nan")))
    return result


def _quality_metrics_from_row(row: dict[str, object]) -> QualityMetrics:
    return QualityMetrics(
        mae=_as_float(row.get("mae")),
        pinball=_as_float(row.get("pinball")),
        q99_exceedance_rate=_as_float(row.get("q99_exceedance_rate")),
        q99_excess_mean=_as_float(row.get("q99_excess_mean")),
        worst_q99_underprediction=_as_float(row.get("worst_q99_underprediction")),
        width_98=_as_float(row.get("width_98")),
        normal_p50_mae=_optional_float(row.get("normal_p50_mae")),
        extreme_p50_mae=_optional_float(row.get("extreme_p50_mae")),
        daily_max_q99_gap=_optional_float(row.get("daily_max_q99_gap")),
    )


def _coalesce(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return float("nan")


def _as_float(value: object) -> float:
    return float(_coalesce(value, 0.0))


def _optional_float(value: object) -> float | None:
    value = _coalesce(value)
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prediction quality flow benchmark registry.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--output",
        default="artifacts_phase2/quality_flow/benchmark_registry.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = (root / args.output).resolve()
    registry = build_quality_flow_registry(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_path, index=False)
    print(f"Wrote {len(registry)} registry rows to {output_path}")


if __name__ == "__main__":
    main()
