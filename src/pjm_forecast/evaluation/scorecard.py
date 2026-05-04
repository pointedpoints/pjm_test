from __future__ import annotations

import numpy as np
import pandas as pd


def build_experiment_scorecard_row(
    *,
    run_name: str,
    model: str,
    seed: int,
    metrics: dict[str, float],
    relative_error: pd.DataFrame,
    tail_regime: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run": run_name,
        "model": model,
        "seed": int(seed),
        "mae": float(metrics.get("mae", np.nan)),
        "rmse": float(metrics.get("rmse", np.nan)),
        "smape": float(metrics.get("smape", np.nan)),
        "pinball": float(metrics.get("pinball", np.nan)),
    }
    row.update(_relative_fields(relative_error))
    row.update(_tail_fields(tail_regime))
    return row


def _relative_fields(relative_error: pd.DataFrame) -> dict[str, float]:
    fields: dict[str, float] = {}
    mapping = {
        ("all", "all"): "all",
        ("actual_price_bin", "10-20"): "10_20",
        ("actual_price_bin", "20-30"): "20_30",
        ("actual_price_bin", "30-50"): "30_50",
        ("actual_price_bin", "50-100"): "50_100",
    }
    for (slice_type, label), suffix in mapping.items():
        match = relative_error.loc[relative_error["slice_type"].eq(slice_type) & relative_error["slice"].eq(label)]
        if match.empty:
            continue
        record = match.iloc[0]
        fields[f"q50_wape_{suffix}"] = float(record.get("wape", np.nan))
        fields[f"q50_smape_{suffix}"] = float(record.get("smape", np.nan))
        fields[f"q50_median_ape_{suffix}"] = float(record.get("median_ape", np.nan))
        fields[f"q50_p75_ape_{suffix}"] = float(record.get("p75_ape", np.nan))
        fields[f"q50_p90_ape_{suffix}"] = float(record.get("p90_ape", np.nan))
    return fields


def _tail_fields(tail_regime: pd.DataFrame) -> dict[str, float]:
    fields: dict[str, float] = {}
    mapping = {
        "all": "all",
        "actual_p95_p99": "p95_p99",
        "actual_gt_p99": "gt_p99",
    }
    for regime, suffix in mapping.items():
        match = tail_regime.loc[tail_regime["regime"].eq(regime)]
        if match.empty:
            continue
        record = match.iloc[0]
        fields[f"q99_coverage_{suffix}"] = float(record.get("q99_upper_coverage", np.nan))
        fields[f"q995_coverage_{suffix}"] = float(record.get("q995_upper_coverage", np.nan))
        fields[f"q99_excess_mean_{suffix}"] = float(record.get("q99_excess_mean", np.nan))
        fields[f"q99_excess_max_{suffix}"] = float(record.get("q99_excess_max", np.nan))
    return fields
