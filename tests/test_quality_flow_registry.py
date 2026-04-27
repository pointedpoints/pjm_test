from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.evaluation.quality_gate import QualityDecision
from scripts.experiments.build_quality_flow_registry import RegistrySpec, build_quality_flow_registry


def _write_metrics(metrics_dir: Path, split: str, *, run: str, mae: float, pinball: float) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"run": run, "model": run, "seed": 7, "mae": mae, "pinball": pinball}]).to_csv(
        metrics_dir / f"{split}_metrics.csv",
        index=False,
    )


def _write_quantile_diagnostics(
    metrics_dir: Path,
    split: str,
    *,
    q99_excess: float,
    width98: float,
) -> None:
    pd.DataFrame(
        [
            {
                "run": "run",
                "model": "model",
                "seed": 7,
                "post_q99_exceedance_rate": 0.02,
                "post_q99_excess_mean": q99_excess,
                "post_worst_q99_underprediction": 50.0,
                "post_width_98": width98,
                "post_daily_max_q99_gap_max": 20.0,
            }
        ]
    ).to_csv(metrics_dir / f"{split}_quantile_diagnostics.csv", index=False)


def _write_regime_metrics(metrics_dir: Path, split: str) -> None:
    pd.DataFrame(
        [
            {"regime": "normal", "p50_mae": 6.0},
            {"regime": "extreme", "p50_mae": 20.0},
            {"regime": "daily_max", "daily_max_q99_gap_max": 15.0},
        ]
    ).to_csv(metrics_dir / f"{split}_regime_metrics.csv", index=False)


def test_build_quality_flow_registry_combines_metrics_diagnostics_and_decisions(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    for metrics_dir, run, mae, pinball, q99_excess, width98 in [
        (baseline_dir, "baseline", 10.0, 3.0, 1.0, 50.0),
        (candidate_dir, "candidate", 9.8, 2.9, 0.8, 55.0),
    ]:
        _write_metrics(metrics_dir, "validation", run=run, mae=mae, pinball=pinball)
        _write_quantile_diagnostics(metrics_dir, "validation", q99_excess=q99_excess, width98=width98)
        _write_regime_metrics(metrics_dir, "validation")

    registry = build_quality_flow_registry(
        tmp_path,
        specs=[
            RegistrySpec(
                label="baseline",
                config_path="configs/baseline.yaml",
                metrics_dir=Path("baseline"),
                splits=("validation",),
                decision_override=QualityDecision.REFERENCE,
            ),
            RegistrySpec(
                label="candidate",
                config_path="configs/candidate.yaml",
                metrics_dir=Path("candidate"),
                splits=("validation",),
            ),
        ],
    )

    candidate = registry.loc[registry["run_name"].eq("candidate")].iloc[0]
    assert candidate["decision"] == "PROMOTE"
    assert candidate["config_path"] == "configs/candidate.yaml"
    assert candidate["normal_p50_mae"] == 6.0
    assert candidate["tail_gain"] > 0.0
    assert candidate["width98_ratio"] > 1.0

