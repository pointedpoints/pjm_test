from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pjm_forecast.quality_flow import build_quality_gate_summary, build_run_manifest


def _write_candidate_inputs(tmp_path: Path, *, spike_audit_status: str = "PASS") -> tuple[Path, Path, Path]:
    metrics_path = tmp_path / "test_metrics.csv"
    diagnostics_path = tmp_path / "test_quantile_diagnostics.csv"
    event_audit_dir = tmp_path / "event_audit"
    event_audit_dir.mkdir()

    pd.DataFrame(
        [
            {
                "run": "nbeatsx_seed_7",
                "model": "nbeatsx",
                "seed": 7,
                "mae": 18.5,
                "pinball": 2.25,
            }
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "run": "nbeatsx_seed_7",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.025,
                "post_q99_excess_mean": 1.5,
                "post_worst_q99_underprediction": 12.0,
                "post_width_98": 105.2,
            }
        ]
    ).to_csv(diagnostics_path, index=False)
    pd.DataFrame(
        [
            {"regime": "all", "before_width_98": 100.0, "after_width_98": 105.2},
            {"regime": "normal", "before_width_98": 100.0, "after_width_98": 105.2},
        ]
    ).to_csv(event_audit_dir / "width_by_regime.csv", index=False)
    (event_audit_dir / "overlay_implementation_audit.json").write_text(
        json.dumps({"selected_variant": "hour_cqr"}),
        encoding="utf-8",
    )
    (event_audit_dir / "spike_score_audit.json").write_text(
        json.dumps({"availability_status": spike_audit_status}),
        encoding="utf-8",
    )
    return metrics_path, diagnostics_path, event_audit_dir


def test_build_quality_gate_summary_marks_passing_canonical_candidate(tmp_path: Path) -> None:
    metrics_path, diagnostics_path, event_audit_dir = _write_candidate_inputs(tmp_path)

    summary = build_quality_gate_summary(
        split="test",
        metrics_path=metrics_path,
        quantile_diagnostics_path=diagnostics_path,
        event_audit_dir=event_audit_dir,
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["split"] == "test"
    assert row["run"] == "nbeatsx_seed_7"
    assert row["model"] == "nbeatsx"
    assert row["seed"] == 7
    assert row["pinball"] == 2.25
    assert row["mae"] == 18.5
    assert row["post_crossing_rate"] == 0.0
    assert row["post_q99_exceedance_rate"] == 0.025
    assert row["post_q99_excess_mean"] == 1.5
    assert row["post_worst_q99_underprediction"] == 12.0
    assert row["post_width_98"] == 105.2
    assert row["event_audit_available"] is True
    assert row["spike_score_audit_status"] == "PASS"
    assert row["all_width98_ratio"] == 1.052
    assert row["normal_width98_ratio"] == 1.052
    assert row["normal_width_status"] == "WARN"
    assert row["decision"] == "CANONICAL_CANDIDATE"


def test_build_quality_gate_summary_preserves_runs_with_missing_diagnostics(tmp_path: Path) -> None:
    metrics_path, diagnostics_path, event_audit_dir = _write_candidate_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "run": "other_run",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.0,
            }
        ]
    ).to_csv(diagnostics_path, index=False)

    summary = build_quality_gate_summary(
        split="test",
        metrics_path=metrics_path,
        quantile_diagnostics_path=diagnostics_path,
        event_audit_dir=event_audit_dir,
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["run"] == "nbeatsx_seed_7"
    assert pd.isna(row["post_crossing_rate"])
    assert pd.isna(row["post_q99_exceedance_rate"])
    assert row["decision"] == "REVIEW_REQUIRED"


def test_build_quality_gate_summary_requires_passing_spike_audit(tmp_path: Path) -> None:
    metrics_path, diagnostics_path, event_audit_dir = _write_candidate_inputs(tmp_path, spike_audit_status="FAIL")

    summary = build_quality_gate_summary(
        split="test",
        metrics_path=metrics_path,
        quantile_diagnostics_path=diagnostics_path,
        event_audit_dir=event_audit_dir,
    )

    row = summary.iloc[0]
    assert row["event_audit_available"] is True
    assert row["spike_score_audit_status"] == "FAIL"
    assert row["decision"] == "REVIEW_REQUIRED"


def test_build_run_manifest_records_source_artifacts(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    event_audit = tmp_path / "event_audit" / "spike_score_audit.json"
    event_audit.parent.mkdir()
    metrics.write_text("run,mae\nnbeatsx_seed_7,18.5\n", encoding="utf-8")
    diagnostics.write_text("run,post_width_98\nnbeatsx_seed_7,105.2\n", encoding="utf-8")
    event_audit.write_text('{"availability_status":"PASS"}\n', encoding="utf-8")

    manifest = build_run_manifest(
        split="test",
        config_path=tmp_path / "config.yaml",
        artifact_paths=[metrics, diagnostics, event_audit],
        model_name="nbeatsx",
        seed=7,
    )

    assert manifest["split"] == "test"
    assert manifest["config_path"] == str(tmp_path / "config.yaml")
    assert manifest["model_name"] == "nbeatsx"
    assert manifest["seed"] == 7
    assert len(manifest["artifacts"]) == 3
    assert all(item["exists"] for item in manifest["artifacts"])
    assert [item["path"] for item in manifest["artifacts"]] == [str(metrics), str(diagnostics), str(event_audit)]
