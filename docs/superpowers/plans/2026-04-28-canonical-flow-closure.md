# Canonical Flow Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the post-model canonical workflow so a current NHITS run can be reproduced, audited, gated, packaged, and reported without ad-hoc experiment commands.

**Architecture:** Keep `Workspace` as the workflow boundary and add small, config-driven stages behind it. Evaluation stays the source of truth; report export remains a derived artifact view. Event-risk audit and quality gate become mainline flow artifacts, not model-quality experiments.

**Tech Stack:** Python 3.12, pandas, pytest, existing `Workspace`, existing `Evaluator`, existing event-risk overlay audit builder, PowerShell/uv CLI commands.

---

## Scope

This plan intentionally does **not** tune model quality, change canonical NHITS parameters, change the time protocol, or promote `peak_hours` over the current all-hours event-risk overlay. It only closes the workflow after the current candidate exists:

- run canonical validation/test
- export event-risk promotion audit
- summarize quality gates
- export report assets
- export the configured canonical model snapshot
- write a machine-readable run manifest
- document the one-command flow and release checklist

## File Structure

- Modify `src/pjm_forecast/workspace.py`
  - Add artifact paths for audit, quality gate, and run manifest.
  - Add `audit_event_risk_overlay(split="test")`.
  - Add `finalize_quality_flow(split="test")`.
  - Generalize snapshot export defaults to the configured benchmark model.

- Modify `src/pjm_forecast/pipeline.py`
  - Add stages after evaluation/export:
    - `audit_event_risk_overlay`
    - `finalize_quality_flow`
    - `export_model_snapshot`

- Modify `scripts/run_pipeline.py`
  - No new parsing model needed if stage names come from `STAGE_ORDER`.

- Create `scripts/audit_event_risk_overlay.py`
  - CLI shim for `Workspace.audit_event_risk_overlay`.

- Create `scripts/finalize_quality_flow.py`
  - CLI shim for `Workspace.finalize_quality_flow`.

- Create `scripts/ops/export_model_snapshot.py`
  - Generic snapshot CLI replacing the NBEATSx-only public entry point for current NHITS use.

- Modify `src/pjm_forecast/ops.py`
  - Keep `export_nbeatsx_snapshot` compatibility.
  - Add `export_configured_model_snapshot(config_path, model_name=None, snapshot_name=None)`.

- Create `src/pjm_forecast/quality_flow.py`
  - Pure functions that read already-written artifacts and produce:
    - `quality_gate_summary.csv`
    - `run_manifest.json`

- Modify `README.md`
  - Update canonical workflow to include audit, finalize, and snapshot export.

- Create `docs/protocol/canonical_release_checklist.md`
  - Human runbook for “before PR / before merge / after merge”.

- Modify tests:
  - `tests/test_workspace.py`
  - `tests/test_ops.py`
  - Create `tests/test_quality_flow_manifest.py`

---

### Task 1: Add Artifact Paths For Flow Outputs

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write the failing test**

Add assertions to `test_workspace_open_respects_root_override_and_artifact_contract`:

```python
assert workspace.artifacts.event_risk_audit_dir("test") == (
    tmp_path / "run" / "artifacts" / "metrics" / "test_event_risk_tail_overlay"
).resolve()
assert workspace.artifacts.quality_gate_summary("test") == (
    tmp_path / "run" / "artifacts" / "metrics" / "test_quality_gate_summary.csv"
).resolve()
assert workspace.artifacts.run_manifest("test") == (
    tmp_path / "run" / "artifacts" / "metrics" / "test_run_manifest.json"
).resolve()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_open_respects_root_override_and_artifact_contract -q
```

Expected: fail with `AttributeError` for missing artifact methods.

- [ ] **Step 3: Implement artifact methods**

Add these methods to `ArtifactStore` in `src/pjm_forecast/workspace.py`:

```python
def event_risk_audit_dir(self, split: str) -> Path:
    return self.directories["metrics_dir"] / f"{split}_event_risk_tail_overlay"

def quality_gate_summary(self, split: str) -> Path:
    return self.directories["metrics_dir"] / f"{split}_quality_gate_summary.csv"

def run_manifest(self, split: str) -> Path:
    return self.directories["metrics_dir"] / f"{split}_run_manifest.json"
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_open_respects_root_override_and_artifact_contract -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src\pjm_forecast\workspace.py tests\test_workspace.py
git commit -m "Add canonical flow artifact paths"
```

---

### Task 2: Add A Pure Quality Flow Manifest Builder

**Files:**
- Create: `src/pjm_forecast/quality_flow.py`
- Test: `tests/test_quality_flow_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quality_flow_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pjm_forecast.quality_flow import build_quality_gate_summary, build_run_manifest


def test_build_quality_gate_summary_marks_passing_canonical_candidate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "test_metrics.csv"
    diagnostics_path = tmp_path / "test_quantile_diagnostics.csv"
    event_audit_dir = tmp_path / "test_event_risk_tail_overlay"
    event_audit_dir.mkdir()
    pd.DataFrame(
        [
            {
                "run": "nhits_tail_grid_weighted_main_test_seed7",
                "model": "nhits_tail_grid_weighted_main",
                "seed": 7,
                "pinball": 3.2483,
                "mae": 10.9858,
            }
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "run": "nhits_tail_grid_weighted_main_test_seed7",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.0162,
                "post_q99_excess_mean": 0.4707,
                "post_worst_q99_underprediction": 279.19,
                "post_width_98": 75.05,
            }
        ]
    ).to_csv(diagnostics_path, index=False)
    (event_audit_dir / "overlay_implementation_audit.json").write_text(
        json.dumps({"test_used_for_selection": False, "q50_changed": False, "crossing_after_overlay": 0.0}),
        encoding="utf-8",
    )
    (event_audit_dir / "spike_score_audit.json").write_text(
        json.dumps({"availability_status": "PASS", "uses_y": False, "uses_horizon_truth": False}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"split": "test", "regime": "all", "width98_ratio": 1.085},
            {"split": "test", "regime": "normal", "width98_ratio": 1.052},
        ]
    ).to_csv(event_audit_dir / "width_by_regime.csv", index=False)

    summary = build_quality_gate_summary(
        split="test",
        metrics_path=metrics_path,
        quantile_diagnostics_path=diagnostics_path,
        event_audit_dir=event_audit_dir,
    )

    row = summary.iloc[0]
    assert row["decision"] == "CANONICAL_CANDIDATE"
    assert row["event_audit_available"] is True
    assert row["normal_width_status"] == "WARN"


def test_build_run_manifest_records_source_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "test_metrics.csv"
    diagnostics_path = tmp_path / "test_quantile_diagnostics.csv"
    quality_path = tmp_path / "test_quality_gate_summary.csv"
    for path in [metrics_path, diagnostics_path, quality_path]:
        path.write_text("run,value\nexample,1\n", encoding="utf-8")

    manifest = build_run_manifest(
        split="test",
        config_path=Path("configs/pjm_day_ahead_current_processed.yaml"),
        artifact_paths=[metrics_path, diagnostics_path, quality_path],
        model_name="nhits_tail_grid_weighted_main",
        seed=7,
    )

    assert manifest["split"] == "test"
    assert manifest["model_name"] == "nhits_tail_grid_weighted_main"
    assert manifest["seed"] == 7
    assert len(manifest["artifacts"]) == 3
    assert all(item["exists"] for item in manifest["artifacts"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
uv run python -m pytest tests\test_quality_flow_manifest.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'pjm_forecast.quality_flow'`.

- [ ] **Step 3: Implement `src/pjm_forecast/quality_flow.py`**

Create:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def build_quality_gate_summary(
    *,
    split: str,
    metrics_path: Path,
    quantile_diagnostics_path: Path,
    event_audit_dir: Path,
) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    diagnostics = pd.read_csv(quantile_diagnostics_path)
    merged = metrics.merge(diagnostics, on="run", how="left", suffixes=("", "_diagnostic"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        event_audit = _load_event_audit(event_audit_dir, split=split)
        crossing = float(row.get("post_crossing_rate", 0.0))
        q99_exceed = float(row.get("post_q99_exceedance_rate", 1.0))
        normal_width_ratio = event_audit["normal_width_ratio"]
        decision = "CANONICAL_CANDIDATE"
        if crossing != 0.0 or q99_exceed > 0.025 or not event_audit["event_audit_available"]:
            decision = "REVIEW_REQUIRED"
        rows.append(
            {
                "split": split,
                "run": row["run"],
                "model": row.get("model"),
                "seed": int(row.get("seed", 0)),
                "pinball": float(row.get("pinball", float("nan"))),
                "mae": float(row.get("mae", float("nan"))),
                "post_crossing_rate": crossing,
                "post_q99_exceedance_rate": q99_exceed,
                "post_q99_excess_mean": float(row.get("post_q99_excess_mean", float("nan"))),
                "post_worst_q99_underprediction": float(row.get("post_worst_q99_underprediction", float("nan"))),
                "post_width_98": float(row.get("post_width_98", float("nan"))),
                "event_audit_available": bool(event_audit["event_audit_available"]),
                "spike_score_audit_status": event_audit["spike_score_audit_status"],
                "all_width98_ratio": event_audit["all_width98_ratio"],
                "normal_width98_ratio": normal_width_ratio,
                "normal_width_status": "PASS" if normal_width_ratio <= 1.05 else "WARN",
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def build_run_manifest(
    *,
    split: str,
    config_path: Path,
    artifact_paths: list[Path],
    model_name: str,
    seed: int,
) -> dict[str, object]:
    return {
        "split": split,
        "config_path": str(config_path),
        "model_name": model_name,
        "seed": int(seed),
        "artifacts": [_artifact_record(path) for path in artifact_paths],
    }


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _load_event_audit(event_audit_dir: Path, *, split: str) -> dict[str, object]:
    implementation_path = event_audit_dir / "overlay_implementation_audit.json"
    spike_path = event_audit_dir / "spike_score_audit.json"
    width_path = event_audit_dir / "width_by_regime.csv"
    if not implementation_path.exists() or not spike_path.exists() or not width_path.exists():
        return {
            "event_audit_available": False,
            "spike_score_audit_status": "MISSING",
            "all_width98_ratio": float("nan"),
            "normal_width_ratio": float("nan"),
        }
    spike = json.loads(spike_path.read_text(encoding="utf-8"))
    width = pd.read_csv(width_path)
    width_split = width.loc[width["split"].eq(split)] if "split" in width.columns else width
    return {
        "event_audit_available": True,
        "spike_score_audit_status": str(spike.get("availability_status", "UNKNOWN")),
        "all_width98_ratio": _regime_ratio(width_split, "all"),
        "normal_width_ratio": _regime_ratio(width_split, "normal"),
    }


def _regime_ratio(width: pd.DataFrame, regime: str) -> float:
    subset = width.loc[width["regime"].eq(regime)]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0]["width98_ratio"])


def _artifact_record(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path) if exists and path.is_file() else None,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
uv run python -m pytest tests\test_quality_flow_manifest.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src\pjm_forecast\quality_flow.py tests\test_quality_flow_manifest.py
git commit -m "Add canonical quality flow manifest builder"
```

---

### Task 3: Mainline Event-Risk Audit Stage

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Create: `scripts/audit_event_risk_overlay.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_workspace.py`:

```python
def test_workspace_audit_event_risk_overlay_writes_expected_files(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["backtest"]["benchmark_models"] = ["nhits_tail_grid_weighted_main"]
    workspace.config.raw["models"]["nhits_tail_grid_weighted_main"] = {"type": "nhits"}
    workspace.config.raw["report"]["quantile_postprocess"] = {
        "calibration": {
            "enabled": True,
            "method": "cqr_asymmetric",
            "group_by": "hour",
            "min_group_size": 1,
            "regime_score_column": "spike_score",
            "regime_threshold": 0.5,
        },
        "event_risk_tail_overlay": {
            "enabled": True,
            "source_split": "validation",
            "risk_score_column": "spike_score",
            "risk_aggregation": "mean",
            "risk_threshold_quantile": 0.5,
            "residual_quantile": 1.0,
            "max_uplift": 50.0,
            "target_quantiles": [0.99, 0.995],
        },
    }
    validation = _quantile_prediction_frame("validation")
    test = _quantile_prediction_frame("test")
    workspace.directories["prediction_dir"].mkdir(parents=True, exist_ok=True)
    validation.to_parquet(
        workspace.artifacts.prediction("nhits_tail_grid_weighted_main", "validation", 7),
        index=False,
    )
    test.to_parquet(
        workspace.artifacts.prediction("nhits_tail_grid_weighted_main", "test", 7),
        index=False,
    )

    output_dir = workspace.audit_event_risk_overlay("test")

    assert (output_dir / "overlay_implementation_audit.json").exists()
    assert (output_dir / "spike_score_audit.json").exists()
    assert (output_dir / "width_by_regime.csv").exists()
```

Add helper:

```python
def _quantile_prediction_frame(split: str) -> pd.DataFrame:
    rows = []
    for day, spike_score in [("2026-01-01", 0.2), ("2026-01-02", 0.95)]:
        for hour in [0, 19]:
            ds = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            for quantile, y_pred in [(0.5, 95.0), (0.95, 100.0), (0.99, 105.0), (0.995, 110.0)]:
                rows.append(
                    {
                        "ds": ds,
                        "y": 120.0 if spike_score > 0.9 else 100.0,
                        "y_pred": y_pred,
                        "quantile": quantile,
                        "model": "nhits_tail_grid_weighted_main",
                        "split": split,
                        "seed": 7,
                        "metadata": "{}",
                        "spike_score": spike_score,
                    }
                )
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_audit_event_risk_overlay_writes_expected_files -q
```

Expected: fail with `AttributeError: 'Workspace' object has no attribute 'audit_event_risk_overlay'`.

- [ ] **Step 3: Implement `Workspace.audit_event_risk_overlay`**

Add imports in `workspace.py`:

```python
from .evaluation.event_risk_tail_overlay import build_event_risk_tail_overlay_audit_artifacts
from .quality_flow import write_json
```

Add method:

```python
def audit_event_risk_overlay(self, split: SplitName = "test") -> Path:
    event_cfg = self.config.report.get("quantile_postprocess", {}).get("event_risk_tail_overlay", {})
    if not bool(event_cfg.get("enabled", False)):
        raise ValueError("event_risk_tail_overlay is not enabled in report.quantile_postprocess.")
    source_split = str(event_cfg.get("source_split", "validation"))
    model_name = str(self.config.backtest["benchmark_models"][0])
    seed = int(self.config.project["benchmark_seed"])
    validation_path = self.artifacts.prediction(model_name, source_split, seed)
    target_path = self.artifacts.prediction(model_name, split, seed)
    validation_frame = pd.read_parquet(validation_path)
    target_frame = pd.read_parquet(target_path)
    risk_score_column = str(event_cfg.get("risk_score_column", "spike_score"))
    audit = build_event_risk_tail_overlay_audit_artifacts(
        validation_frame,
        test_frame=target_frame,
        validation_holdout_days=int(event_cfg.get("validation_holdout_days", 91)),
        risk_score_column=risk_score_column,
        risk_aggregation=str(event_cfg.get("risk_aggregation", "mean")),
        risk_threshold_quantile=float(event_cfg.get("risk_threshold_quantile", 0.90)),
        residual_quantile=float(event_cfg.get("residual_quantile", 1.0)),
        max_uplift=float(event_cfg.get("max_uplift", 50.0)),
        target_quantiles=event_cfg.get("target_quantiles", [0.99, 0.995]),
        calibration_min_group_size=int(
            self.config.report.get("quantile_postprocess", {}).get("calibration", {}).get("min_group_size", 24)
        ),
        interval_coverage_floors=self.config.report.get("quantile_postprocess", {})
        .get("calibration", {})
        .get("interval_coverage_floors"),
        risk_score_input_columns=self._spike_score_input_columns(risk_score_column),
    )
    output_dir = self.artifacts.event_risk_audit_dir(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "overlay_implementation_audit.json", audit.implementation_audit)
    write_json(output_dir / "spike_score_audit.json", audit.spike_score_audit)
    audit.active_day_diagnostics.to_csv(output_dir / "active_day_diagnostics.csv", index=False)
    audit.active_days_by_month.to_csv(output_dir / "active_days_by_month.csv", index=False)
    audit.width_by_regime.to_csv(output_dir / "width_by_regime.csv", index=False)
    audit.pinball_by_quantile.to_csv(output_dir / "pinball_by_quantile.csv", index=False)
    audit.conservative_variant_grid.to_csv(output_dir / "conservative_variant_grid.csv", index=False)
    audit.daily_max_gap_detail.to_csv(output_dir / "daily_max_gap_detail.csv", index=False)
    audit.event_day_before_after.to_csv(output_dir / "event_day_before_after.csv", index=False)
    return output_dir
```

Add helper:

```python
def _spike_score_input_columns(self, risk_score_column: str) -> list[str]:
    for feature in self.config.raw.get("features", {}).get("derived_features", []) or []:
        if feature.get("kind") == "spike_score" and feature.get("name") == risk_score_column:
            return [str(item["source"]) for item in feature.get("inputs", []) if "source" in item]
    return []
```

- [ ] **Step 4: Create CLI shim**

Create `scripts/audit_event_risk_overlay.py`:

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Write canonical event-risk overlay audit artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    output_dir = Workspace.open(args.config).audit_event_risk_overlay(split=args.split)
    print(f"Wrote event-risk overlay audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_audit_event_risk_overlay_writes_expected_files -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\workspace.py scripts\audit_event_risk_overlay.py tests\test_workspace.py
git commit -m "Add canonical event risk audit stage"
```

---

### Task 4: Add Quality Finalization Stage

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Create: `scripts/finalize_quality_flow.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workspace.py`:

```python
def test_workspace_finalize_quality_flow_writes_summary_and_manifest(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    metrics_dir = workspace.directories["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"run": "seasonal_naive_test_seed7", "model": "seasonal_naive", "seed": 7, "mae": 1.0, "pinball": 1.0}]
    ).to_csv(workspace.artifacts.metrics("test"), index=False)
    pd.DataFrame(
        [
            {
                "run": "seasonal_naive_test_seed7",
                "post_crossing_rate": 0.0,
                "post_q99_exceedance_rate": 0.02,
                "post_q99_excess_mean": 0.1,
                "post_worst_q99_underprediction": 10.0,
                "post_width_98": 20.0,
            }
        ]
    ).to_csv(workspace.artifacts.quantile_diagnostics("test"), index=False)

    output_paths = workspace.finalize_quality_flow("test")

    assert workspace.artifacts.quality_gate_summary("test") in output_paths
    assert workspace.artifacts.run_manifest("test") in output_paths
    assert workspace.artifacts.quality_gate_summary("test").exists()
    assert workspace.artifacts.run_manifest("test").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_finalize_quality_flow_writes_summary_and_manifest -q
```

Expected: fail with `AttributeError` for missing method.

- [ ] **Step 3: Implement method**

Add imports:

```python
from .quality_flow import build_quality_gate_summary, build_run_manifest, write_json
```

Add method to `Workspace`:

```python
def finalize_quality_flow(self, split: SplitName = "test") -> list[Path]:
    model_name = str(self.config.backtest["benchmark_models"][0])
    seed = int(self.config.project["benchmark_seed"])
    quality = build_quality_gate_summary(
        split=split,
        metrics_path=self.artifacts.metrics(split),
        quantile_diagnostics_path=self.artifacts.quantile_diagnostics(split),
        event_audit_dir=self.artifacts.event_risk_audit_dir(split),
    )
    quality_path = self.artifacts.quality_gate_summary(split)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(quality_path, index=False)
    manifest = build_run_manifest(
        split=split,
        config_path=self.config.path,
        model_name=model_name,
        seed=seed,
        artifact_paths=[
            self.artifacts.metrics(split),
            self.artifacts.quantile_diagnostics(split),
            self.artifacts.regime_metrics(split),
            self.artifacts.spike_score_diagnostics(split),
            self.artifacts.scenario_diagnostics(split),
            self.artifacts.dm(split),
            quality_path,
        ],
    )
    manifest_path = write_json(self.artifacts.run_manifest(split), manifest)
    return [quality_path, manifest_path]
```

- [ ] **Step 4: Create CLI**

Create `scripts/finalize_quality_flow.py`:

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Write canonical quality gate summary and run manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    paths = Workspace.open(args.config).finalize_quality_flow(split=args.split)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_finalize_quality_flow_writes_summary_and_manifest -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\workspace.py scripts\finalize_quality_flow.py tests\test_workspace.py
git commit -m "Add canonical quality finalization stage"
```

---

### Task 5: Wire New Stages Into Pipeline

**Files:**
- Modify: `src/pjm_forecast/pipeline.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Write failing stage-order test**

Update `test_pipeline_stage_order_excludes_retrieval` and rename it to `test_pipeline_stage_order_includes_quality_closure`:

```python
def test_pipeline_stage_order_includes_quality_closure() -> None:
    assert STAGE_ORDER == [
        "prepare_data",
        "tune_model",
        "backtest_all_models",
        "evaluate_and_plot",
        "audit_event_risk_overlay",
        "export_report_assets",
        "finalize_quality_flow",
        "export_model_snapshot",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_pipeline_stage_order_includes_quality_closure -q
```

Expected: fail because current stage order omits the new stages.

- [ ] **Step 3: Implement pipeline stages**

Update `STAGE_ORDER` in `src/pjm_forecast/pipeline.py`:

```python
STAGE_ORDER = [
    "prepare_data",
    "tune_model",
    "backtest_all_models",
    "evaluate_and_plot",
    "audit_event_risk_overlay",
    "export_report_assets",
    "finalize_quality_flow",
    "export_model_snapshot",
]
```

Add stage functions:

```python
def _run_audit_event_risk_overlay(workspace: Workspace, split: str) -> None:
    workspace.audit_event_risk_overlay(split=split)


def _run_finalize_quality_flow(workspace: Workspace, split: str) -> None:
    workspace.finalize_quality_flow(split=split)


def _run_export_model_snapshot(workspace: Workspace, split: str) -> None:
    del split
    model_name = str(workspace.config.backtest["benchmark_models"][0])
    workspace.export_model_snapshot(model_name=model_name)
```

Add entries to `STAGE_FUNCTIONS`:

```python
"audit_event_risk_overlay": _run_audit_event_risk_overlay,
"finalize_quality_flow": _run_finalize_quality_flow,
"export_model_snapshot": _run_export_model_snapshot,
```

- [ ] **Step 4: Run targeted test**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_pipeline_stage_order_includes_quality_closure -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src\pjm_forecast\pipeline.py tests\test_workspace.py
git commit -m "Wire canonical quality closure into pipeline"
```

---

### Task 6: Add Generic Snapshot Export CLI

**Files:**
- Modify: `src/pjm_forecast/ops.py`
- Create: `scripts/ops/export_model_snapshot.py`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ops.py`:

```python
def test_export_configured_model_snapshot_uses_configured_benchmark(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected = tmp_path / "artifacts" / "models" / "nhits_tail_grid_weighted_main_snapshot"

    class StubWorkspace:
        config = type("Config", (), {"backtest": {"benchmark_models": ["nhits_tail_grid_weighted_main"]}})()

        def export_model_snapshot(self, *, model_name: str, snapshot_name: str | None = None) -> Path:
            captured["model_name"] = model_name
            captured["snapshot_name"] = snapshot_name
            return expected

    monkeypatch.setattr("pjm_forecast.ops.Workspace.open", lambda config_path: StubWorkspace())

    output = ops.export_configured_model_snapshot("configs/pjm_day_ahead_current_processed.yaml")

    assert output == expected
    assert captured["model_name"] == "nhits_tail_grid_weighted_main"
    assert captured["snapshot_name"] == "nhits_tail_grid_weighted_main_snapshot"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_ops.py::test_export_configured_model_snapshot_uses_configured_benchmark -q
```

Expected: fail with `AttributeError`.

- [ ] **Step 3: Implement op**

Add to `src/pjm_forecast/ops.py`:

```python
def export_configured_model_snapshot(
    config_path: str,
    model_name: str | None = None,
    snapshot_name: str | None = None,
) -> Path:
    workspace = Workspace.open(config_path)
    resolved_model_name = model_name or str(workspace.config.backtest["benchmark_models"][0])
    resolved_snapshot_name = snapshot_name or f"{resolved_model_name}_snapshot"
    return workspace.export_model_snapshot(model_name=resolved_model_name, snapshot_name=resolved_snapshot_name)
```

- [ ] **Step 4: Create CLI**

Create `scripts/ops/export_model_snapshot.py`:

```python
from __future__ import annotations

import argparse

from pjm_forecast.ops import export_configured_model_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fitted snapshot for the configured benchmark model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--snapshot-name")
    args = parser.parse_args()
    output_dir = export_configured_model_snapshot(
        args.config,
        model_name=args.model_name,
        snapshot_name=args.snapshot_name,
    )
    print(f"Exported model snapshot to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_ops.py -q
```

Expected: all tests in `tests/test_ops.py` pass.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\ops.py scripts\ops\export_model_snapshot.py tests\test_ops.py
git commit -m "Add generic configured model snapshot export"
```

---

### Task 7: Export New Closure Artifacts Into Report Bundle

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Write failing test**

Extend `test_workspace_main_flow_writes_predictions_metrics_and_report` after finalization is available:

```python
workspace.finalize_quality_flow("test")
rebuilt = workspace.export_report("test")
assert workspace.artifacts.report_asset("test_quality_gate_summary.csv") in rebuilt
assert workspace.artifacts.report_asset("test_run_manifest.json") in rebuilt
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_main_flow_writes_predictions_metrics_and_report -q
```

Expected: fail because `export_report_bundle` does not copy the new files.

- [ ] **Step 3: Update report export**

Add these sources to `ArtifactStore.export_report_bundle`:

```python
self.quality_gate_summary(split),
self.run_manifest(split),
```

- [ ] **Step 4: Run test**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_main_flow_writes_predictions_metrics_and_report -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src\pjm_forecast\workspace.py tests\test_workspace.py
git commit -m "Export quality closure artifacts in report bundle"
```

---

### Task 8: Update Documentation And Release Checklist

**Files:**
- Modify: `README.md`
- Create: `docs/protocol/canonical_release_checklist.md`

- [ ] **Step 1: Update README canonical workflow**

Replace the pipeline wrapper section with:

```markdown
Run the canonical closure pipeline:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split validation
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

The wrapper writes:

- predictions under `artifacts_current/predictions/`
- metrics under `artifacts_current/metrics/`
- event-risk audit artifacts under `artifacts_current/metrics/{split}_event_risk_tail_overlay/`
- report exports under `artifacts_current/report/`
- model snapshot under `artifacts_current/models/nhits_tail_grid_weighted_main_snapshot/`
- run manifest under `artifacts_current/metrics/{split}_run_manifest.json`
```

- [ ] **Step 2: Add release checklist**

Create `docs/protocol/canonical_release_checklist.md`:

```markdown
# Canonical Release Checklist

## Before Running

- Confirm branch is clean except intended changes: `git status --short`
- Confirm config is `configs/pjm_day_ahead_current_processed.yaml`
- Confirm v1 time protocol remains local-time naive; no UTC remapping.

## Run

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split validation
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test
uv run python -m pytest
```

## Gate

- `test_quality_gate_summary.csv` has decision `CANONICAL_CANDIDATE` or better.
- `post_crossing_rate == 0`.
- `post_q99_exceedance_rate <= 0.025`.
- `spike_score_audit_status == PASS`.
- `normal_width_status` is `PASS` or explicitly documented as `WARN`.

## Artifacts To Inspect

- `artifacts_current/metrics/test_metrics.csv`
- `artifacts_current/metrics/test_quantile_diagnostics.csv`
- `artifacts_current/metrics/test_event_risk_tail_overlay/width_by_regime.csv`
- `artifacts_current/metrics/test_quality_gate_summary.csv`
- `artifacts_current/metrics/test_run_manifest.json`
- `artifacts_current/report/`
- `artifacts_current/models/nhits_tail_grid_weighted_main_snapshot/manifest.json`

## Before Push

- Run `uv run python -m pytest`.
- Confirm generated heavyweight artifacts are not staged.
- Commit only source, tests, docs, and small intentional configs.
```

- [ ] **Step 3: Commit**

```powershell
git add README.md docs\protocol\canonical_release_checklist.md
git commit -m "Document canonical flow closure runbook"
```

---

### Task 9: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
uv run python -m pytest tests\test_quality_flow_manifest.py tests\test_workspace.py tests\test_ops.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full tests**

Run:

```powershell
uv run python -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Run pipeline smoke on test using existing artifacts**

If canonical predictions already exist, run only closure stages:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test --start-from audit_event_risk_overlay --stop-after finalize_quality_flow
```

Expected:

- `artifacts_current/metrics/test_event_risk_tail_overlay/overlay_implementation_audit.json`
- `artifacts_current/metrics/test_quality_gate_summary.csv`
- `artifacts_current/metrics/test_run_manifest.json`

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: only intended source/test/doc files are modified; generated artifacts are not staged.

- [ ] **Step 5: Commit any final fix**

Only if Step 4 shows intended uncommitted source/test/doc changes:

```powershell
git add <intended-files>
git commit -m "Complete canonical flow closure"
```

---

## Self-Review

Spec coverage:

- Mainline event-risk audit: Task 3.
- Quality gate summary: Task 2 and Task 4.
- Run manifest: Task 2 and Task 4.
- Pipeline integration: Task 5.
- Snapshot/export completion: Task 6.
- Report bundle completion: Task 7.
- README/runbook: Task 8.
- Verification: Task 9.

Placeholder scan:

- No `TBD`.
- No open-ended “add appropriate handling”.
- Every new function has a concrete test and command.

Type consistency:

- `event_risk_audit_dir(split)`, `quality_gate_summary(split)`, and `run_manifest(split)` are defined in Task 1 and used consistently later.
- `build_quality_gate_summary` and `build_run_manifest` signatures are defined in Task 2 and used consistently in Task 4.
- `export_configured_model_snapshot` is defined in Task 6 and used only through `ops`/CLI.

