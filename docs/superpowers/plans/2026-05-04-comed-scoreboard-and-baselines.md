# COMED Scoreboard and Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible COMED experiment scorecard that reports q50 relative error, tail regime coverage, and baseline comparisons for the current NHITS pipeline before any larger model changes.

**Architecture:** Add focused evaluation modules for relative-error slices and tail-regime diagnostics, wire them into `Evaluator` and `ArtifactStore`, and add an experiment helper for baseline comparison using existing model registry entries. Keep model-training changes out of this first plan; this plan creates the scoreboard needed to evaluate later P50, tail, frequency, filtered-target, and RAG experiments.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing `Workspace`, `Evaluator`, `ArtifactStore`, and model registry.

---

## Scope

This plan implements only:

1. Repository cleanup before execution.
2. Evaluation hardening for q50 relative error and actual-regime tail coverage.
3. A baseline scorecard workflow for existing model families.
4. Tests and documentation for the new artifacts.

This plan does not implement:

- VST / target-transform grid.
- Spike-filtered target training.
- Frequency-domain features or decomposition.
- Quantile-aware RAG.
- Local tail expert.
- Scenario/copula refinement.

Those are intentionally deferred until this scoreboard exists.

## Execution Preconditions

Before starting Task 1, clean the repository state:

- Decide what to do with the existing untracked file:
  - `docs/superpowers/plans/2026-05-04-comed-forecasting-research-roadmap.md`
- Recommended action: keep it as background research and commit it separately before implementing this plan.
- Create an implementation branch:

```powershell
git switch -c codex/comed-scoreboard-and-baselines
```

- Confirm the worktree is otherwise clean:

```powershell
git status --short --branch
```

Expected:

```text
## codex/comed-scoreboard-and-baselines
```

The `git status` command may print access warnings for local scratch directories such as `ptbt_run_1/` or `pytest-cache-files-*`. Those warnings are acceptable only if no tracked or intentionally edited project files are hidden by them.

## Files

### Create

- `src/pjm_forecast/evaluation/relative_error.py`  
  Computes q50 relative-error slices by actual price bin, month, and hour.

- `src/pjm_forecast/evaluation/tail_regime.py`  
  Computes actual-price-regime tail coverage and daily peak tail gaps.

- `src/pjm_forecast/evaluation/scorecard.py`  
  Builds one compact run-level scorecard row per prediction run by combining existing metrics, relative-error slices, and tail diagnostics.

- `scripts/experiments/scorecard_baselines.py`  
  Runs or summarizes configured baseline prediction runs and writes a consolidated baseline scorecard.

- `tests/test_relative_error.py`  
  Unit tests for q50 relative-error slice calculations.

- `tests/test_tail_regime.py`  
  Unit tests for actual-regime tail coverage and daily peak q99 gap calculations.

- `tests/test_scorecard.py`  
  Unit tests for run-level scorecard composition.

### Modify

- `src/pjm_forecast/workspace.py`  
  Add artifact paths and CSV writer methods for the new evaluation artifacts.

- `src/pjm_forecast/evaluation/evaluator.py`  
  Add methods that compute and write the new artifacts during evaluation.

- `src/pjm_forecast/evaluation/__init__.py`  
  Export the new evaluation functions.

- `scripts/evaluate_and_plot.py`  
  No CLI argument changes are required. The existing `Workspace.open(config).evaluate(split)` path should write the new artifacts automatically.

- `README.md`  
  Document the new scorecard outputs and how to run the baseline scorecard workflow.

---

### Task 0: Repository Cleanup Gate

**Files:**
- Inspect: `docs/superpowers/plans/2026-05-04-comed-forecasting-research-roadmap.md`
- Inspect: current git status

- [ ] **Step 1: Confirm current status**

Run:

```powershell
git status --short --branch
```

Expected before cleanup:

```text
## main...origin/main
?? docs/superpowers/plans/2026-05-04-comed-forecasting-research-roadmap.md
```

- [ ] **Step 2: Review the untracked research roadmap**

Run:

```powershell
Get-Content docs\superpowers\plans\2026-05-04-comed-forecasting-research-roadmap.md -TotalCount 80
```

Expected: file starts with:

```text
# COMED Day-Ahead Probabilistic Forecasting Research Roadmap
```

- [ ] **Step 3: Commit the research roadmap separately if keeping it**

Run:

```powershell
git add docs\superpowers\plans\2026-05-04-comed-forecasting-research-roadmap.md
git commit -m "docs: add COMED forecasting research roadmap"
```

Expected:

```text
[main <hash>] docs: add COMED forecasting research roadmap
```

If the file should not be kept, do not delete it silently. Ask for explicit confirmation before removing an untracked document.

- [ ] **Step 4: Create the implementation branch**

Run:

```powershell
git switch -c codex/comed-scoreboard-and-baselines
```

Expected:

```text
Switched to a new branch 'codex/comed-scoreboard-and-baselines'
```

- [ ] **Step 5: Verify clean implementation start**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## codex/comed-scoreboard-and-baselines
```

Access-denied warnings for local scratch directories are acceptable if no project file appears as modified or untracked.

---

### Task 1: Add ArtifactStore Paths for New Evaluation Outputs

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Test: `tests/test_scorecard.py`

- [ ] **Step 1: Write the failing ArtifactStore path test**

Create `tests/test_scorecard.py` with this initial content:

```python
from pathlib import Path

import pandas as pd

from pjm_forecast.workspace import ArtifactStore


def test_artifact_store_writes_scorecard_outputs(tmp_path: Path) -> None:
    store = ArtifactStore(
        directories={
            "metrics_dir": tmp_path / "metrics",
            "plots_dir": tmp_path / "plots",
            "prediction_dir": tmp_path / "predictions",
            "processed_data_dir": tmp_path / "processed",
            "hyperparameter_dir": tmp_path / "hyperparameters",
            "report_dir": tmp_path / "report",
            "artifact_dir": tmp_path,
        }
    )
    frame = pd.DataFrame([{"run": "model_test_seed7", "q50_wape": 0.25}])

    relative_path = store.write_relative_error("test", frame)
    tail_path = store.write_tail_regime_diagnostics("test", frame)
    scorecard_path = store.write_experiment_scorecard("test", frame)

    assert relative_path == tmp_path / "metrics" / "test_relative_error.csv"
    assert tail_path == tmp_path / "metrics" / "test_tail_regime_diagnostics.csv"
    assert scorecard_path == tmp_path / "metrics" / "test_experiment_scorecard.csv"
    assert pd.read_csv(scorecard_path).loc[0, "run"] == "model_test_seed7"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests\test_scorecard.py::test_artifact_store_writes_scorecard_outputs -v
```

Expected: FAIL with an `AttributeError` for `write_relative_error`.

- [ ] **Step 3: Add ArtifactStore path and writer methods**

Modify `src/pjm_forecast/workspace.py` inside `ArtifactStore`:

```python
    def relative_error(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_relative_error.csv"

    def tail_regime_diagnostics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_tail_regime_diagnostics.csv"

    def experiment_scorecard(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_experiment_scorecard.csv"
```

Add writer methods near the existing `write_regime_metrics` methods:

```python
    def write_relative_error(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.relative_error(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path

    def write_tail_regime_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.tail_regime_diagnostics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path

    def write_experiment_scorecard(self, split: str, scorecard_df: pd.DataFrame) -> Path:
        output_path = self.experiment_scorecard(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard_df.to_csv(output_path, index=False)
        return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests\test_scorecard.py::test_artifact_store_writes_scorecard_outputs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\pjm_forecast\workspace.py tests\test_scorecard.py
git commit -m "feat: add scorecard artifact paths"
```

Expected: commit succeeds.

---

### Task 2: Implement q50 Relative-Error Diagnostics

**Files:**
- Create: `src/pjm_forecast/evaluation/relative_error.py`
- Modify: `src/pjm_forecast/evaluation/__init__.py`
- Test: `tests/test_relative_error.py`

- [ ] **Step 1: Write failing tests for price-bin, month, and hour slices**

Create `tests/test_relative_error.py`:

```python
import numpy as np
import pandas as pd

from pjm_forecast.evaluation.relative_error import compute_relative_error_diagnostics


def _frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 01:00",
            "2026-02-01 00:00",
            "2026-02-01 01:00",
        ]
    )
    y_values = [20.0, 40.0, 80.0, 200.0]
    q50_values = [25.0, 30.0, 100.0, 140.0]
    rows = []
    for ds, y, q50 in zip(timestamps, y_values, q50_values, strict=True):
        rows.append({"ds": ds, "y": y, "y_pred": q50, "quantile": 0.50, "model": "m", "split": "test", "seed": 7})
        rows.append({"ds": ds, "y": y, "y_pred": q50 + 10.0, "quantile": 0.90, "model": "m", "split": "test", "seed": 7})
    return pd.DataFrame(rows)


def test_relative_error_reports_price_bins() -> None:
    result = compute_relative_error_diagnostics(_frame())
    price_bins = result.loc[result["slice_type"].eq("actual_price_bin")]

    row = price_bins.loc[price_bins["slice"].eq("20-30")].iloc[0]
    assert row["n_hours"] == 1
    assert row["q50_mae"] == 5.0
    assert np.isclose(row["wape"], 0.25)
    assert np.isclose(row["median_ape"], 0.25)


def test_relative_error_reports_month_and_hour_slices() -> None:
    result = compute_relative_error_diagnostics(_frame())
    january = result.loc[(result["slice_type"].eq("month")) & (result["slice"].eq("2026-01"))].iloc[0]
    hour_zero = result.loc[(result["slice_type"].eq("hour")) & (result["slice"].eq("0"))].iloc[0]

    assert january["n_hours"] == 2
    assert np.isclose(january["q50_mae"], 7.5)
    assert hour_zero["n_hours"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests\test_relative_error.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `pjm_forecast.evaluation.relative_error`.

- [ ] **Step 3: Implement relative-error module**

Create `src/pjm_forecast/evaluation/relative_error.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame, point_prediction_view


PRICE_BINS: list[tuple[str, float, float]] = [
    ("<=10", float("-inf"), 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100-200", 100.0, 200.0),
    (">200", 200.0, float("inf")),
]


def compute_relative_error_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    point = _q50_point_view(predictions)
    if point.empty:
        return pd.DataFrame(columns=_columns())

    point["ds"] = pd.to_datetime(point["ds"])
    point["abs_error"] = (point["y"].astype(float) - point["y_pred"].astype(float)).abs()
    denominator = point["y"].astype(float).abs().replace(0.0, np.nan)
    point["ape"] = point["abs_error"] / denominator
    smape_denominator = point["y"].astype(float).abs() + point["y_pred"].astype(float).abs()
    point["smape"] = np.where(smape_denominator == 0.0, np.nan, 2.0 * point["abs_error"] / smape_denominator)
    point["month"] = point["ds"].dt.to_period("M").astype(str)
    point["hour"] = point["ds"].dt.hour.astype(str)

    rows: list[dict[str, object]] = [_summarize("all", "all", point)]
    for label, lower, upper in PRICE_BINS:
        if np.isneginf(lower):
            subset = point.loc[point["y"].astype(float) <= upper]
        elif np.isposinf(upper):
            subset = point.loc[point["y"].astype(float) > lower]
        else:
            subset = point.loc[(point["y"].astype(float) > lower) & (point["y"].astype(float) <= upper)]
        rows.append(_summarize("actual_price_bin", label, subset))

    for month, subset in point.groupby("month", sort=True):
        rows.append(_summarize("month", str(month), subset))
    for hour, subset in point.groupby("hour", sort=True):
        rows.append(_summarize("hour", str(hour), subset))

    return pd.DataFrame(rows, columns=_columns())


def _q50_point_view(predictions: pd.DataFrame) -> pd.DataFrame:
    if not is_quantile_prediction_frame(predictions):
        return point_prediction_view(predictions).loc[:, ["ds", "y", "y_pred"]].copy()
    frame = predictions.copy()
    frame["quantile"] = frame["quantile"].astype(float)
    q50 = frame.loc[np.isclose(frame["quantile"], 0.50), ["ds", "y", "y_pred"]].copy()
    return q50.sort_values("ds").reset_index(drop=True)


def _summarize(slice_type: str, label: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "slice_type": slice_type,
            "slice": label,
            "n_hours": 0,
            "actual_mean": np.nan,
            "actual_median": np.nan,
            "q50_mae": np.nan,
            "q50_bias_mean": np.nan,
            "q50_bias_median": np.nan,
            "wape": np.nan,
            "smape": np.nan,
            "median_ape": np.nan,
            "p75_ape": np.nan,
            "p90_ape": np.nan,
            "share_ape_le_10pct": np.nan,
            "share_ape_le_25pct": np.nan,
            "share_ape_le_50pct": np.nan,
        }
    y = frame["y"].astype(float)
    y_pred = frame["y_pred"].astype(float)
    error = y - y_pred
    abs_error = frame["abs_error"].astype(float)
    ape = frame["ape"].astype(float)
    return {
        "slice_type": slice_type,
        "slice": label,
        "n_hours": int(len(frame)),
        "actual_mean": float(y.mean()),
        "actual_median": float(y.median()),
        "q50_mae": float(abs_error.mean()),
        "q50_bias_mean": float(error.mean()),
        "q50_bias_median": float(error.median()),
        "wape": float(abs_error.sum() / y.abs().sum()) if float(y.abs().sum()) > 0.0 else np.nan,
        "smape": float(frame["smape"].mean()),
        "median_ape": float(ape.median()),
        "p75_ape": float(ape.quantile(0.75)),
        "p90_ape": float(ape.quantile(0.90)),
        "share_ape_le_10pct": float((ape <= 0.10).mean()),
        "share_ape_le_25pct": float((ape <= 0.25).mean()),
        "share_ape_le_50pct": float((ape <= 0.50).mean()),
    }


def _columns() -> list[str]:
    return [
        "slice_type",
        "slice",
        "n_hours",
        "actual_mean",
        "actual_median",
        "q50_mae",
        "q50_bias_mean",
        "q50_bias_median",
        "wape",
        "smape",
        "median_ape",
        "p75_ape",
        "p90_ape",
        "share_ape_le_10pct",
        "share_ape_le_25pct",
        "share_ape_le_50pct",
    ]
```

- [ ] **Step 4: Export the function**

Modify `src/pjm_forecast/evaluation/__init__.py`:

```python
from .relative_error import compute_relative_error_diagnostics
```

Keep existing exports intact.

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests\test_relative_error.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\relative_error.py src\pjm_forecast\evaluation\__init__.py tests\test_relative_error.py
git commit -m "feat: add q50 relative error diagnostics"
```

Expected: commit succeeds.

---

### Task 3: Implement Actual-Regime Tail Diagnostics

**Files:**
- Create: `src/pjm_forecast/evaluation/tail_regime.py`
- Modify: `src/pjm_forecast/evaluation/__init__.py`
- Test: `tests/test_tail_regime.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tail_regime.py`:

```python
import numpy as np
import pandas as pd

from pjm_forecast.evaluation.tail_regime import compute_daily_peak_tail_gap, compute_tail_regime_diagnostics


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h")
    y_values = [10.0, 20.0, 30.0, 40.0, 100.0, 120.0, 200.0, 300.0]
    rows = []
    for ds, y in zip(timestamps, y_values, strict=True):
        rows.append({"ds": ds, "y": y, "y_pred": y - 1.0, "quantile": 0.50, "model": "m", "split": "test", "seed": 7})
        rows.append({"ds": ds, "y": y, "y_pred": y + 10.0, "quantile": 0.99, "model": "m", "split": "test", "seed": 7})
        rows.append({"ds": ds, "y": y, "y_pred": y + 20.0, "quantile": 0.995, "model": "m", "split": "test", "seed": 7})
    return pd.DataFrame(rows)


def test_tail_regime_reports_actual_price_segments() -> None:
    result = compute_tail_regime_diagnostics(_frame())
    assert set(result["regime"]) == {"all", "actual_le_p50", "actual_p50_p80", "actual_p80_p90", "actual_p90_p95", "actual_p95_p99", "actual_gt_p99"}
    all_row = result.loc[result["regime"].eq("all")].iloc[0]
    assert all_row["n_hours"] == 8
    assert np.isclose(all_row["q99_upper_coverage"], 1.0)


def test_daily_peak_tail_gap_reports_peak_hour_gap() -> None:
    result = compute_daily_peak_tail_gap(_frame())
    assert len(result) == 1
    row = result.iloc[0]
    assert row["actual_max"] == 300.0
    assert row["actual_peak_hour"] == 7
    assert row["peak_q99_gap"] == -10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests\test_tail_regime.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `pjm_forecast.evaluation.tail_regime`.

- [ ] **Step 3: Implement tail-regime module**

Create `src/pjm_forecast/evaluation/tail_regime.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import is_quantile_prediction_frame


def compute_tail_regime_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    grid, y_true = _quantile_grid(predictions)
    if grid.empty:
        return pd.DataFrame(columns=_tail_columns())

    q99 = _resolve_quantile(grid, 0.99)
    q995 = _resolve_quantile(grid, 0.995)
    if q99 is None or q995 is None:
        return pd.DataFrame(columns=_tail_columns())

    thresholds = {
        "p50": float(y_true.quantile(0.50)),
        "p80": float(y_true.quantile(0.80)),
        "p90": float(y_true.quantile(0.90)),
        "p95": float(y_true.quantile(0.95)),
        "p99": float(y_true.quantile(0.99)),
    }
    masks = {
        "all": pd.Series(True, index=y_true.index),
        "actual_le_p50": y_true <= thresholds["p50"],
        "actual_p50_p80": (y_true > thresholds["p50"]) & (y_true <= thresholds["p80"]),
        "actual_p80_p90": (y_true > thresholds["p80"]) & (y_true <= thresholds["p90"]),
        "actual_p90_p95": (y_true > thresholds["p90"]) & (y_true <= thresholds["p95"]),
        "actual_p95_p99": (y_true > thresholds["p95"]) & (y_true <= thresholds["p99"]),
        "actual_gt_p99": y_true > thresholds["p99"],
    }
    rows = [_summarize_tail(label, mask, y_true, grid[q99], grid[q995]) for label, mask in masks.items()]
    return pd.DataFrame(rows, columns=_tail_columns())


def compute_daily_peak_tail_gap(predictions: pd.DataFrame) -> pd.DataFrame:
    grid, y_true = _quantile_grid(predictions)
    if grid.empty:
        return pd.DataFrame(columns=_daily_columns())
    q99 = _resolve_quantile(grid, 0.99)
    q995 = _resolve_quantile(grid, 0.995)
    if q99 is None or q995 is None:
        return pd.DataFrame(columns=_daily_columns())

    frame = pd.DataFrame({"y": y_true, "q99": grid[q99], "q995": grid[q995]}, index=grid.index)
    frame["day"] = pd.DatetimeIndex(frame.index).floor("D")
    rows: list[dict[str, object]] = []
    for day, day_frame in frame.groupby("day", sort=True):
        peak_timestamp = day_frame["y"].idxmax()
        peak = day_frame.loc[peak_timestamp]
        rows.append(
            {
                "day": pd.Timestamp(day).date().isoformat(),
                "actual_max": float(peak["y"]),
                "actual_peak_hour": int(pd.Timestamp(peak_timestamp).hour),
                "peak_q99": float(peak["q99"]),
                "peak_q995": float(peak["q995"]),
                "peak_q99_gap": float(peak["y"] - peak["q99"]),
                "peak_q995_gap": float(peak["y"] - peak["q995"]),
                "daily_q99_upper_coverage": float((day_frame["y"] <= day_frame["q99"]).mean()),
                "daily_q995_upper_coverage": float((day_frame["y"] <= day_frame["q995"]).mean()),
            }
        )
    return pd.DataFrame(rows, columns=_daily_columns())


def _quantile_grid(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not is_quantile_prediction_frame(predictions):
        return pd.DataFrame(), pd.Series(dtype=float)
    frame = predictions.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["quantile"] = frame["quantile"].astype(float)
    grid = frame.pivot(index="ds", columns="quantile", values="y_pred").sort_index(axis=1)
    y_true = frame.groupby("ds", sort=True)["y"].first().reindex(grid.index).astype(float)
    return grid, y_true


def _resolve_quantile(grid: pd.DataFrame, target: float) -> float | None:
    for column in grid.columns:
        if np.isclose(float(column), target):
            return float(column)
    return None


def _summarize_tail(label: str, mask: pd.Series, y: pd.Series, q99: pd.Series, q995: pd.Series) -> dict[str, object]:
    subset_y = y.loc[mask]
    subset_q99 = q99.loc[mask]
    subset_q995 = q995.loc[mask]
    if subset_y.empty:
        return {
            "regime": label,
            "n_hours": 0,
            "actual_mean": np.nan,
            "actual_max": np.nan,
            "q99_upper_coverage": np.nan,
            "q995_upper_coverage": np.nan,
            "q99_excess_mean": np.nan,
            "q99_excess_p95": np.nan,
            "q99_excess_max": np.nan,
        }
    q99_excess = (subset_y - subset_q99).clip(lower=0.0)
    return {
        "regime": label,
        "n_hours": int(len(subset_y)),
        "actual_mean": float(subset_y.mean()),
        "actual_max": float(subset_y.max()),
        "q99_upper_coverage": float((subset_y <= subset_q99).mean()),
        "q995_upper_coverage": float((subset_y <= subset_q995).mean()),
        "q99_excess_mean": float(q99_excess.mean()),
        "q99_excess_p95": float(q99_excess.quantile(0.95)),
        "q99_excess_max": float(q99_excess.max()),
    }


def _tail_columns() -> list[str]:
    return [
        "regime",
        "n_hours",
        "actual_mean",
        "actual_max",
        "q99_upper_coverage",
        "q995_upper_coverage",
        "q99_excess_mean",
        "q99_excess_p95",
        "q99_excess_max",
    ]


def _daily_columns() -> list[str]:
    return [
        "day",
        "actual_max",
        "actual_peak_hour",
        "peak_q99",
        "peak_q995",
        "peak_q99_gap",
        "peak_q995_gap",
        "daily_q99_upper_coverage",
        "daily_q995_upper_coverage",
    ]
```

- [ ] **Step 4: Export the functions**

Modify `src/pjm_forecast/evaluation/__init__.py`:

```python
from .tail_regime import compute_daily_peak_tail_gap, compute_tail_regime_diagnostics
```

Keep existing exports intact.

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests\test_tail_regime.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\tail_regime.py src\pjm_forecast\evaluation\__init__.py tests\test_tail_regime.py
git commit -m "feat: add actual-regime tail diagnostics"
```

Expected: commit succeeds.

---

### Task 4: Build Run-Level Experiment Scorecard

**Files:**
- Create: `src/pjm_forecast/evaluation/scorecard.py`
- Modify: `tests/test_scorecard.py`

- [ ] **Step 1: Add failing scorecard composition test**

Append to `tests/test_scorecard.py`:

```python
import numpy as np

from pjm_forecast.evaluation.scorecard import build_experiment_scorecard_row


def test_scorecard_row_pulls_normal_and_tail_slices() -> None:
    relative = pd.DataFrame(
        [
            {"slice_type": "all", "slice": "all", "wape": 0.20, "smape": 0.30, "median_ape": 0.10, "p75_ape": 0.25, "p90_ape": 0.50},
            {"slice_type": "actual_price_bin", "slice": "20-30", "wape": 0.25, "smape": 0.24, "median_ape": 0.18, "p75_ape": 0.33, "p90_ape": 0.54},
            {"slice_type": "actual_price_bin", "slice": "30-50", "wape": 0.20, "smape": 0.20, "median_ape": 0.15, "p75_ape": 0.27, "p90_ape": 0.42},
        ]
    )
    tail = pd.DataFrame(
        [
            {"regime": "all", "q99_upper_coverage": 0.984, "q995_upper_coverage": 0.990, "q99_excess_mean": 0.47, "q99_excess_max": 279.0},
            {"regime": "actual_gt_p99", "q99_upper_coverage": 0.636, "q995_upper_coverage": 0.682, "q99_excess_mean": 33.0, "q99_excess_max": 279.0},
        ]
    )
    metrics = {"mae": 10.0, "rmse": 20.0, "smape": 28.0, "pinball": 3.2}

    row = build_experiment_scorecard_row(
        run_name="nhits_test_seed7",
        model="nhits",
        seed=7,
        metrics=metrics,
        relative_error=relative,
        tail_regime=tail,
    )

    assert row["run"] == "nhits_test_seed7"
    assert np.isclose(row["q50_wape_all"], 0.20)
    assert np.isclose(row["q50_wape_20_30"], 0.25)
    assert np.isclose(row["q99_coverage_gt_p99"], 0.636)
    assert np.isclose(row["q99_excess_mean_gt_p99"], 33.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests\test_scorecard.py::test_scorecard_row_pulls_normal_and_tail_slices -v
```

Expected: FAIL with `ModuleNotFoundError` for `pjm_forecast.evaluation.scorecard`.

- [ ] **Step 3: Implement scorecard module**

Create `src/pjm_forecast/evaluation/scorecard.py`:

```python
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
```

- [ ] **Step 4: Export scorecard function**

Modify `src/pjm_forecast/evaluation/__init__.py`:

```python
from .scorecard import build_experiment_scorecard_row
```

Keep existing exports intact.

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests\test_scorecard.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\scorecard.py src\pjm_forecast\evaluation\__init__.py tests\test_scorecard.py
git commit -m "feat: add experiment scorecard rows"
```

Expected: commit succeeds.

---

### Task 5: Wire New Diagnostics Into Evaluator

**Files:**
- Modify: `src/pjm_forecast/evaluation/evaluator.py`
- Modify: `src/pjm_forecast/workspace.py`
- Test: `tests/test_scorecard.py`

- [ ] **Step 1: Add protocol methods to Evaluator artifact store type**

Modify `_ArtifactStoreLike` in `src/pjm_forecast/evaluation/evaluator.py`:

```python
    def write_relative_error(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_tail_regime_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...

    def write_experiment_scorecard(self, split: str, scorecard_df: pd.DataFrame) -> Path: ...
```

- [ ] **Step 2: Import new functions**

Modify imports in `src/pjm_forecast/evaluation/evaluator.py`:

```python
from .relative_error import compute_relative_error_diagnostics
from .scorecard import build_experiment_scorecard_row
from .tail_regime import compute_daily_peak_tail_gap, compute_tail_regime_diagnostics
```

- [ ] **Step 3: Add Evaluator methods**

Add methods to `Evaluator`:

```python
    def compute_relative_error(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            diagnostics = compute_relative_error_diagnostics(run.frame)
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        output = output.sort_values(["model", "seed", "run", "slice_type", "slice"]).reset_index(drop=True)
        self.artifacts.write_relative_error(bundle.split, output)
        return output

    def compute_tail_regime_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        rows = []
        for run in bundle.runs:
            diagnostics = compute_tail_regime_diagnostics(run.frame)
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        output = output.sort_values(["model", "seed", "run", "regime"]).reset_index(drop=True)
        self.artifacts.write_tail_regime_diagnostics(bundle.split, output)
        return output

    def compute_experiment_scorecard(
        self,
        bundle: EvaluationBundle,
        metrics_df: pd.DataFrame,
        relative_error_df: pd.DataFrame,
        tail_regime_df: pd.DataFrame,
    ) -> pd.DataFrame:
        metric_lookup = metrics_df.set_index("run").to_dict("index") if not metrics_df.empty else {}
        rows = []
        for run in bundle.runs:
            relative = relative_error_df.loc[relative_error_df["run"].eq(run.name)].copy()
            tail = tail_regime_df.loc[tail_regime_df["run"].eq(run.name)].copy()
            rows.append(
                build_experiment_scorecard_row(
                    run_name=run.name,
                    model=run.model,
                    seed=run.seed,
                    metrics=metric_lookup.get(run.name, {}),
                    relative_error=relative,
                    tail_regime=tail,
                )
            )
        output = pd.DataFrame(rows).sort_values(["pinball", "model", "seed", "run"]).reset_index(drop=True)
        self.artifacts.write_experiment_scorecard(bundle.split, output)
        return output
```

- [ ] **Step 4: Wire methods into workspace evaluation flow**

Find `Workspace.evaluate` in `src/pjm_forecast/workspace.py`. After existing metrics/regime/quantile diagnostics are computed, add:

```python
        relative_error_df = evaluator.compute_relative_error(bundle)
        tail_regime_df = evaluator.compute_tail_regime_diagnostics(bundle)
        evaluator.compute_experiment_scorecard(bundle, metrics_df, relative_error_df, tail_regime_df)
```

Use the local variable names already present in `Workspace.evaluate`. If the existing method names differ, adapt only the variable names, not the design.

- [ ] **Step 5: Add integration test using a fake bundle only if existing tests do not cover `Workspace.evaluate`**

Append to `tests/test_scorecard.py`:

```python
def test_scorecard_artifact_columns_are_stable() -> None:
    expected = {
        "run",
        "model",
        "seed",
        "mae",
        "rmse",
        "smape",
        "pinball",
        "q50_wape_all",
        "q50_wape_20_30",
        "q50_wape_30_50",
        "q99_coverage_all",
        "q99_coverage_gt_p99",
        "q99_excess_mean_gt_p99",
    }
    row = build_experiment_scorecard_row(
        run_name="r",
        model="m",
        seed=7,
        metrics={"mae": 1.0, "rmse": 2.0, "smape": 3.0, "pinball": 4.0},
        relative_error=pd.DataFrame(
            [
                {"slice_type": "all", "slice": "all", "wape": 0.1, "smape": 0.2, "median_ape": 0.1, "p75_ape": 0.2, "p90_ape": 0.3},
                {"slice_type": "actual_price_bin", "slice": "20-30", "wape": 0.3, "smape": 0.4, "median_ape": 0.3, "p75_ape": 0.4, "p90_ape": 0.5},
                {"slice_type": "actual_price_bin", "slice": "30-50", "wape": 0.5, "smape": 0.6, "median_ape": 0.5, "p75_ape": 0.6, "p90_ape": 0.7},
            ]
        ),
        tail_regime=pd.DataFrame(
            [
                {"regime": "all", "q99_upper_coverage": 0.98, "q995_upper_coverage": 0.99, "q99_excess_mean": 0.1, "q99_excess_max": 1.0},
                {"regime": "actual_gt_p99", "q99_upper_coverage": 0.70, "q995_upper_coverage": 0.75, "q99_excess_mean": 5.0, "q99_excess_max": 10.0},
            ]
        ),
    )
    assert expected.issubset(row.keys())
```

- [ ] **Step 6: Run targeted tests**

Run:

```powershell
pytest tests\test_relative_error.py tests\test_tail_regime.py tests\test_scorecard.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\evaluator.py src\pjm_forecast\workspace.py tests\test_scorecard.py
git commit -m "feat: write experiment scorecards during evaluation"
```

Expected: commit succeeds.

---

### Task 6: Add Baseline Scorecard Experiment Helper

**Files:**
- Create: `scripts/experiments/scorecard_baselines.py`
- Test: `tests/test_scorecard.py`
- Docs: `README.md`

- [ ] **Step 1: Add a small pure function test for baseline command planning**

Append to `tests/test_scorecard.py`:

```python
from scripts.experiments.scorecard_baselines import baseline_model_names


def test_baseline_model_names_are_defaulted_for_comed_scorecard() -> None:
    assert baseline_model_names(None) == [
        "seasonal_naive",
        "lear",
        "lightgbm_quantile",
        "xgboost_quantile",
        "nhits_tail_grid_weighted_main",
    ]
    assert baseline_model_names("a,b") == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests\test_scorecard.py::test_baseline_model_names_are_defaulted_for_comed_scorecard -v
```

Expected: FAIL with `ModuleNotFoundError` for `scripts.experiments.scorecard_baselines`.

- [ ] **Step 3: Create script package marker if needed**

If `scripts/experiments/__init__.py` does not exist, create it as an empty file:

```python
```

- [ ] **Step 4: Implement baseline helper**

Create `scripts/experiments/scorecard_baselines.py`:

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


DEFAULT_BASELINE_MODELS = [
    "seasonal_naive",
    "lear",
    "lightgbm_quantile",
    "xgboost_quantile",
    "nhits_tail_grid_weighted_main",
]


def baseline_model_names(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_BASELINE_MODELS)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_scorecard_baselines(config_path: str, split: str, models: list[str], run_backtest: bool) -> None:
    workspace = Workspace.open(config_path)
    if run_backtest:
        workspace.backtest(split=split, model_names=models)
    workspace.evaluate(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--models", default=None, help="Comma-separated model names. Defaults to COMED scoreboard baselines.")
    parser.add_argument("--run-backtest", action="store_true", help="Run rolling backtest before evaluating existing predictions.")
    args = parser.parse_args()
    run_scorecard_baselines(
        config_path=args.config,
        split=args.split,
        models=baseline_model_names(args.models),
        run_backtest=bool(args.run_backtest),
    )


if __name__ == "__main__":
    main()
```

If `Workspace.backtest` does not currently accept `model_names`, use the existing workspace/backtest entry point and adjust this script to call the existing method signature. Do not add a new broad workflow API unless required by the current `Workspace` interface.

- [ ] **Step 5: Run targeted test**

Run:

```powershell
pytest tests\test_scorecard.py::test_baseline_model_names_are_defaulted_for_comed_scorecard -v
```

Expected: PASS.

- [ ] **Step 6: Smoke-check help output**

Run:

```powershell
python scripts\experiments\scorecard_baselines.py --help
```

Expected: prints CLI help with `--config`, `--split`, `--models`, and `--run-backtest`.

- [ ] **Step 7: Commit**

Run:

```powershell
git add scripts\experiments\scorecard_baselines.py scripts\experiments\__init__.py tests\test_scorecard.py
git commit -m "feat: add baseline scorecard helper"
```

Expected: commit succeeds.

---

### Task 7: Document Scorecard Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README section**

Add this section near the evaluation/reporting workflow:

```markdown
### COMED scorecard diagnostics

The evaluation stage writes additional scorecard artifacts for COMED model comparison:

- `artifacts_current/metrics/test_relative_error.csv`
- `artifacts_current/metrics/test_tail_regime_diagnostics.csv`
- `artifacts_current/metrics/test_experiment_scorecard.csv`

These files separate normal-price q50 relative error from upper-tail event risk. The intended first comparison set is:

- `seasonal_naive`
- `lear`
- `lightgbm_quantile`
- `xgboost_quantile`
- `nhits_tail_grid_weighted_main`

Run evaluation on existing predictions:

```powershell
python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

Run the baseline helper on existing predictions:

```powershell
python scripts\experiments\scorecard_baselines.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

Run baseline backtests and then evaluate:

```powershell
python scripts\experiments\scorecard_baselines.py --config configs\pjm_day_ahead_current_processed.yaml --split test --run-backtest
```

Use `test_experiment_scorecard.csv` for promotion decisions. Do not promote a candidate from global pinball, MAE, sMAPE, or coverage alone.
```

- [ ] **Step 2: Run docs-free targeted tests**

Run:

```powershell
pytest tests\test_relative_error.py tests\test_tail_regime.py tests\test_scorecard.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: document COMED scorecard workflow"
```

Expected: commit succeeds.

---

### Task 8: Final Verification

**Files:**
- All files touched by Tasks 1-7

- [ ] **Step 1: Run targeted evaluation tests**

Run:

```powershell
pytest tests\test_relative_error.py tests\test_tail_regime.py tests\test_scorecard.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run existing evaluation-adjacent tests**

Run:

```powershell
pytest tests\test_regime_metrics.py tests\test_config_contracts.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run full test suite if time permits**

Run:

```powershell
pytest
```

Expected: all tests PASS. Known warnings about optional dependencies or deprecated package APIs are acceptable if tests pass and warnings are not new failures.

- [ ] **Step 4: Verify generated artifact names through unit tests or a local evaluation run**

If existing predictions are present, run:

```powershell
python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

Expected new files:

```text
artifacts_current/metrics/test_relative_error.csv
artifacts_current/metrics/test_tail_regime_diagnostics.csv
artifacts_current/metrics/test_experiment_scorecard.csv
```

Do not commit generated artifact files unless the project owner explicitly asks for report assets to be committed.

- [ ] **Step 5: Check git status**

Run:

```powershell
git status --short --branch
```

Expected: clean branch after commits, or only intentionally generated untracked artifacts under `artifacts_current/`.

---

## Follow-Up Plans After This One

After this plan is complete and the scorecard is available, create separate implementation plans for:

1. `2026-05-04-comed-p50-mainline.md`  
   Feature restoration, P50-friendly model objective, and q50 point baselines.

2. `2026-05-04-comed-local-tail-expert.md`  
   Hour-level local q99/q995 residual uplift using validation OOS memory.

3. `2026-05-04-comed-vst-filter-frequency-probes.md`  
   VST grid, spike-filtered target probes, and low-risk frequency diagnostics/features.

4. `2026-05-04-comed-quantile-rag.md`  
   Quantile-aware q50 residual RAG and later local tail residual RAG.

Do not start those plans until this scorecard makes candidate comparisons reliable.

## Self-Review

- Spec coverage: this plan covers cleanup, evaluation hardening, baseline scorecard helper, tests, docs, and verification. It intentionally defers model changes and frequency/RAG/scenario work.
- Placeholder scan: no step uses unspecified placeholders for files, commands, or expected outputs.
- Type consistency: new functions use `pd.DataFrame` inputs and return `pd.DataFrame` or `dict[str, object]`, matching existing evaluation patterns.
- Scope check: this is a single implementation unit. It produces useful, testable software without requiring model retraining.

