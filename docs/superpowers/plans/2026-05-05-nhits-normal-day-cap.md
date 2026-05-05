# NHITS Normal-Day Causal Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a focused NHITS normal-day experiment that trains with causal hourly target capping and evaluates normal-day q50 relative error against ground truth.

**Architecture:** Reuse the existing spike-filter target wrapper for training. Add a small evaluation module for actual-normal-day and forecast-low-risk-day relative-error diagnostics, wire it through `Evaluator` and `ArtifactStore`, and expose compact scorecard fields for promotion review. Keep the canonical config unchanged and add a separate NHITS experiment config.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing `Workspace`, `Evaluator`, `ArtifactStore`, `SpikeFilteredTargetModel`, and NHITS adapter.

---

## File Structure

- Create `src/pjm_forecast/evaluation/normal_day.py`  
  Computes q50 relative-error diagnostics for actual normal days and forecast low-risk days.

- Modify `src/pjm_forecast/evaluation/__init__.py`  
  Exports the normal-day diagnostic function.

- Modify `src/pjm_forecast/workspace.py`  
  Adds artifact path and writer for `{split}_normal_day_diagnostics.csv`, includes it in report export, and calls evaluator wiring.

- Modify `src/pjm_forecast/evaluation/evaluator.py`  
  Adds `compute_normal_day_diagnostics(...)` and passes the result into experiment scorecard construction.

- Modify `src/pjm_forecast/evaluation/scorecard.py`  
  Adds optional normal-day fields to the run-level scorecard row while preserving callers that do not pass normal-day diagnostics.

- Modify `tests/test_scorecard.py`  
  Extends artifact and evaluator tests for the new diagnostic output and scorecard fields.

- Create `tests/test_normal_day.py`  
  Unit tests for actual-normal-day and forecast-low-risk-day relative-error diagnostics.

- Create `configs/experiments/pjm_current_validation_nhits_normal_cap.yaml`  
  NHITS-only validation experiment with `target_filter.enabled: true`.

- Modify `tests/test_model_registry_target_filter.py`  
  Adds a config-level check that the NHITS normal-cap experiment builds a `SpikeFilteredTargetModel`.

The current untracked `configs/experiments/pjm_current_validation_nhits_spike_filtered_target.yaml` is not reused by this plan.

## Task 1: Add Normal-Day Diagnostic Module

**Files:**
- Create: `src/pjm_forecast/evaluation/normal_day.py`
- Create: `tests/test_normal_day.py`
- Modify: `src/pjm_forecast/evaluation/__init__.py`

- [ ] **Step 1: Write failing tests for normal-day relative error**

Create `tests/test_normal_day.py`:

```python
import numpy as np
import pandas as pd

from pjm_forecast.evaluation.normal_day import compute_normal_day_diagnostics


def _frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 01:00",
            "2026-01-02 00:00",
            "2026-01-02 01:00",
            "2026-01-03 00:00",
            "2026-01-03 01:00",
        ]
    )
    actual = [20.0, 22.0, 30.0, 32.0, 200.0, 260.0]
    q50 = [25.0, 20.0, 27.0, 36.0, 150.0, 210.0]
    spike_score = [0.10, 0.20, 0.30, 0.25, 0.85, 0.95]
    rows = []
    for ds, y, y_pred, score in zip(timestamps, actual, q50, spike_score, strict=True):
        rows.append(
            {
                "ds": ds,
                "y": y,
                "y_pred": y_pred,
                "quantile": 0.50,
                "model": "nhits",
                "split": "validation",
                "seed": 7,
                "metadata": "{}",
                "spike_score": score,
            }
        )
        rows.append(
            {
                "ds": ds,
                "y": y,
                "y_pred": y_pred + 20.0,
                "quantile": 0.99,
                "model": "nhits",
                "split": "validation",
                "seed": 7,
                "metadata": "{}",
                "spike_score": score,
            }
        )
    return pd.DataFrame(rows)


def test_normal_day_diagnostics_prioritize_relative_error() -> None:
    diagnostics = compute_normal_day_diagnostics(
        _frame(),
        actual_daily_max_quantile=0.67,
        low_risk_score_column="spike_score",
        low_risk_threshold=0.50,
        low_risk_aggregation="mean",
    )

    actual_normal = diagnostics.loc[diagnostics["segment"].eq("actual_normal_day")].iloc[0]
    assert actual_normal["n_days"] == 2
    assert actual_normal["n_hours"] == 4
    assert actual_normal["q50_mae"] == 3.5
    assert np.isclose(actual_normal["q50_wape"], 14.0 / 104.0)
    assert actual_normal["median_ape"] < actual_normal["p90_ape"]


def test_normal_day_diagnostics_reports_forecast_low_risk_days() -> None:
    diagnostics = compute_normal_day_diagnostics(
        _frame(),
        actual_daily_max_quantile=0.67,
        low_risk_score_column="spike_score",
        low_risk_threshold=0.50,
        low_risk_aggregation="mean",
    )

    low_risk = diagnostics.loc[diagnostics["segment"].eq("forecast_low_risk_day")].iloc[0]
    high_risk = diagnostics.loc[diagnostics["segment"].eq("forecast_high_risk_day")].iloc[0]
    assert low_risk["n_days"] == 2
    assert high_risk["n_days"] == 1
    assert low_risk["q50_wape"] < high_risk["q50_wape"]


def test_normal_day_diagnostics_handles_missing_spike_score() -> None:
    frame = _frame().drop(columns=["spike_score"])

    diagnostics = compute_normal_day_diagnostics(frame, low_risk_score_column="spike_score")

    low_risk = diagnostics.loc[diagnostics["segment"].eq("forecast_low_risk_day")].iloc[0]
    assert low_risk["n_days"] == 0
    assert np.isnan(low_risk["q50_wape"])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_normal_day.py -q --basetemp=.tmp_pytest_normal_day_red
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pjm_forecast.evaluation.normal_day'`.

- [ ] **Step 3: Implement normal-day diagnostics**

Create `src/pjm_forecast/evaluation/normal_day.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import point_prediction_view


NORMAL_DAY_COLUMNS = [
    "segment",
    "n_days",
    "n_hours",
    "actual_daily_max_threshold",
    "low_risk_score_column",
    "low_risk_score_threshold",
    "actual_mean",
    "actual_median",
    "q50_mae",
    "q50_bias_mean",
    "q50_wape",
    "smape",
    "median_ape",
    "p75_ape",
    "p90_ape",
]


def compute_normal_day_diagnostics(
    predictions: pd.DataFrame,
    *,
    actual_daily_max_quantile: float = 0.95,
    low_risk_score_column: str = "spike_score",
    low_risk_threshold: float = 0.50,
    low_risk_aggregation: str = "mean",
) -> pd.DataFrame:
    if not 0.0 < float(actual_daily_max_quantile) < 1.0:
        raise ValueError("actual_daily_max_quantile must be in (0, 1).")
    if low_risk_aggregation not in {"mean", "max"}:
        raise ValueError("low_risk_aggregation must be 'mean' or 'max'.")

    point = point_prediction_view(predictions).copy()
    if point.empty:
        return pd.DataFrame(columns=NORMAL_DAY_COLUMNS)

    point["ds"] = pd.to_datetime(point["ds"])
    point["day"] = point["ds"].dt.floor("D")
    point["abs_error"] = (point["y"].astype(float) - point["y_pred"].astype(float)).abs()
    denominator = point["y"].astype(float).abs().replace(0.0, np.nan)
    point["ape"] = point["abs_error"] / denominator
    smape_denominator = point["y"].astype(float).abs() + point["y_pred"].astype(float).abs()
    point["smape"] = np.where(smape_denominator == 0.0, np.nan, 2.0 * point["abs_error"] / smape_denominator)

    daily_max = point.groupby("day", sort=True)["y"].max().astype(float)
    actual_threshold = float(daily_max.quantile(float(actual_daily_max_quantile)))
    actual_normal_days = set(daily_max.loc[daily_max <= actual_threshold].index)

    rows = [
        _summarize("all", point, actual_threshold, low_risk_score_column, low_risk_threshold),
        _summarize(
            "actual_normal_day",
            point.loc[point["day"].isin(actual_normal_days)],
            actual_threshold,
            low_risk_score_column,
            low_risk_threshold,
        ),
        _summarize(
            "actual_spike_day",
            point.loc[~point["day"].isin(actual_normal_days)],
            actual_threshold,
            low_risk_score_column,
            low_risk_threshold,
        ),
    ]

    if low_risk_score_column in point.columns:
        daily_score = point.groupby("day", sort=True)[low_risk_score_column].agg(low_risk_aggregation).astype(float)
        low_risk_days = set(daily_score.loc[daily_score <= float(low_risk_threshold)].index)
        high_risk_days = set(daily_score.index) - low_risk_days
        rows.extend(
            [
                _summarize(
                    "forecast_low_risk_day",
                    point.loc[point["day"].isin(low_risk_days)],
                    actual_threshold,
                    low_risk_score_column,
                    low_risk_threshold,
                ),
                _summarize(
                    "forecast_high_risk_day",
                    point.loc[point["day"].isin(high_risk_days)],
                    actual_threshold,
                    low_risk_score_column,
                    low_risk_threshold,
                ),
            ]
        )
    else:
        rows.extend(
            [
                _empty_row("forecast_low_risk_day", actual_threshold, low_risk_score_column, low_risk_threshold),
                _empty_row("forecast_high_risk_day", actual_threshold, low_risk_score_column, low_risk_threshold),
            ]
        )

    return pd.DataFrame(rows, columns=NORMAL_DAY_COLUMNS)


def _summarize(
    segment: str,
    frame: pd.DataFrame,
    actual_threshold: float,
    score_column: str,
    score_threshold: float,
) -> dict[str, object]:
    if frame.empty:
        return _empty_row(segment, actual_threshold, score_column, score_threshold)

    y = frame["y"].astype(float)
    y_pred = frame["y_pred"].astype(float)
    error = y - y_pred
    abs_error = frame["abs_error"].astype(float)
    ape = frame["ape"].astype(float)
    return {
        "segment": segment,
        "n_days": int(frame["day"].nunique()),
        "n_hours": int(len(frame)),
        "actual_daily_max_threshold": float(actual_threshold),
        "low_risk_score_column": score_column,
        "low_risk_score_threshold": float(score_threshold),
        "actual_mean": float(y.mean()),
        "actual_median": float(y.median()),
        "q50_mae": float(abs_error.mean()),
        "q50_bias_mean": float(error.mean()),
        "q50_wape": float(abs_error.sum() / y.abs().sum()) if float(y.abs().sum()) > 0.0 else np.nan,
        "smape": float(frame["smape"].mean()),
        "median_ape": float(ape.median()),
        "p75_ape": float(ape.quantile(0.75)),
        "p90_ape": float(ape.quantile(0.90)),
    }


def _empty_row(segment: str, actual_threshold: float, score_column: str, score_threshold: float) -> dict[str, object]:
    return {
        "segment": segment,
        "n_days": 0,
        "n_hours": 0,
        "actual_daily_max_threshold": float(actual_threshold) if not np.isnan(actual_threshold) else np.nan,
        "low_risk_score_column": score_column,
        "low_risk_score_threshold": float(score_threshold),
        "actual_mean": np.nan,
        "actual_median": np.nan,
        "q50_mae": np.nan,
        "q50_bias_mean": np.nan,
        "q50_wape": np.nan,
        "smape": np.nan,
        "median_ape": np.nan,
        "p75_ape": np.nan,
        "p90_ape": np.nan,
    }
```

- [ ] **Step 4: Export the function**

Modify `src/pjm_forecast/evaluation/__init__.py` and add this import next to the other evaluation helpers:

```python
from .normal_day import compute_normal_day_diagnostics
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_normal_day.py -q --basetemp=.tmp_pytest_normal_day_green
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\normal_day.py src\pjm_forecast\evaluation\__init__.py tests\test_normal_day.py
git commit -m "feat: add normal-day relative error diagnostics"
```

Expected: commit succeeds.

## Task 2: Wire Normal-Day Diagnostics Into Artifacts and Evaluator

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Modify: `src/pjm_forecast/evaluation/evaluator.py`
- Modify: `tests/test_scorecard.py`

- [ ] **Step 1: Extend the artifact store test first**

Modify `test_artifact_store_writes_scorecard_outputs` in `tests/test_scorecard.py`:

```python
    normal_day_path = store.write_normal_day_diagnostics("test", frame)
    relative_path = store.write_relative_error("test", frame)
    tail_path = store.write_tail_regime_diagnostics("test", frame)
    scorecard_path = store.write_experiment_scorecard("test", frame)

    assert normal_day_path == tmp_path / "metrics" / "test_normal_day_diagnostics.csv"
    assert relative_path == tmp_path / "metrics" / "test_relative_error.csv"
```

Add the writer to `_CapturingArtifacts`:

```python
    def write_normal_day_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        self.written["normal_day"] = diagnostics_df
        return Path(f"{split}_normal_day_diagnostics.csv")
```

Modify `test_evaluator_writes_scorecard_artifacts`:

```python
    normal_day = evaluator.compute_normal_day_diagnostics(bundle)
    scorecard = evaluator.compute_experiment_scorecard(bundle, metrics, relative_error, tail_regime, normal_day)

    assert set(artifacts.written) == {"metrics", "relative_error", "tail_regime", "normal_day", "scorecard"}
    assert artifacts.written["normal_day"].loc[0, "run"] == "nhits_test_seed7"
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_scorecard.py::test_artifact_store_writes_scorecard_outputs tests\test_scorecard.py::test_evaluator_writes_scorecard_artifacts -q --basetemp=.tmp_pytest_normal_day_wiring_red
```

Expected: FAIL with `AttributeError` for `write_normal_day_diagnostics` or `compute_normal_day_diagnostics`.

- [ ] **Step 3: Add ArtifactStore path and writer**

Modify `ArtifactStore` in `src/pjm_forecast/workspace.py`.

Add this path method near the other metrics paths:

```python
    def normal_day_diagnostics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_normal_day_diagnostics.csv"
```

Add this writer near the other CSV writer methods:

```python
    def write_normal_day_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.normal_day_diagnostics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path
```

Add the source to `export_report_bundle` before `relative_error(split)`:

```python
            self.normal_day_diagnostics(split),
```

- [ ] **Step 4: Add Evaluator protocol and method**

Modify imports in `src/pjm_forecast/evaluation/evaluator.py`:

```python
from .normal_day import compute_normal_day_diagnostics
```

Add to `_ArtifactStoreLike`:

```python
    def write_normal_day_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path: ...
```

Add this method to `Evaluator` after `compute_spike_score_diagnostics`:

```python
    def compute_normal_day_diagnostics(self, bundle: EvaluationBundle) -> pd.DataFrame:
        normal_cfg = self.schema.config.report.get("normal_day_evaluation", {})
        rows = []
        for run in bundle.runs:
            diagnostics = compute_normal_day_diagnostics(
                run.frame,
                actual_daily_max_quantile=float(normal_cfg.get("actual_daily_max_quantile", 0.95)),
                low_risk_score_column=str(normal_cfg.get("low_risk_score_column", "spike_score")),
                low_risk_threshold=float(normal_cfg.get("low_risk_threshold", 0.50)),
                low_risk_aggregation=str(normal_cfg.get("low_risk_aggregation", "mean")),
            )
            diagnostics.insert(0, "seed", run.seed)
            diagnostics.insert(0, "model", run.model)
            diagnostics.insert(0, "run", run.name)
            rows.append(diagnostics)
        diagnostics_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
        if not diagnostics_df.empty:
            diagnostics_df = diagnostics_df.sort_values(["model", "seed", "run", "segment"]).reset_index(drop=True)
        self.artifacts.write_normal_day_diagnostics(bundle.split, diagnostics_df)
        return diagnostics_df
```

- [ ] **Step 5: Wire Workspace.evaluate**

Modify `Workspace.evaluate` in `src/pjm_forecast/workspace.py`:

```python
        normal_day_df = evaluator.compute_normal_day_diagnostics(bundle)
        relative_error_df = evaluator.compute_relative_error(bundle)
        tail_regime_df = evaluator.compute_tail_regime_diagnostics(bundle)
        evaluator.compute_experiment_scorecard(bundle, metrics_df, relative_error_df, tail_regime_df, normal_day_df)
```

- [ ] **Step 6: Run targeted tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_normal_day.py tests\test_scorecard.py -q --basetemp=.tmp_pytest_normal_day_wiring_green
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src\pjm_forecast\workspace.py src\pjm_forecast\evaluation\evaluator.py tests\test_scorecard.py
git commit -m "feat: write normal-day diagnostics during evaluation"
```

Expected: commit succeeds.

## Task 3: Add Normal-Day Fields to Experiment Scorecard

**Files:**
- Modify: `src/pjm_forecast/evaluation/scorecard.py`
- Modify: `tests/test_scorecard.py`

- [ ] **Step 1: Add failing scorecard field test**

Append this test to `tests/test_scorecard.py`:

```python
def test_scorecard_row_pulls_normal_day_relative_error_fields() -> None:
    normal_day = pd.DataFrame(
        [
            {
                "segment": "actual_normal_day",
                "q50_wape": 0.18,
                "median_ape": 0.12,
                "p75_ape": 0.24,
                "p90_ape": 0.41,
                "smape": 0.20,
            },
            {
                "segment": "forecast_low_risk_day",
                "q50_wape": 0.16,
                "median_ape": 0.10,
                "p75_ape": 0.22,
                "p90_ape": 0.38,
                "smape": 0.18,
            },
        ]
    )

    row = build_experiment_scorecard_row(
        run_name="nhits_normal_cap_validation_seed7",
        model="nhits_normal_cap",
        seed=7,
        metrics={"mae": 8.0, "rmse": 15.0, "smape": 25.0, "pinball": 2.8},
        relative_error=pd.DataFrame(),
        tail_regime=pd.DataFrame(),
        normal_day=normal_day,
    )

    assert row["actual_normal_day_q50_wape"] == 0.18
    assert row["actual_normal_day_p75_ape"] == 0.24
    assert row["forecast_low_risk_day_q50_wape"] == 0.16
    assert row["forecast_low_risk_day_p90_ape"] == 0.38
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_scorecard.py::test_scorecard_row_pulls_normal_day_relative_error_fields -q --basetemp=.tmp_pytest_scorecard_normal_red
```

Expected: FAIL with `TypeError` because `build_experiment_scorecard_row` does not accept `normal_day`.

- [ ] **Step 3: Update scorecard row builder**

Modify the signature in `src/pjm_forecast/evaluation/scorecard.py`:

```python
def build_experiment_scorecard_row(
    *,
    run_name: str,
    model: str,
    seed: int,
    metrics: dict[str, float],
    relative_error: pd.DataFrame,
    tail_regime: pd.DataFrame,
    normal_day: pd.DataFrame | None = None,
) -> dict[str, object]:
```

Inside the function, after `row.update(_tail_fields(tail_regime))`, add:

```python
    row.update(_normal_day_fields(normal_day if normal_day is not None else pd.DataFrame()))
```

Add this helper at the bottom of the file:

```python
def _normal_day_fields(normal_day: pd.DataFrame) -> dict[str, float]:
    fields: dict[str, float] = {}
    if normal_day.empty or "segment" not in normal_day.columns:
        return fields
    mapping = {
        "actual_normal_day": "actual_normal_day",
        "forecast_low_risk_day": "forecast_low_risk_day",
    }
    for segment, prefix in mapping.items():
        match = normal_day.loc[normal_day["segment"].eq(segment)]
        if match.empty:
            continue
        record = match.iloc[0]
        fields[f"{prefix}_q50_wape"] = float(record.get("q50_wape", np.nan))
        fields[f"{prefix}_median_ape"] = float(record.get("median_ape", np.nan))
        fields[f"{prefix}_p75_ape"] = float(record.get("p75_ape", np.nan))
        fields[f"{prefix}_p90_ape"] = float(record.get("p90_ape", np.nan))
        fields[f"{prefix}_smape"] = float(record.get("smape", np.nan))
    return fields
```

- [ ] **Step 4: Update Evaluator scorecard call**

Modify `compute_experiment_scorecard` in `src/pjm_forecast/evaluation/evaluator.py`:

```python
    def compute_experiment_scorecard(
        self,
        bundle: EvaluationBundle,
        metrics_df: pd.DataFrame,
        relative_error_df: pd.DataFrame,
        tail_regime_df: pd.DataFrame,
        normal_day_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
```

Inside the loop, before `rows.append(...)`, add:

```python
            normal_day = (
                normal_day_df.loc[normal_day_df["run"].eq(run.name)]
                if normal_day_df is not None and not normal_day_df.empty
                else pd.DataFrame()
            )
```

Pass `normal_day=normal_day` into `build_experiment_scorecard_row(...)`.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_scorecard.py tests\test_normal_day.py -q --basetemp=.tmp_pytest_scorecard_normal_green
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\scorecard.py src\pjm_forecast\evaluation\evaluator.py tests\test_scorecard.py
git commit -m "feat: add normal-day fields to experiment scorecards"
```

Expected: commit succeeds.

## Task 4: Add NHITS Normal-Cap Experiment Config

**Files:**
- Create: `configs/experiments/pjm_current_validation_nhits_normal_cap.yaml`
- Modify: `tests/test_model_registry_target_filter.py`

- [ ] **Step 1: Add failing config/registry test**

Append to `tests/test_model_registry_target_filter.py`:

```python
def test_nhits_normal_cap_experiment_wraps_nhits_model() -> None:
    config = load_config("configs/experiments/pjm_current_validation_nhits_normal_cap.yaml")

    model = build_model(config, "nhits_normal_cap", seed=7)

    assert isinstance(model, SpikeFilteredTargetModel)
    assert model.filter_config.min_history == 60
    assert model.filter_config.window_observations == 365
```

If `load_config` is not imported in that file, add:

```python
from pjm_forecast.config import ProjectConfig, load_config
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_model_registry_target_filter.py::test_nhits_normal_cap_experiment_wraps_nhits_model -q --basetemp=.tmp_pytest_nhits_config_red
```

Expected: FAIL with `FileNotFoundError` for `pjm_current_validation_nhits_normal_cap.yaml`.

- [ ] **Step 3: Create NHITS normal-cap config**

Create `configs/experiments/pjm_current_validation_nhits_normal_cap.yaml` by copying `configs/pjm_day_ahead_current_processed.yaml`, then make these exact changes:

```yaml
project:
  name: "pjm_current_validation_nhits_normal_cap"
  directories:
    raw_data_dir: "../data/raw"
    processed_data_dir: "../data/processed_current"
    artifact_dir: "../artifacts_phase2/nhits_normal_cap"
    hyperparameter_dir: "../artifacts_phase2/nhits_normal_cap/hyperparameters"
    prediction_dir: "../artifacts_phase2/nhits_normal_cap/predictions"
    metrics_dir: "../artifacts_phase2/nhits_normal_cap/metrics"
    plots_dir: "../artifacts_phase2/nhits_normal_cap/plots"
    report_dir: "../artifacts_phase2/nhits_normal_cap/report"
```

Change the backtest model list:

```yaml
backtest:
  benchmark_models: ["nhits_normal_cap"]
```

Change tuning model name:

```yaml
tuning:
  model_name: "nhits_normal_cap"
```

Rename the model key and enable target filtering:

```yaml
models:
  nhits_normal_cap:
    type: "nhits"
    h: 24
    input_size: 336
    max_steps: 300
    learning_rate: 0.0005
    batch_size: 16
    dropout_prob_theta: 0.05
    scaler_type: "identity"
    target_transform: "asinh_q95"
    exog_scaler: "zscore"
    loss_name: "huber_mqloss"
    loss_delta: 0.75
    quantiles: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995]
    quantile_weights: [1.50, 1.30, 1.15, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.10, 1.30, 1.70, 2.10, 2.70, 3.20]
    quantile_deltas: [1.50, 1.25, 1.00, 0.85, 0.75, 0.75, 0.75, 0.75, 0.75, 0.85, 1.00, 1.25, 1.40, 1.60, 1.80]
    monotonicity_penalty: 0.03
    early_stop_patience_steps: 10
    val_check_steps: 50
    validation_size: 168
    windows_batch_size: 512
    ensemble_aggregation: "mean"
    ensemble_members:
      - seed_offset: 0
    stack_types: ["identity", "identity", "identity"]
    n_blocks: [1, 1, 1]
    n_pool_kernel_size: [2, 2, 1]
    n_freq_downsample: [4, 2, 1]
    mlp_units:
      - [768, 768]
      - [768, 768]
      - [768, 768]
    target_filter:
      enabled: true
      window_observations: 365
      min_history: 60
      quantile: 0.95
      fallback_quantile: 0.975
      iqr_multiplier: 3.0
```

Add normal-day evaluation config under `report`:

```yaml
report:
  normal_day_evaluation:
    actual_daily_max_quantile: 0.95
    low_risk_score_column: "spike_score"
    low_risk_threshold: 0.50
    low_risk_aggregation: "mean"
```

Keep the rest of the copied current config unchanged, including the timezone-naive time protocol.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_model_registry_target_filter.py::test_nhits_normal_cap_experiment_wraps_nhits_model -q --basetemp=.tmp_pytest_nhits_config_green
```

Expected: PASS.

- [ ] **Step 5: Validate config loading directly**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "from pjm_forecast.config import load_config; cfg=load_config(r'configs\experiments\pjm_current_validation_nhits_normal_cap.yaml'); print(cfg.project['name'], cfg.backtest['benchmark_models'])"
```

Expected output contains:

```text
pjm_current_validation_nhits_normal_cap ['nhits_normal_cap']
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add configs\experiments\pjm_current_validation_nhits_normal_cap.yaml tests\test_model_registry_target_filter.py
git commit -m "exp: add NHITS normal-day cap validation config"
```

Expected: commit succeeds.

## Task 5: Verification and Experiment Runbook

**Files:**
- No source changes unless tests uncover issues.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_normal_day.py tests\test_scorecard.py tests\test_model_registry_target_filter.py -q --basetemp=.tmp_pytest_nhits_normal_focused
```

Expected: PASS.

- [ ] **Step 2: Run full test suite with workspace-local temp**

Run:

```powershell
$env:TMP='D:\pjm_remaster\.tmp'
$env:TEMP='D:\pjm_remaster\.tmp'
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp_pytest_nhits_normal_full
```

Expected: PASS. The known `pkg_resources` deprecation warning is acceptable.

- [ ] **Step 3: Run validation backtest**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
```

Expected: writes `artifacts_phase2\nhits_normal_cap\predictions\nhits_normal_cap_validation_seed7.parquet`.

- [ ] **Step 4: Run validation evaluation**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
```

Expected: writes:

```text
artifacts_phase2\nhits_normal_cap\metrics\validation_normal_day_diagnostics.csv
artifacts_phase2\nhits_normal_cap\metrics\validation_relative_error.csv
artifacts_phase2\nhits_normal_cap\metrics\validation_experiment_scorecard.csv
```

- [ ] **Step 5: Run filter diagnostics**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\experiments\spike_filter_diagnostics.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
```

Expected: writes `artifacts_phase2\nhits_normal_cap\metrics\validation_spike_filter_diagnostics.csv`.

- [ ] **Step 6: Summarize relative-error results**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "import pandas as pd; p=r'artifacts_phase2\nhits_normal_cap\metrics\validation_experiment_scorecard.csv'; df=pd.read_csv(p); cols=['run','model','mae','pinball','actual_normal_day_q50_wape','actual_normal_day_median_ape','actual_normal_day_p75_ape','actual_normal_day_p90_ape','forecast_low_risk_day_q50_wape','forecast_low_risk_day_median_ape','forecast_low_risk_day_p75_ape','forecast_low_risk_day_p90_ape','q50_wape_20_30','q50_wape_30_50']; print(df.loc[:, [c for c in cols if c in df.columns]].to_string(index=False))"
```

Expected: table includes the normal-day and forecast-low-risk relative-error columns.

- [ ] **Step 7: Compare against current validation scorecard**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "import pandas as pd; base=pd.read_csv(r'artifacts_current\metrics\validation_experiment_scorecard.csv'); cand=pd.read_csv(r'artifacts_phase2\nhits_normal_cap\metrics\validation_experiment_scorecard.csv'); cols=['model','mae','pinball','actual_normal_day_q50_wape','forecast_low_risk_day_q50_wape','q50_wape_20_30','q50_wape_30_50']; print('BASE'); print(base.loc[:, [c for c in cols if c in base.columns]].to_string(index=False)); print('CANDIDATE'); print(cand.loc[:, [c for c in cols if c in cand.columns]].to_string(index=False))"
```

Expected: both tables print. If current validation artifacts predate normal-day diagnostics and lack those columns, rerun current evaluation after Task 2 before comparing normal-day fields.

- [ ] **Step 8: Check git status**

Run:

```powershell
git status --short --branch
```

Expected: source/config/test changes are committed. Generated artifacts remain ignored. The pre-existing untracked placeholder config may still appear unless handled separately by the user.

## Self-Review

- Spec coverage: Tasks 1-3 implement actual-normal-day and forecast-low-risk-day relative-error diagnostics and scorecard fields. Task 4 implements the NHITS-only normal-cap experiment config. Task 5 covers tests and experiment run commands.
- Placeholder scan: no step contains unresolved placeholders; each code and command block is concrete.
- Type consistency: `normal_day_df` is passed through Evaluator into `build_experiment_scorecard_row` as an optional `pd.DataFrame`, preserving existing callers.
- Scope check: the plan does not add LEAR, tree baselines, frequency features, RAG, or canonical promotion.
