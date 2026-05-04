# Spike-Filtered Normal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional spike-filtered target-training path that trains selected models on a causal capped target while preserving original `y` for prediction artifacts and evaluation.

**Architecture:** Keep the canonical feature store unchanged. Add a focused spike-filter module, wrap configured models at registry time, and run the first validation experiment on LightGBM/XGBoost quantile baselines. Add a separate diagnostics CLI so the filter can be audited without changing prediction parquet contracts.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing `ForecastModel` interface, existing `Workspace`/`ArtifactStore` workflow.

---

## File Structure

- Create `src/pjm_forecast/spike_filter.py`  
  Owns causal hour-group spike detection and target replacement. It has no dependency on model classes or workspace orchestration.
- Create `src/pjm_forecast/models/target_filter.py`  
  Owns a small `ForecastModel` wrapper. It transforms only the fit frame by replacing `y` with `y_train_clean`, then delegates to the wrapped model.
- Modify `src/pjm_forecast/models/registry.py`  
  Reads `models.<name>.target_filter` and wraps only models where `enabled: true`. Removes `target_filter` before passing model params to LightGBM/XGBoost/NHITS adapters.
- Create `src/pjm_forecast/evaluation/spike_filter_diagnostics.py`  
  Computes retrain-anchor filter diagnostics for a split using the same rolling-window protocol as backtest.
- Modify `src/pjm_forecast/workspace.py`  
  Adds an artifact path/writer for spike-filter diagnostics and a workspace method used by the CLI.
- Create `scripts/experiments/spike_filter_diagnostics.py`  
  CLI shim over `Workspace.compute_spike_filter_diagnostics(split)`.
- Create `configs/experiments/pjm_current_validation_spike_filtered_tree.yaml`  
  First validation-only experiment config comparing unfiltered and filtered LightGBM/XGBoost quantile baselines.
- Tests:
  - `tests/test_spike_filter.py`
  - `tests/test_target_filter_model.py`
  - Extend `tests/test_models.py` or create `tests/test_model_registry_target_filter.py`
  - `tests/test_spike_filter_diagnostics.py`

## Task 1: Causal Spike Filter

**Files:**
- Create: `src/pjm_forecast/spike_filter.py`
- Test: `tests/test_spike_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spike_filter.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter


def _hourly_frame(days: int = 90) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01 00:00:00")
    for offset in range(days * 24):
        ds = start + pd.Timedelta(hours=offset)
        base = 20.0 + float(ds.hour)
        rows.append({"unique_id": "PJM_COMED", "ds": ds, "y": base})
    return pd.DataFrame(rows)


def test_spike_filter_caps_only_after_minimum_prior_hour_history() -> None:
    frame = _hourly_frame(days=70)
    spike_ts = pd.Timestamp("2024-03-05 17:00:00")
    frame.loc[frame["ds"].eq(spike_ts), "y"] = 500.0

    result = apply_spike_filter(
        frame,
        SpikeFilterConfig(window_observations=365, min_history=60, quantile=0.95, iqr_multiplier=3.0),
    )

    spike_row = result.loc[result["ds"].eq(spike_ts)].iloc[0]
    assert bool(spike_row["is_training_spike"])
    assert spike_row["y_train_clean"] < spike_row["y"]
    assert spike_row["spike_residual"] == spike_row["y"] - spike_row["y_train_clean"]

    first_17 = result.loc[result["ds"].dt.hour.eq(17)].iloc[0]
    assert not bool(first_17["is_training_spike"])
    assert np.isnan(first_17["spike_threshold"])
    assert first_17["y_train_clean"] == first_17["y"]


def test_spike_filter_is_causal_with_later_extreme_values() -> None:
    prefix = _hourly_frame(days=70)
    target_ts = pd.Timestamp("2024-03-01 17:00:00")
    prefix.loc[prefix["ds"].eq(target_ts), "y"] = 120.0

    full = pd.concat(
        [
            prefix,
            pd.DataFrame(
                {
                    "unique_id": ["PJM_COMED"] * 24,
                    "ds": pd.date_range("2024-03-11 00:00:00", periods=24, freq="h"),
                    "y": [10000.0] * 24,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("ds").reset_index(drop=True)

    config = SpikeFilterConfig(window_observations=365, min_history=60)
    prefix_row = apply_spike_filter(prefix, config).loc[lambda df: df["ds"].eq(target_ts)].iloc[0]
    full_row = apply_spike_filter(full, config).loc[lambda df: df["ds"].eq(target_ts)].iloc[0]

    assert prefix_row["spike_threshold"] == full_row["spike_threshold"]
    assert prefix_row["y_train_clean"] == full_row["y_train_clean"]
    assert bool(prefix_row["is_training_spike"]) == bool(full_row["is_training_spike"])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_spike_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pjm_forecast.spike_filter'`.

- [ ] **Step 3: Implement minimal spike filter**

Create `src/pjm_forecast/spike_filter.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpikeFilterConfig:
    window_observations: int = 365
    min_history: int = 60
    quantile: float = 0.95
    fallback_quantile: float = 0.975
    iqr_multiplier: float = 3.0
    iqr_epsilon: float = 1e-8
    target_column: str = "y"


def apply_spike_filter(frame: pd.DataFrame, config: SpikeFilterConfig | None = None) -> pd.DataFrame:
    cfg = config or SpikeFilterConfig()
    required = ["ds", cfg.target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"spike filter input is missing required columns: {missing}")
    if cfg.window_observations <= 0:
        raise ValueError("window_observations must be positive.")
    if cfg.min_history <= 0:
        raise ValueError("min_history must be positive.")
    if not 0.0 < cfg.quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    if not 0.0 < cfg.fallback_quantile < 1.0:
        raise ValueError("fallback_quantile must be in (0, 1).")

    result = frame.copy()
    result["ds"] = pd.to_datetime(result["ds"])
    original_index = result.index
    result = result.sort_values("ds").reset_index(drop=False).rename(columns={"index": "_original_index"})
    result["_hour"] = result["ds"].dt.hour

    thresholds = pd.Series(np.nan, index=result.index, dtype=float)
    for _, group in result.groupby("_hour", sort=False):
        values = group[cfg.target_column].astype(float)
        shifted = values.shift(1)
        rolling = shifted.rolling(window=cfg.window_observations, min_periods=cfg.min_history)
        q_low = rolling.quantile(0.25)
        q_high = rolling.quantile(0.75)
        iqr = q_high - q_low
        robust_threshold = rolling.quantile(cfg.quantile) + cfg.iqr_multiplier * iqr
        fallback_threshold = rolling.quantile(cfg.fallback_quantile)
        threshold = robust_threshold.where(iqr > cfg.iqr_epsilon, fallback_threshold)
        thresholds.loc[group.index] = threshold

    y = result[cfg.target_column].astype(float)
    result["spike_threshold"] = thresholds
    result["is_training_spike"] = result["spike_threshold"].notna() & (y > result["spike_threshold"])
    result["y_train_clean"] = y.where(~result["is_training_spike"], result["spike_threshold"])
    result["spike_residual"] = y - result["y_train_clean"]
    result.loc[~result["is_training_spike"], "spike_residual"] = 0.0

    result = result.drop(columns=["_hour"]).set_index("_original_index").loc[original_index]
    result.index.name = None
    return result
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_spike_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\pjm_forecast\spike_filter.py tests\test_spike_filter.py
git commit -m "feat: add causal spike target filter"
```

## Task 2: Target Filter Model Wrapper

**Files:**
- Create: `src/pjm_forecast/models/target_filter.py`
- Test: `tests/test_target_filter_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_target_filter_model.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pjm_forecast.models.base import ForecastModel
from pjm_forecast.models.target_filter import SpikeFilteredTargetModel
from pjm_forecast.spike_filter import SpikeFilterConfig


class RecordingModel(ForecastModel):
    name = "recording"

    def __init__(self) -> None:
        self.fit_frame: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.fit_frame = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ds": future_df["ds"], "y_pred": 1.0})

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "RecordingModel":
        raise NotImplementedError


def test_spike_filtered_target_model_replaces_only_training_y() -> None:
    train = pd.DataFrame(
        {
            "unique_id": ["PJM_COMED"] * (70 * 24),
            "ds": pd.date_range("2024-01-01", periods=70 * 24, freq="h"),
            "y": [20.0] * (70 * 24),
        }
    )
    train.loc[train["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"] = 500.0
    future = train.tail(24).copy()
    base = RecordingModel()
    model = SpikeFilteredTargetModel(
        base_model=base,
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    model.fit(train)
    predictions = model.predict(history_df=train, future_df=future)

    assert base.fit_frame is not None
    original_spike = train.loc[train["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"].iloc[0]
    fitted_spike = base.fit_frame.loc[base.fit_frame["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"].iloc[0]
    assert fitted_spike < original_spike
    assert "y_train_clean" not in base.fit_frame.columns
    assert list(predictions.columns) == ["ds", "y_pred"]


def test_spike_filtered_target_model_exposes_last_diagnostics() -> None:
    train = pd.DataFrame(
        {
            "unique_id": ["PJM_COMED"] * (70 * 24),
            "ds": pd.date_range("2024-01-01", periods=70 * 24, freq="h"),
            "y": [20.0] * (70 * 24),
        }
    )
    train.loc[train["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"] = 500.0
    model = SpikeFilteredTargetModel(
        base_model=RecordingModel(),
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    model.fit(train)

    diagnostics = model.last_filter_diagnostics
    assert diagnostics["rows"] == float(len(train))
    assert diagnostics["spike_count"] >= 1.0
    assert diagnostics["max_spike_residual"] > 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_target_filter_model.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pjm_forecast.models.target_filter'`.

- [ ] **Step 3: Implement wrapper**

Create `src/pjm_forecast/models/target_filter.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter

from .base import ForecastModel


@dataclass
class SpikeFilteredTargetModel(ForecastModel):
    base_model: ForecastModel
    filter_config: SpikeFilterConfig = field(default_factory=SpikeFilterConfig)
    name: str = "spike_filtered_target"
    supports_fitted_snapshot: bool = False
    last_filter_diagnostics: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.name = getattr(self.base_model, "name", self.name)
        self.supports_fitted_snapshot = bool(getattr(self.base_model, "supports_fitted_snapshot", False))

    def fit(self, train_df: pd.DataFrame) -> None:
        filtered = apply_spike_filter(train_df, self.filter_config)
        self.last_filter_diagnostics = _summarize_filter(filtered)
        fit_frame = train_df.copy()
        fit_frame["y"] = filtered["y_train_clean"].astype(float)
        self.base_model.fit(fit_frame)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        return self.base_model.predict(history_df=history_df, future_df=future_df)

    def save(self, path: Path) -> None:
        self.base_model.save(path)

    @classmethod
    def load(cls, path: Path) -> "SpikeFilteredTargetModel":
        raise NotImplementedError("Load the wrapped base model through its own adapter.")


def _summarize_filter(filtered: pd.DataFrame) -> dict[str, float]:
    spike_mask = filtered["is_training_spike"].astype(bool)
    residual = filtered["spike_residual"].astype(float)
    rows = float(len(filtered))
    spike_count = float(spike_mask.sum())
    return {
        "rows": rows,
        "spike_count": spike_count,
        "spike_share": 0.0 if rows == 0.0 else spike_count / rows,
        "mean_spike_residual": 0.0 if spike_count == 0.0 else float(residual.loc[spike_mask].mean()),
        "max_spike_residual": 0.0 if spike_count == 0.0 else float(residual.loc[spike_mask].max()),
    }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_target_filter_model.py tests\test_spike_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\pjm_forecast\models\target_filter.py tests\test_target_filter_model.py
git commit -m "feat: add spike-filtered target model wrapper"
```

## Task 3: Registry Configuration Support

**Files:**
- Modify: `src/pjm_forecast/models/registry.py`
- Test: `tests/test_model_registry_target_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_registry_target_filter.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from pjm_forecast.config import ProjectConfig
from pjm_forecast.models.registry import build_model
from pjm_forecast.models.target_filter import SpikeFilteredTargetModel


def _config_with_filtered_lightgbm() -> ProjectConfig:
    raw = yaml.safe_load(Path("configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml").read_text(encoding="utf-8"))
    raw["models"]["lightgbm_q_filtered"] = dict(raw["models"]["lightgbm_q"])
    raw["models"]["lightgbm_q_filtered"]["target_filter"] = {
        "enabled": True,
        "window_observations": 365,
        "min_history": 60,
        "quantile": 0.95,
        "fallback_quantile": 0.975,
        "iqr_multiplier": 3.0,
    }
    return ProjectConfig(raw=raw, path=Path("configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml").resolve())


def test_registry_wraps_model_when_target_filter_is_enabled() -> None:
    config = _config_with_filtered_lightgbm()

    model = build_model(config, "lightgbm_q_filtered", seed=7)

    assert isinstance(model, SpikeFilteredTargetModel)
    assert model.filter_config.min_history == 60
    assert model.filter_config.window_observations == 365


def test_registry_leaves_model_unwrapped_when_target_filter_is_absent() -> None:
    config = _config_with_filtered_lightgbm()

    model = build_model(config, "lightgbm_q", seed=7)

    assert not isinstance(model, SpikeFilteredTargetModel)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_model_registry_target_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: FAIL because `lightgbm_q_filtered` is not wrapped, or because `target_filter` leaks into the LightGBM estimator params.

- [ ] **Step 3: Implement registry wrapping**

Modify `src/pjm_forecast/models/registry.py`:

```python
from .target_filter import SpikeFilteredTargetModel
from pjm_forecast.spike_filter import SpikeFilterConfig
```

Inside `build_model`, after `model_cfg` is resolved and before `model_type = model_cfg.pop("type")`, add:

```python
    target_filter_cfg = model_cfg.pop("target_filter", {}) or {}
```

Replace every direct `return SomeModel(...)` in the function with assignment to `model`, then return through:

```python
    return _maybe_wrap_target_filter(model, target_filter_cfg)
```

Add this helper at the bottom of the file:

```python
def _maybe_wrap_target_filter(model, target_filter_cfg: dict):
    if not bool(target_filter_cfg.get("enabled", False)):
        return model
    filter_config = SpikeFilterConfig(
        window_observations=int(target_filter_cfg.get("window_observations", 365)),
        min_history=int(target_filter_cfg.get("min_history", 60)),
        quantile=float(target_filter_cfg.get("quantile", 0.95)),
        fallback_quantile=float(target_filter_cfg.get("fallback_quantile", 0.975)),
        iqr_multiplier=float(target_filter_cfg.get("iqr_multiplier", 3.0)),
    )
    return SpikeFilteredTargetModel(base_model=model, filter_config=filter_config)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_model_registry_target_filter.py tests\test_target_filter_model.py tests\test_spike_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\pjm_forecast\models\registry.py tests\test_model_registry_target_filter.py
git commit -m "feat: wire spike target filter into model registry"
```

## Task 4: Filter Diagnostics CLI

**Files:**
- Create: `src/pjm_forecast/evaluation/spike_filter_diagnostics.py`
- Modify: `src/pjm_forecast/workspace.py`
- Create: `scripts/experiments/spike_filter_diagnostics.py`
- Test: `tests/test_spike_filter_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spike_filter_diagnostics.py`:

```python
from __future__ import annotations

import pandas as pd

from pjm_forecast.evaluation.spike_filter_diagnostics import compute_retrain_spike_filter_diagnostics
from pjm_forecast.spike_filter import SpikeFilterConfig


def _feature_frame(days: int = 90) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["PJM_COMED"] * (days * 24),
            "ds": pd.date_range("2024-01-01", periods=days * 24, freq="h"),
            "y": [20.0] * (days * 24),
        }
    )
    frame.loc[frame["ds"].eq(pd.Timestamp("2024-03-05 17:00:00")), "y"] = 500.0
    return frame


def test_retrain_spike_filter_diagnostics_reports_anchor_windows() -> None:
    feature_df = _feature_frame()
    forecast_days = [
        pd.Timestamp("2024-03-11 00:00:00"),
        pd.Timestamp("2024-03-12 00:00:00"),
        pd.Timestamp("2024-03-18 00:00:00"),
    ]

    diagnostics = compute_retrain_spike_filter_diagnostics(
        feature_df=feature_df,
        forecast_days=forecast_days,
        rolling_window_days=80,
        retrain_weekday=0,
        filter_config=SpikeFilterConfig(window_observations=365, min_history=60),
    )

    assert list(diagnostics["forecast_day"]) == [pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-18")]
    assert diagnostics["rows"].gt(0).all()
    assert diagnostics["spike_count"].ge(1).any()
    assert diagnostics["max_spike_residual"].gt(0).any()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_spike_filter_diagnostics.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pjm_forecast.evaluation.spike_filter_diagnostics'`.

- [ ] **Step 3: Implement diagnostics module**

Create `src/pjm_forecast/evaluation/spike_filter_diagnostics.py`:

```python
from __future__ import annotations

import pandas as pd

from pjm_forecast.spike_filter import SpikeFilterConfig, apply_spike_filter


def compute_retrain_spike_filter_diagnostics(
    *,
    feature_df: pd.DataFrame,
    forecast_days: list[pd.Timestamp],
    rolling_window_days: int,
    retrain_weekday: int,
    filter_config: SpikeFilterConfig,
) -> pd.DataFrame:
    rows = []
    for index, forecast_day in enumerate(forecast_days):
        if index > 0 and forecast_day.weekday() != retrain_weekday:
            continue
        history_end = forecast_day - pd.Timedelta(hours=1)
        window_start = forecast_day - pd.Timedelta(days=rolling_window_days)
        history_df = feature_df.loc[(feature_df["ds"] >= window_start) & (feature_df["ds"] <= history_end)].copy()
        filtered = apply_spike_filter(history_df, filter_config)
        spike_mask = filtered["is_training_spike"].astype(bool)
        residual = filtered["spike_residual"].astype(float)
        rows.append(
            {
                "forecast_day": forecast_day.normalize(),
                "window_start": window_start,
                "window_end": history_end,
                "rows": len(filtered),
                "spike_count": int(spike_mask.sum()),
                "spike_share": 0.0 if len(filtered) == 0 else float(spike_mask.mean()),
                "mean_spike_residual": 0.0 if not spike_mask.any() else float(residual.loc[spike_mask].mean()),
                "max_spike_residual": 0.0 if not spike_mask.any() else float(residual.loc[spike_mask].max()),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Wire workspace and CLI**

Modify `src/pjm_forecast/workspace.py` imports:

```python
from .evaluation.spike_filter_diagnostics import compute_retrain_spike_filter_diagnostics
from .spike_filter import SpikeFilterConfig
```

Add to `ArtifactStore`:

```python
    def spike_filter_diagnostics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_spike_filter_diagnostics.csv"

    def write_spike_filter_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.spike_filter_diagnostics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path
```

Add to `Workspace`:

```python
    def compute_spike_filter_diagnostics(self, split: SplitName = "validation") -> Path:
        model_name = str(self.config.backtest["benchmark_models"][0])
        target_filter_cfg = self.config.models.get(model_name, {}).get("target_filter", {}) or {}
        filter_config = SpikeFilterConfig(
            window_observations=int(target_filter_cfg.get("window_observations", 365)),
            min_history=int(target_filter_cfg.get("min_history", 60)),
            quantile=float(target_filter_cfg.get("quantile", 0.95)),
            fallback_quantile=float(target_filter_cfg.get("fallback_quantile", 0.975)),
            iqr_multiplier=float(target_filter_cfg.get("iqr_multiplier", 3.0)),
        )
        diagnostics = compute_retrain_spike_filter_diagnostics(
            feature_df=self.feature_frame(),
            forecast_days=self.split_days(split),
            rolling_window_days=int(self.config.backtest["rolling_window_days"]),
            retrain_weekday=int(self.config.backtest["retrain_weekday"]),
            filter_config=filter_config,
        )
        return self.artifacts.write_spike_filter_diagnostics(split, diagnostics)
```

Create `scripts/experiments/spike_filter_diagnostics.py`:

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    output_path = Workspace.open(args.config).compute_spike_filter_diagnostics(split=args.split)
    print(f"Wrote spike-filter diagnostics to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_spike_filter_diagnostics.py tests\test_spike_filter.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\pjm_forecast\evaluation\spike_filter_diagnostics.py src\pjm_forecast\workspace.py scripts\experiments\spike_filter_diagnostics.py tests\test_spike_filter_diagnostics.py
git commit -m "feat: add spike filter diagnostics"
```

## Task 5: Validation Experiment Config

**Files:**
- Create: `configs/experiments/pjm_current_validation_spike_filtered_tree.yaml`

- [ ] **Step 1: Create config from benchmark floor**

Copy `configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml` to `configs/experiments/pjm_current_validation_spike_filtered_tree.yaml`.

Make these exact changes:

```yaml
project:
  name: "pjm_current_validation_spike_filtered_tree"
  directories:
    raw_data_dir: "../data/raw"
    processed_data_dir: "../data/processed_phase1_benchmark_floor"
    artifact_dir: "../artifacts_phase2/spike_filtered_tree"
    hyperparameter_dir: "../artifacts_phase2/spike_filtered_tree/hyperparameters"
    prediction_dir: "../artifacts_phase2/spike_filtered_tree/predictions"
    metrics_dir: "../artifacts_phase2/spike_filtered_tree/metrics"
    plots_dir: "../artifacts_phase2/spike_filtered_tree/plots"
    report_dir: "../artifacts_phase2/spike_filtered_tree/report"

backtest:
  benchmark_models: ["lightgbm_q", "lightgbm_q_filtered", "xgboost_q", "xgboost_q_filtered"]

models:
  lightgbm_q_filtered:
    type: "lightgbm_quantile"
    loss_name: "mqloss"
    quantiles: [0.10, 0.50, 0.90]
    n_estimators: 200
    learning_rate: 0.05
    num_leaves: 31
    min_child_samples: 40
    subsample: 0.9
    colsample_bytree: 0.9
    reg_lambda: 0.0
    n_jobs: -1
    target_filter:
      enabled: true
      window_observations: 365
      min_history: 60
      quantile: 0.95
      fallback_quantile: 0.975
      iqr_multiplier: 3.0
  xgboost_q_filtered:
    type: "xgboost_quantile"
    loss_name: "mqloss"
    quantiles: [0.10, 0.50, 0.90]
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 6
    min_child_weight: 4.0
    subsample: 0.9
    colsample_bytree: 0.9
    reg_lambda: 1.0
    n_jobs: -1
    target_filter:
      enabled: true
      window_observations: 365
      min_history: 60
      quantile: 0.95
      fallback_quantile: 0.975
      iqr_multiplier: 3.0
```

Do not use YAML anchors if the repo's config loader or future edits would make the file less explicit. Duplicate the model keys.

- [ ] **Step 2: Validate config loading**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "from pjm_forecast.config import load_config; load_config(r'configs\experiments\pjm_current_validation_spike_filtered_tree.yaml'); print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

Run:

```powershell
git add configs\experiments\pjm_current_validation_spike_filtered_tree.yaml
git commit -m "exp: add spike-filtered tree validation config"
```

## Task 6: Unit Verification and Validation Rerun

**Files:**
- No source files unless fixes are required.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_spike_filter.py tests\test_target_filter_model.py tests\test_model_registry_target_filter.py tests\test_spike_filter_diagnostics.py tests\test_scorecard.py -q -o cache_dir=.tmp\.pytest_cache
```

Expected: PASS. If pytest emits cache permission warnings only, record them but do not treat them as code failures.

- [ ] **Step 2: Clear stale experiment artifacts**

Before the formal validation run, remove only this experiment's generated artifact directories:

```powershell
Remove-Item -Recurse -Force artifacts_phase2\spike_filtered_tree\predictions, artifacts_phase2\spike_filtered_tree\metrics, artifacts_phase2\spike_filtered_tree\plots -ErrorAction SilentlyContinue
```

Verify the resolved targets are inside `D:\pjm_remaster\artifacts_phase2\spike_filtered_tree` before deleting.

- [ ] **Step 3: Run validation backtest**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_spike_filtered_tree.yaml --split validation
```

Expected: writes validation prediction parquet files under `artifacts_phase2\spike_filtered_tree\predictions`.

- [ ] **Step 4: Run evaluation**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_spike_filtered_tree.yaml --split validation
```

Expected: writes `artifacts_phase2\spike_filtered_tree\metrics\validation_experiment_scorecard.csv`.

- [ ] **Step 5: Run filter diagnostics**

Run:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\experiments\spike_filter_diagnostics.py --config configs\experiments\pjm_current_validation_spike_filtered_tree.yaml --split validation
```

Expected: writes `artifacts_phase2\spike_filtered_tree\metrics\validation_spike_filter_diagnostics.csv`.

- [ ] **Step 6: Summarize results**

Read:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "import pandas as pd; p='artifacts_phase2/spike_filtered_tree/metrics/validation_experiment_scorecard.csv'; print(pd.read_csv(p).loc[:, ['model','mae','smape','pinball','wape_all','wape_10_20','wape_20_30','wape_30_50','wape_50_100']].to_string(index=False))"
```

Read:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -c "import pandas as pd; p='artifacts_phase2/spike_filtered_tree/metrics/validation_spike_filter_diagnostics.csv'; df=pd.read_csv(p); print(df[['spike_count','spike_share','mean_spike_residual','max_spike_residual']].describe().to_string())"
```

Compare filtered vs unfiltered tree rows. Record whether filtered training improves ordinary bins and whether pinball worsens materially.

- [ ] **Step 7: Commit final implementation state**

If all implementation commits are already present and only docs/results notes changed, commit those notes. Do not commit generated predictions, metrics, plots, or large artifacts.

Run:

```powershell
git status --short
```

Expected: generated artifact files are untracked or ignored; source/config/doc changes are committed.
