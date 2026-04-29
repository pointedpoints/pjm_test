# Scenario Generation And Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable flow that turns quantile forecasts into full daily price paths, then keeps a smaller set of representative paths with probabilities.

**Architecture:** Keep the existing quantile forecast flow unchanged. Add a new scenario layer after evaluation: fit the dependence pattern on validation predictions, generate many 24-hour paths for each forecast day, reduce those paths to fewer representative paths, and write clear diagnostics. `Workspace` remains the workflow boundary and artifact owner.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, pytest, existing `pjm_forecast.copula`, existing `Workspace`, existing pipeline wrapper.

---

## Plain-Language Outcome

After this work, a user can run the pipeline and get:

- A raw scenario file: many possible 24-hour price paths per day.
- A reduced scenario file: fewer paths per day, each with a probability.
- A diagnostics file: checks that the reduced paths still look like the raw paths for daily max, daily spread, ramp, and average price.
- A config section that controls how many raw paths to generate and how many reduced paths to keep.

The important rule is no data leakage: validation can be used to learn the sampling pattern and choose settings; test is only used as the target period where the already chosen method is applied and evaluated.

## Scope

This plan does:

- Add Latin Hypercube Sampling as a lower-noise way to generate paths from the existing copula code.
- Add raw scenario artifacts.
- Add reduced scenario artifacts.
- Add a simple Wasserstein-style reduction based on distances between full daily paths.
- Add tests and docs.
- Add pipeline stages after `evaluate_and_plot`.

This plan does not:

- Change NHITS training.
- Change the v1 local-time protocol.
- Tune prediction quality.
- Add external optimization libraries.
- Replace the current scalar scenario diagnostics.

## File Structure

- Modify `src/pjm_forecast/copula.py`
  - Add `sampling_method="mc" | "lhs"` support.
  - Keep existing random Monte Carlo behavior as the default.

- Create `src/pjm_forecast/scenarios/generation.py`
  - Read validation and target predictions.
  - Fit the copula on validation.
  - Generate daily raw paths for the target split.
  - Return a long DataFrame that can be written to parquet.

- Create `src/pjm_forecast/scenarios/reduction.py`
  - Reduce raw paths day by day.
  - Use medoids, meaning kept paths are real generated paths.
  - Assign each raw path to the closest kept path and turn cluster sizes into probabilities.

- Create `src/pjm_forecast/scenarios/diagnostics.py`
  - Compare raw and reduced paths.
  - Measure whether daily max, spread, ramp, and mean were preserved.

- Create `src/pjm_forecast/scenarios/__init__.py`
  - Export the small public API for the scenario package.

- Modify `src/pjm_forecast/workspace.py`
  - Add artifact paths for raw, reduced, and diagnostics files.
  - Add `generate_scenarios(split="test")`.
  - Add `reduce_scenarios(split="test")`.
  - Add `diagnose_scenarios(split="test")`.

- Modify `src/pjm_forecast/pipeline.py`
  - Add stages after `evaluate_and_plot` and before quality finalization.

- Modify `src/pjm_forecast/config.py`
  - Validate `report.scenario_generation`.

- Modify `configs/pjm_day_ahead_current_processed.yaml`
  - Add default scenario generation settings.

- Create CLI shims:
  - `scripts/generate_scenarios.py`
  - `scripts/reduce_scenarios.py`
  - `scripts/diagnose_scenarios.py`

- Modify docs:
  - `README.md`
  - `docs/protocol/canonical_release_checklist.md`

- Tests:
  - Modify `tests/test_copula.py`
  - Create `tests/test_scenario_generation.py`
  - Create `tests/test_scenario_reduction.py`
  - Create `tests/test_scenario_diagnostics.py`
  - Modify `tests/test_workspace.py`
  - Modify `tests/test_config_contracts.py`

---

### Task 1: Add LHS Sampling To Existing Copula Code

**Files:**
- Modify: `src/pjm_forecast/copula.py`
- Test: `tests/test_copula.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_copula.py`:

```python
def test_gaussian_copula_lhs_sample_is_reproducible() -> None:
    uniforms = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.7],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    copula = GaussianCopula.fit(uniforms)

    first = copula.sample(16, random_state=7, sampling_method="lhs")
    second = copula.sample(16, random_state=7, sampling_method="lhs")

    assert first.shape == (16, 2)
    assert np.allclose(first, second)
    assert np.all(first > 0.0)
    assert np.all(first < 1.0)


def test_student_t_copula_lhs_sample_is_reproducible() -> None:
    uniforms = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.7],
            [0.4, 0.5],
            [0.6, 0.65],
        ],
        dtype=float,
    )
    copula = StudentTCopula.fit(uniforms, dof_grid=[3.0, 5.0])

    first = copula.sample(16, random_state=7, sampling_method="lhs")
    second = copula.sample(16, random_state=7, sampling_method="lhs")

    assert first.shape == (16, 2)
    assert np.allclose(first, second)
    assert np.all(first > 0.0)
    assert np.all(first < 1.0)


def test_copula_rejects_unknown_sampling_method() -> None:
    copula = GaussianCopula.fit(np.array([[0.2, 0.3], [0.8, 0.7], [0.5, 0.5]], dtype=float))

    with pytest.raises(ValueError, match="Unsupported sampling_method"):
        copula.sample(8, random_state=7, sampling_method="bad")
```

Also add `import pytest` at the top of the file if it is not present.

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_copula.py::test_gaussian_copula_lhs_sample_is_reproducible tests\test_copula.py::test_student_t_copula_lhs_sample_is_reproducible tests\test_copula.py::test_copula_rejects_unknown_sampling_method -q
```

Expected: fail because `sample()` does not accept `sampling_method`.

- [ ] **Step 3: Implement LHS helper functions**

Add these helpers near the bottom of `src/pjm_forecast/copula.py`:

```python
def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    return random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)


def _validate_sampling_method(sampling_method: str) -> str:
    method = str(sampling_method).lower()
    if method not in {"mc", "lhs"}:
        raise ValueError(f"Unsupported sampling_method={sampling_method!r}.")
    return method


def _lhs_unit_cube(n_samples: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    result = np.empty((n_samples, dimension), dtype=float)
    for column in range(dimension):
        order = rng.permutation(n_samples)
        jitter = rng.uniform(0.0, 1.0, size=n_samples)
        result[:, column] = (order + jitter) / float(n_samples)
    return _clip_unit_interval(result)


def _correlate_latent_independent_samples(independent_latent: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(_nearest_correlation(correlation))
    return np.asarray(independent_latent, dtype=float) @ chol.T
```

- [ ] **Step 4: Update `BaseCopula.sample` signature**

Change the method signature in `BaseCopula`:

```python
def sample(
    self,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
    sampling_method: str = "mc",
) -> np.ndarray:
    raise NotImplementedError
```

- [ ] **Step 5: Update `GaussianCopula.sample`**

Replace the current method body with:

```python
def sample(
    self,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
    sampling_method: str = "mc",
) -> np.ndarray:
    method = _validate_sampling_method(sampling_method)
    rng = _rng(random_state)
    if method == "lhs":
        independent = stats.norm.ppf(_lhs_unit_cube(n_samples, self.dimension, rng))
        latent = _correlate_latent_independent_samples(independent, self.correlation)
    else:
        latent = rng.multivariate_normal(np.zeros(self.dimension), self.correlation, size=n_samples)
    return _clip_unit_interval(stats.norm.cdf(latent))
```

- [ ] **Step 6: Update `StudentTCopula.sample`**

Replace the current method body with:

```python
def sample(
    self,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
    sampling_method: str = "mc",
) -> np.ndarray:
    method = _validate_sampling_method(sampling_method)
    rng = _rng(random_state)
    if method == "lhs":
        independent = stats.t.ppf(_lhs_unit_cube(n_samples, self.dimension, rng), df=self.degrees_of_freedom)
        latent = _correlate_latent_independent_samples(independent, self.correlation)
    else:
        samples = stats.multivariate_t(
            loc=np.zeros(self.dimension),
            shape=self.correlation,
            df=self.degrees_of_freedom,
        ).rvs(size=n_samples, random_state=rng)
        latent = np.atleast_2d(samples)
    return _clip_unit_interval(stats.t.cdf(latent, df=self.degrees_of_freedom))
```

- [ ] **Step 7: Update `ScenarioMarginals.sample_paths` and `sample_copula_scenarios`**

Change both functions to pass through `sampling_method`.

```python
def sample_paths(
    self,
    copula: BaseCopula,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
    sampling_method: str = "mc",
) -> np.ndarray:
    uniforms = copula.sample(n_samples, random_state=random_state, sampling_method=sampling_method)
    return self.ppf(uniforms)
```

```python
def sample_copula_scenarios(
    copula: BaseCopula,
    marginals: ScenarioMarginals,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
    sampling_method: str = "mc",
) -> np.ndarray:
    return marginals.sample_paths(
        copula,
        n_samples,
        random_state=random_state,
        sampling_method=sampling_method,
    )
```

- [ ] **Step 8: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_copula.py -q
```

Expected: all `test_copula.py` tests pass.

- [ ] **Step 9: Commit**

```powershell
git add src\pjm_forecast\copula.py tests\test_copula.py
git commit -m "Add LHS copula sampling"
```

---

### Task 2: Add Raw Scenario Generation

**Files:**
- Create: `src/pjm_forecast/scenarios/__init__.py`
- Create: `src/pjm_forecast/scenarios/generation.py`
- Test: `tests/test_scenario_generation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenario_generation.py`:

```python
from __future__ import annotations

import pandas as pd

from pjm_forecast.prepared_data import prediction_metadata
from pjm_forecast.scenarios.generation import generate_raw_scenarios


def _prediction_frame(split: str) -> pd.DataFrame:
    rows = []
    days = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    for day_index, forecast_day in enumerate(days):
        metadata = prediction_metadata(forecast_day)
        for hour in range(3):
            ds = forecast_day + pd.Timedelta(hours=hour)
            center = 20.0 + day_index + hour
            for quantile, offset in [(0.1, -2.0), (0.5, 0.0), (0.9, 2.0)]:
                rows.append(
                    {
                        "ds": ds,
                        "y": center,
                        "y_pred": center + offset,
                        "model": "nhits_tail_grid_weighted_main",
                        "split": split,
                        "seed": 7,
                        "quantile": quantile,
                        "metadata": metadata,
                    }
                )
    return pd.DataFrame(rows)


def test_generate_raw_scenarios_writes_one_row_per_day_scenario_hour() -> None:
    output = generate_raw_scenarios(
        source_predictions=_prediction_frame("validation"),
        target_predictions=_prediction_frame("test"),
        run_name="nhits_tail_grid_weighted_main_test_seed7",
        split="test",
        family="student_t",
        dof_grid=[3.0, 5.0],
        tail_policy="linear",
        sampling_method="lhs",
        n_scenarios=8,
        random_seed=7,
    )

    assert len(output) == 3 * 8 * 3
    assert set(output["sampling_method"]) == {"lhs"}
    assert set(output["scenario_kind"]) == {"raw"}
    assert output["probability"].nunique() == 1
    assert abs(float(output["probability"].iloc[0]) - 0.125) < 1e-12
    assert output.groupby(["forecast_day", "scenario_id"]).size().eq(3).all()


def test_generate_raw_scenarios_is_reproducible() -> None:
    first = generate_raw_scenarios(
        source_predictions=_prediction_frame("validation"),
        target_predictions=_prediction_frame("test"),
        run_name="nhits_tail_grid_weighted_main_test_seed7",
        split="test",
        family="gaussian",
        tail_policy="linear",
        sampling_method="lhs",
        n_scenarios=8,
        random_seed=7,
    )
    second = generate_raw_scenarios(
        source_predictions=_prediction_frame("validation"),
        target_predictions=_prediction_frame("test"),
        run_name="nhits_tail_grid_weighted_main_test_seed7",
        split="test",
        family="gaussian",
        tail_policy="linear",
        sampling_method="lhs",
        n_scenarios=8,
        random_seed=7,
    )

    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_scenario_generation.py -q
```

Expected: fail because `pjm_forecast.scenarios.generation` does not exist.

- [ ] **Step 3: Create package init**

Create `src/pjm_forecast/scenarios/__init__.py`:

```python
from __future__ import annotations

from .generation import generate_raw_scenarios

__all__ = ["generate_raw_scenarios"]
```

- [ ] **Step 4: Implement raw scenario generation**

Create `src/pjm_forecast/scenarios/generation.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.copula import build_quantile_surface_panel, fit_copula_from_predictions, sample_copula_scenarios
from pjm_forecast.prediction_contract import is_quantile_prediction_frame


def generate_raw_scenarios(
    *,
    source_predictions: pd.DataFrame,
    target_predictions: pd.DataFrame,
    run_name: str,
    split: str,
    family: str = "student_t",
    dof_grid: list[float] | None = None,
    tail_policy: str = "linear",
    sampling_method: str = "lhs",
    n_scenarios: int = 1024,
    random_seed: int = 7,
) -> pd.DataFrame:
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    if not is_quantile_prediction_frame(source_predictions):
        raise ValueError("source_predictions must contain quantile predictions.")
    if not is_quantile_prediction_frame(target_predictions):
        raise ValueError("target_predictions must contain quantile predictions.")

    copula, source_panel = fit_copula_from_predictions(
        source_predictions,
        family=family,
        dof_grid=dof_grid,
        tail_policy=tail_policy,
    )
    target_panel = build_quantile_surface_panel(target_predictions, tail_policy=tail_policy)
    rows: list[dict[str, object]] = []
    probability = 1.0 / float(n_scenarios)

    for day_index, forecast_day in enumerate(target_panel.forecast_days):
        marginals = target_panel.marginals_for_day(forecast_day)
        paths = sample_copula_scenarios(
            copula,
            marginals,
            n_scenarios,
            random_state=int(random_seed) + int(day_index),
            sampling_method=sampling_method,
        )
        for scenario_id in range(n_scenarios):
            for horizon, ds in enumerate(marginals.ds_index):
                rows.append(
                    {
                        "run": run_name,
                        "split": split,
                        "forecast_day": pd.Timestamp(forecast_day),
                        "scenario_id": int(scenario_id),
                        "scenario_kind": "raw",
                        "source_scenario_id": int(scenario_id),
                        "ds": pd.Timestamp(ds),
                        "horizon": int(horizon),
                        "y_scenario": float(paths[scenario_id, horizon]),
                        "probability": probability,
                        "sampling_method": str(sampling_method).lower(),
                        "copula_family": "student_t" if family.lower() in {"student_t", "t"} else "gaussian",
                        "tail_policy": str(tail_policy).lower(),
                        "random_seed": int(random_seed),
                        "source_train_days": int(len(source_panel.forecast_days)),
                    }
                )

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_scenario_generation.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\scenarios\__init__.py src\pjm_forecast\scenarios\generation.py tests\test_scenario_generation.py
git commit -m "Add raw scenario generation"
```

---

### Task 3: Add Scenario Reduction

**Files:**
- Create: `src/pjm_forecast/scenarios/reduction.py`
- Modify: `src/pjm_forecast/scenarios/__init__.py`
- Test: `tests/test_scenario_reduction.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenario_reduction.py`:

```python
from __future__ import annotations

import pandas as pd

from pjm_forecast.scenarios.reduction import reduce_scenarios


def _raw_scenarios() -> pd.DataFrame:
    rows = []
    forecast_day = pd.Timestamp("2026-01-01")
    paths = {
        0: [10.0, 11.0, 12.0],
        1: [10.1, 11.1, 12.1],
        2: [20.0, 22.0, 24.0],
        3: [19.9, 21.9, 23.9],
    }
    for scenario_id, values in paths.items():
        for horizon, value in enumerate(values):
            rows.append(
                {
                    "run": "example",
                    "split": "test",
                    "forecast_day": forecast_day,
                    "scenario_id": scenario_id,
                    "scenario_kind": "raw",
                    "source_scenario_id": scenario_id,
                    "ds": forecast_day + pd.Timedelta(hours=horizon),
                    "horizon": horizon,
                    "y_scenario": value,
                    "probability": 0.25,
                    "sampling_method": "lhs",
                    "copula_family": "student_t",
                    "tail_policy": "linear",
                    "random_seed": 7,
                }
            )
    return pd.DataFrame(rows)


def test_reduce_scenarios_keeps_real_paths_and_reweights_probability() -> None:
    reduced = reduce_scenarios(_raw_scenarios(), n_reduced_scenarios=2, random_seed=7)

    assert set(reduced["scenario_kind"]) == {"reduced"}
    assert reduced["scenario_id"].nunique() == 2
    assert reduced.groupby("scenario_id").size().eq(3).all()
    probabilities = reduced.groupby("scenario_id")["probability"].first()
    assert abs(float(probabilities.sum()) - 1.0) < 1e-12
    assert set(round(float(value), 6) for value in probabilities) == {0.5}
    assert set(reduced["source_scenario_id"]).issubset({0, 1, 2, 3})


def test_reduce_scenarios_rejects_too_many_requested_paths() -> None:
    with pytest.raises(ValueError, match="n_reduced_scenarios"):
        reduce_scenarios(_raw_scenarios(), n_reduced_scenarios=5, random_seed=7)
```

Add `import pytest` to this file.

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_scenario_reduction.py -q
```

Expected: fail because `pjm_forecast.scenarios.reduction` does not exist.

- [ ] **Step 3: Implement reduction**

Create `src/pjm_forecast/scenarios/reduction.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def reduce_scenarios(
    raw_scenarios: pd.DataFrame,
    *,
    n_reduced_scenarios: int = 32,
    distance: str = "l2",
    hour_weights: list[float] | None = None,
    random_seed: int = 7,
) -> pd.DataFrame:
    if n_reduced_scenarios <= 0:
        raise ValueError("n_reduced_scenarios must be positive.")
    output_frames = []
    for forecast_day, day_frame in raw_scenarios.groupby("forecast_day", sort=True):
        reduced_day = _reduce_one_day(
            day_frame,
            forecast_day=pd.Timestamp(forecast_day),
            n_reduced_scenarios=n_reduced_scenarios,
            distance=distance,
            hour_weights=hour_weights,
            random_seed=random_seed,
        )
        output_frames.append(reduced_day)
    if not output_frames:
        return pd.DataFrame(columns=list(raw_scenarios.columns))
    return pd.concat(output_frames, ignore_index=True)


def _reduce_one_day(
    day_frame: pd.DataFrame,
    *,
    forecast_day: pd.Timestamp,
    n_reduced_scenarios: int,
    distance: str,
    hour_weights: list[float] | None,
    random_seed: int,
) -> pd.DataFrame:
    matrix, scenario_ids, ds_index = _scenario_matrix(day_frame)
    if n_reduced_scenarios > len(scenario_ids):
        raise ValueError("n_reduced_scenarios cannot exceed the number of raw scenarios for a day.")
    weights = _hour_weights(matrix.shape[1], hour_weights)
    distances = _pairwise_distances(matrix, weights=weights, distance=distance)
    medoid_positions = _greedy_medoids(distances, n_reduced_scenarios, random_seed=random_seed)
    assigned_positions = np.argmin(distances[:, medoid_positions], axis=1)
    rows: list[dict[str, object]] = []
    template = day_frame.iloc[0].to_dict()

    for reduced_id, medoid_position in enumerate(medoid_positions):
        raw_members = np.where(assigned_positions == reduced_id)[0]
        probability = float(len(raw_members) / len(scenario_ids))
        source_scenario_id = int(scenario_ids[medoid_position])
        medoid_values = matrix[medoid_position]
        for horizon, ds in enumerate(ds_index):
            row = dict(template)
            row.update(
                {
                    "forecast_day": forecast_day,
                    "scenario_id": int(reduced_id),
                    "scenario_kind": "reduced",
                    "source_scenario_id": source_scenario_id,
                    "ds": pd.Timestamp(ds),
                    "horizon": int(horizon),
                    "y_scenario": float(medoid_values[horizon]),
                    "probability": probability,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _scenario_matrix(day_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    ordered = day_frame.sort_values(["scenario_id", "horizon"]).copy()
    scenario_ids = ordered["scenario_id"].drop_duplicates().to_numpy(dtype=int)
    ds_index = pd.DatetimeIndex(ordered.loc[ordered["scenario_id"].eq(scenario_ids[0]), "ds"])
    pivot = ordered.pivot(index="scenario_id", columns="horizon", values="y_scenario").sort_index()
    if pivot.isna().any().any():
        raise ValueError("raw scenarios must contain a complete path for every scenario_id.")
    return pivot.to_numpy(dtype=float), pivot.index.to_numpy(dtype=int), ds_index


def _hour_weights(horizon: int, hour_weights: list[float] | None) -> np.ndarray:
    if hour_weights is None:
        return np.ones(horizon, dtype=float)
    weights = np.asarray(hour_weights, dtype=float)
    if weights.shape != (horizon,):
        raise ValueError("hour_weights must match the path horizon.")
    if np.any(weights <= 0.0):
        raise ValueError("hour_weights must be positive.")
    return weights


def _pairwise_distances(matrix: np.ndarray, *, weights: np.ndarray, distance: str) -> np.ndarray:
    weighted = np.asarray(matrix, dtype=float) * np.sqrt(weights.reshape(1, -1))
    delta = weighted[:, None, :] - weighted[None, :, :]
    if distance == "l2":
        return np.sqrt(np.sum(delta * delta, axis=2))
    if distance == "l1":
        return np.sum(np.abs(delta), axis=2)
    raise ValueError(f"Unsupported scenario reduction distance={distance!r}.")


def _greedy_medoids(distances: np.ndarray, n_medoids: int, *, random_seed: int) -> np.ndarray:
    del random_seed
    selected = [int(np.argmin(np.sum(distances, axis=1)))]
    while len(selected) < n_medoids:
        best_candidate = None
        best_cost = float("inf")
        for candidate in range(distances.shape[0]):
            if candidate in selected:
                continue
            trial = np.asarray(selected + [candidate], dtype=int)
            cost = float(np.sum(np.min(distances[:, trial], axis=1)))
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate
        selected.append(int(best_candidate))
    return np.asarray(selected, dtype=int)
```

- [ ] **Step 4: Export function**

Update `src/pjm_forecast/scenarios/__init__.py`:

```python
from __future__ import annotations

from .generation import generate_raw_scenarios
from .reduction import reduce_scenarios

__all__ = ["generate_raw_scenarios", "reduce_scenarios"]
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_scenario_reduction.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\scenarios\__init__.py src\pjm_forecast\scenarios\reduction.py tests\test_scenario_reduction.py
git commit -m "Add scenario reduction"
```

---

### Task 4: Add Raw-vs-Reduced Diagnostics

**Files:**
- Create: `src/pjm_forecast/scenarios/diagnostics.py`
- Modify: `src/pjm_forecast/scenarios/__init__.py`
- Test: `tests/test_scenario_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenario_diagnostics.py`:

```python
from __future__ import annotations

import pandas as pd

from pjm_forecast.scenarios.diagnostics import compute_reduction_diagnostics


def _paths(kind: str) -> pd.DataFrame:
    rows = []
    forecast_day = pd.Timestamp("2026-01-01")
    paths = {
        0: ([10.0, 11.0, 12.0], 0.5),
        1: ([20.0, 21.0, 22.0], 0.5),
    }
    for scenario_id, (values, probability) in paths.items():
        for horizon, value in enumerate(values):
            rows.append(
                {
                    "run": "example",
                    "split": "test",
                    "forecast_day": forecast_day,
                    "scenario_id": scenario_id,
                    "scenario_kind": kind,
                    "source_scenario_id": scenario_id,
                    "ds": forecast_day + pd.Timedelta(hours=horizon),
                    "horizon": horizon,
                    "y_scenario": value,
                    "probability": probability,
                }
            )
    return pd.DataFrame(rows)


def test_compute_reduction_diagnostics_reports_zero_error_for_identical_sets() -> None:
    diagnostics = compute_reduction_diagnostics(_paths("raw"), _paths("reduced"))

    row = diagnostics.iloc[0]
    assert row["split"] == "test"
    assert row["forecast_days"] == 1
    assert row["raw_scenarios_per_day"] == 2
    assert row["reduced_scenarios_per_day"] == 2
    assert row["daily_mean_abs_error"] == 0.0
    assert row["daily_max_abs_error"] == 0.0
    assert row["daily_spread_abs_error"] == 0.0
    assert row["daily_ramp_abs_error"] == 0.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_scenario_diagnostics.py -q
```

Expected: fail because `pjm_forecast.scenarios.diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostics**

Create `src/pjm_forecast/scenarios/diagnostics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reduction_diagnostics(raw_scenarios: pd.DataFrame, reduced_scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for forecast_day, raw_day in raw_scenarios.groupby("forecast_day", sort=True):
        reduced_day = reduced_scenarios.loc[reduced_scenarios["forecast_day"].eq(forecast_day)]
        raw_stats = _weighted_daily_stats(raw_day)
        reduced_stats = _weighted_daily_stats(reduced_day)
        rows.append(
            {
                "split": str(raw_day["split"].iloc[0]),
                "forecast_day": pd.Timestamp(forecast_day),
                "forecast_days": 1,
                "raw_scenarios_per_day": int(raw_day["scenario_id"].nunique()),
                "reduced_scenarios_per_day": int(reduced_day["scenario_id"].nunique()),
                "daily_mean_abs_error": abs(raw_stats["mean"] - reduced_stats["mean"]),
                "daily_max_abs_error": abs(raw_stats["max"] - reduced_stats["max"]),
                "daily_spread_abs_error": abs(raw_stats["spread"] - reduced_stats["spread"]),
                "daily_ramp_abs_error": abs(raw_stats["ramp"] - reduced_stats["ramp"]),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    summary = {
        "split": str(daily["split"].iloc[0]),
        "forecast_day": "ALL",
        "forecast_days": int(len(daily)),
        "raw_scenarios_per_day": int(daily["raw_scenarios_per_day"].median()),
        "reduced_scenarios_per_day": int(daily["reduced_scenarios_per_day"].median()),
        "daily_mean_abs_error": float(daily["daily_mean_abs_error"].mean()),
        "daily_max_abs_error": float(daily["daily_max_abs_error"].mean()),
        "daily_spread_abs_error": float(daily["daily_spread_abs_error"].mean()),
        "daily_ramp_abs_error": float(daily["daily_ramp_abs_error"].mean()),
    }
    return pd.concat([pd.DataFrame([summary]), daily], ignore_index=True)


def _weighted_daily_stats(frame: pd.DataFrame) -> dict[str, float]:
    paths = frame.pivot(index="scenario_id", columns="horizon", values="y_scenario").sort_index()
    probabilities = frame.groupby("scenario_id")["probability"].first().reindex(paths.index).to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()
    values = paths.to_numpy(dtype=float)
    daily_mean = np.mean(values, axis=1)
    daily_max = np.max(values, axis=1)
    daily_spread = np.max(values, axis=1) - np.min(values, axis=1)
    daily_ramp = np.max(np.abs(np.diff(values, axis=1)), axis=1) if values.shape[1] > 1 else np.zeros(values.shape[0])
    return {
        "mean": float(np.sum(probabilities * daily_mean)),
        "max": float(np.sum(probabilities * daily_max)),
        "spread": float(np.sum(probabilities * daily_spread)),
        "ramp": float(np.sum(probabilities * daily_ramp)),
    }
```

- [ ] **Step 4: Export function**

Update `src/pjm_forecast/scenarios/__init__.py`:

```python
from __future__ import annotations

from .diagnostics import compute_reduction_diagnostics
from .generation import generate_raw_scenarios
from .reduction import reduce_scenarios

__all__ = ["compute_reduction_diagnostics", "generate_raw_scenarios", "reduce_scenarios"]
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_scenario_diagnostics.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src\pjm_forecast\scenarios\__init__.py src\pjm_forecast\scenarios\diagnostics.py tests\test_scenario_diagnostics.py
git commit -m "Add scenario reduction diagnostics"
```

---

### Task 5: Add Config Contract

**Files:**
- Modify: `src/pjm_forecast/config.py`
- Modify: `tests/test_config_contracts.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_config_contracts.py`:

```python
def test_current_config_enables_scenario_generation() -> None:
    config = load_config("configs/pjm_day_ahead_current_processed.yaml")

    generation = config.report["scenario_generation"]
    assert generation["enabled"] is True
    assert generation["source_split"] == "validation"
    assert generation["sampling_method"] == "lhs"
    assert generation["n_raw_scenarios"] == 1024
    assert generation["n_reduced_scenarios"] == 32


def test_load_config_rejects_invalid_scenario_generation_counts(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        [
            ("report", "scenario_generation"),
        ],
        {
            "enabled": True,
            "source_split": "validation",
            "copula_family": "student_t",
            "tail_policy": "linear",
            "sampling_method": "lhs",
            "n_raw_scenarios": 16,
            "n_reduced_scenarios": 32,
            "random_seed": 7,
        },
    )

    with pytest.raises(ValueError, match="n_reduced_scenarios"):
        load_config(config_path)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_config_contracts.py::test_current_config_enables_scenario_generation tests\test_config_contracts.py::test_load_config_rejects_invalid_scenario_generation_counts -q
```

Expected: fail because `scenario_generation` is not configured and not validated.

- [ ] **Step 3: Add constants**

In `src/pjm_forecast/config.py`, add:

```python
SCENARIO_SAMPLING_METHODS = {"mc", "lhs"}
SCENARIO_REDUCTION_METHODS = {"wasserstein_k_medoids"}
SCENARIO_REDUCTION_DISTANCES = {"l1", "l2"}
```

- [ ] **Step 4: Call the validator**

In `ProjectConfig.validate_runtime_contracts`, after `self.validate_scenario_evaluation_contracts()`, add:

```python
self.validate_scenario_generation_contracts()
```

- [ ] **Step 5: Add scenario generation validation**

Add this method to `ProjectConfig`:

```python
def validate_scenario_generation_contracts(self) -> None:
    generation_cfg = self.report.get("scenario_generation", {})
    if not generation_cfg:
        return
    enabled = generation_cfg.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("report.scenario_generation.enabled must be a boolean when configured.")
    source_split = generation_cfg.get("source_split", "validation")
    if source_split != "validation":
        raise ValueError("report.scenario_generation.source_split currently only supports 'validation'.")
    family = str(generation_cfg.get("copula_family", "student_t"))
    if family not in SCENARIO_COPULA_FAMILIES:
        raise ValueError(f"Unsupported report.scenario_generation.copula_family={family!r}.")
    tail_policy = str(generation_cfg.get("tail_policy", "linear"))
    if tail_policy not in SCENARIO_TAIL_POLICIES:
        raise ValueError(f"Unsupported report.scenario_generation.tail_policy={tail_policy!r}.")
    sampling_method = str(generation_cfg.get("sampling_method", "lhs"))
    if sampling_method not in SCENARIO_SAMPLING_METHODS:
        raise ValueError(f"Unsupported report.scenario_generation.sampling_method={sampling_method!r}.")
    n_raw = int(generation_cfg.get("n_raw_scenarios", 1024))
    n_reduced = int(generation_cfg.get("n_reduced_scenarios", 32))
    if n_raw <= 0:
        raise ValueError("report.scenario_generation.n_raw_scenarios must be a positive integer.")
    if n_reduced <= 0 or n_reduced > n_raw:
        raise ValueError("report.scenario_generation.n_reduced_scenarios must be positive and no larger than n_raw_scenarios.")
    random_seed = generation_cfg.get("random_seed", 7)
    if not isinstance(random_seed, int):
        raise ValueError("report.scenario_generation.random_seed must be an integer.")
    reduction = generation_cfg.get("reduction", {})
    if not isinstance(reduction, dict):
        raise ValueError("report.scenario_generation.reduction must be a mapping.")
    method = str(reduction.get("method", "wasserstein_k_medoids"))
    if method not in SCENARIO_REDUCTION_METHODS:
        raise ValueError(f"Unsupported report.scenario_generation.reduction.method={method!r}.")
    distance = str(reduction.get("distance", "l2"))
    if distance not in SCENARIO_REDUCTION_DISTANCES:
        raise ValueError(f"Unsupported report.scenario_generation.reduction.distance={distance!r}.")
    dof_grid = generation_cfg.get("dof_grid")
    if dof_grid is not None:
        if not isinstance(dof_grid, list) or not dof_grid:
            raise ValueError("report.scenario_generation.dof_grid must be a non-empty list when configured.")
        for value in dof_grid:
            if float(value) <= 2.0:
                raise ValueError("report.scenario_generation.dof_grid values must be > 2.")
```

- [ ] **Step 6: Update canonical config**

Add this under `report:` in `configs/pjm_day_ahead_current_processed.yaml`:

```yaml
  scenario_generation:
    enabled: true
    source_split: "validation"
    copula_family: "student_t"
    tail_policy: "linear"
    dof_grid: [3.0, 5.0, 7.0, 10.0]
    sampling_method: "lhs"
    n_raw_scenarios: 1024
    n_reduced_scenarios: 32
    random_seed: 7
    reduction:
      method: "wasserstein_k_medoids"
      distance: "l2"
      hour_weighting: "uniform"
```

- [ ] **Step 7: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_config_contracts.py::test_current_config_enables_scenario_generation tests\test_config_contracts.py::test_load_config_rejects_invalid_scenario_generation_counts -q
```

Expected: `2 passed`.

- [ ] **Step 8: Commit**

```powershell
git add src\pjm_forecast\config.py configs\pjm_day_ahead_current_processed.yaml tests\test_config_contracts.py
git commit -m "Add scenario generation config contract"
```

---

### Task 6: Add Workspace Artifacts And Methods

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing artifact path assertions**

Add to `test_workspace_open_respects_root_override_and_artifact_contract`:

```python
assert workspace.artifacts.raw_scenarios("test") == (
    tmp_path / "run" / "artifacts" / "scenarios" / "test_raw_scenarios.parquet"
).resolve()
assert workspace.artifacts.reduced_scenarios("test") == (
    tmp_path / "run" / "artifacts" / "scenarios" / "test_reduced_scenarios.parquet"
).resolve()
assert workspace.artifacts.scenario_reduction_diagnostics("test") == (
    tmp_path / "run" / "artifacts" / "metrics" / "test_scenario_reduction_diagnostics.csv"
).resolve()
```

- [ ] **Step 2: Run path test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_open_respects_root_override_and_artifact_contract -q
```

Expected: fail because the artifact methods do not exist.

- [ ] **Step 3: Add artifact methods**

Add methods to `ArtifactStore` in `src/pjm_forecast/workspace.py`:

```python
def raw_scenarios(self, split: str) -> Path:
    return self.directories["artifact_dir"] / "scenarios" / f"{split}_raw_scenarios.parquet"


def reduced_scenarios(self, split: str) -> Path:
    return self.directories["artifact_dir"] / "scenarios" / f"{split}_reduced_scenarios.parquet"


def scenario_reduction_diagnostics(self, split: str) -> Path:
    return self.directories["metrics_dir"] / f"{split}_scenario_reduction_diagnostics.csv"
```

- [ ] **Step 4: Write failing workspace flow test**

Add this test to `tests/test_workspace.py`:

```python
def test_workspace_generates_reduces_and_diagnoses_scenarios(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    config_path = _write_temp_config(tmp_path, csv_path)
    workspace = Workspace.open(config_path)
    workspace.config.raw["backtest"]["benchmark_models"] = ["nhits_tail_grid_weighted_main"]
    workspace.config.raw["models"]["nhits_tail_grid_weighted_main"] = {"type": "nhits"}
    workspace.config.raw["report"]["scenario_generation"] = {
        "enabled": True,
        "source_split": "validation",
        "copula_family": "gaussian",
        "tail_policy": "linear",
        "sampling_method": "lhs",
        "n_raw_scenarios": 8,
        "n_reduced_scenarios": 2,
        "random_seed": 7,
        "reduction": {"method": "wasserstein_k_medoids", "distance": "l2", "hour_weighting": "uniform"},
    }
    for split in ["validation", "test"]:
        path = workspace.artifacts.prediction("nhits_tail_grid_weighted_main", split, 7)
        path.parent.mkdir(parents=True, exist_ok=True)
        _quantile_prediction_frame(split).to_parquet(path, index=False)

    raw_path = workspace.generate_scenarios("test")
    reduced_path = workspace.reduce_scenarios("test")
    diagnostics_path = workspace.diagnose_scenarios("test")

    assert raw_path == workspace.artifacts.raw_scenarios("test")
    assert reduced_path == workspace.artifacts.reduced_scenarios("test")
    assert diagnostics_path == workspace.artifacts.scenario_reduction_diagnostics("test")
    assert raw_path.exists()
    assert reduced_path.exists()
    assert diagnostics_path.exists()
```

- [ ] **Step 5: Run workspace test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_generates_reduces_and_diagnoses_scenarios -q
```

Expected: fail because `Workspace.generate_scenarios` does not exist.

- [ ] **Step 6: Add imports**

Add to `src/pjm_forecast/workspace.py`:

```python
from .scenarios import compute_reduction_diagnostics, generate_raw_scenarios, reduce_scenarios
```

- [ ] **Step 7: Add helper to read scenario config**

Add to `Workspace`:

```python
def _scenario_generation_config(self) -> Mapping[str, object]:
    cfg = self.config.report.get("scenario_generation", {})
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        raise ValueError("report.scenario_generation must be enabled.")
    return cfg
```

- [ ] **Step 8: Add `generate_scenarios`**

Add to `Workspace`:

```python
def generate_scenarios(self, split: SplitName = "test") -> Path:
    cfg = self._scenario_generation_config()
    source_split = str(cfg.get("source_split", "validation"))
    if source_split != "validation":
        raise ValueError("scenario generation must use validation source_split.")
    model_name = str(cfg.get("model_name", self.config.backtest["benchmark_models"][0]))
    seed = int(cfg.get("seed", self.config.project["benchmark_seed"]))
    source_predictions = pd.read_parquet(self.artifacts.prediction(model_name, source_split, seed))
    target_predictions = pd.read_parquet(self.artifacts.prediction(model_name, split, seed))
    raw = generate_raw_scenarios(
        source_predictions=source_predictions,
        target_predictions=target_predictions,
        run_name=f"{model_name}_{split}_seed{seed}",
        split=split,
        family=str(cfg.get("copula_family", "student_t")),
        dof_grid=cfg.get("dof_grid"),
        tail_policy=str(cfg.get("tail_policy", "linear")),
        sampling_method=str(cfg.get("sampling_method", "lhs")),
        n_scenarios=int(cfg.get("n_raw_scenarios", 1024)),
        random_seed=int(cfg.get("random_seed", 7)),
    )
    output_path = self.artifacts.raw_scenarios(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output_path, index=False)
    return output_path
```

- [ ] **Step 9: Add `reduce_scenarios`**

Add to `Workspace`:

```python
def reduce_scenarios(self, split: SplitName = "test") -> Path:
    cfg = self._scenario_generation_config()
    reduction_cfg = cfg.get("reduction", {})
    if not isinstance(reduction_cfg, Mapping):
        raise ValueError("report.scenario_generation.reduction must be a mapping.")
    raw = pd.read_parquet(self.artifacts.raw_scenarios(split))
    reduced = reduce_scenarios(
        raw,
        n_reduced_scenarios=int(cfg.get("n_reduced_scenarios", 32)),
        distance=str(reduction_cfg.get("distance", "l2")),
        hour_weights=None,
        random_seed=int(cfg.get("random_seed", 7)),
    )
    output_path = self.artifacts.reduced_scenarios(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(output_path, index=False)
    return output_path
```

- [ ] **Step 10: Add `diagnose_scenarios`**

Add to `Workspace`:

```python
def diagnose_scenarios(self, split: SplitName = "test") -> Path:
    raw = pd.read_parquet(self.artifacts.raw_scenarios(split))
    reduced = pd.read_parquet(self.artifacts.reduced_scenarios(split))
    diagnostics = compute_reduction_diagnostics(raw, reduced)
    output_path = self.artifacts.scenario_reduction_diagnostics(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    return output_path
```

- [ ] **Step 11: Export diagnostics in report bundle**

Add `self.scenario_reduction_diagnostics(split)` to the source list in `ArtifactStore.export_report_bundle`.

- [ ] **Step 12: Run tests**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_open_respects_root_override_and_artifact_contract tests\test_workspace.py::test_workspace_generates_reduces_and_diagnoses_scenarios -q
```

Expected: both tests pass.

- [ ] **Step 13: Commit**

```powershell
git add src\pjm_forecast\workspace.py tests\test_workspace.py
git commit -m "Add scenario generation workspace flow"
```

---

### Task 7: Add CLI Shims

**Files:**
- Create: `scripts/generate_scenarios.py`
- Create: `scripts/reduce_scenarios.py`
- Create: `scripts/diagnose_scenarios.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Create `scripts/generate_scenarios.py`**

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw scenario paths from quantile predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    output_path = Workspace.open(args.config).generate_scenarios(split=args.split)
    print(f"Wrote raw scenarios to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/reduce_scenarios.py`**

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce raw scenario paths to representative paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    output_path = Workspace.open(args.config).reduce_scenarios(split=args.split)
    print(f"Wrote reduced scenarios to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/diagnose_scenarios.py`**

```python
from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw and reduced scenario paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    output_path = Workspace.open(args.config).diagnose_scenarios(split=args.split)
    print(f"Wrote scenario reduction diagnostics to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run import checks**

Run:

```powershell
uv run python scripts\generate_scenarios.py --help
uv run python scripts\reduce_scenarios.py --help
uv run python scripts\diagnose_scenarios.py --help
```

Expected: each command prints help and exits with code 0.

- [ ] **Step 5: Commit**

```powershell
git add scripts\generate_scenarios.py scripts\reduce_scenarios.py scripts\diagnose_scenarios.py
git commit -m "Add scenario generation CLI commands"
```

---

### Task 8: Wire Scenario Stages Into Pipeline

**Files:**
- Modify: `src/pjm_forecast/pipeline.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Update stage-order test first**

Update `test_pipeline_stage_order_includes_quality_closure`:

```python
def test_pipeline_stage_order_includes_quality_closure() -> None:
    assert STAGE_ORDER == [
        "prepare_data",
        "tune_model",
        "backtest_all_models",
        "evaluate_and_plot",
        "generate_scenarios",
        "reduce_scenarios",
        "diagnose_scenarios",
        "audit_event_risk_overlay",
        "finalize_quality_flow",
        "export_report_assets",
        "export_model_snapshot",
    ]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_pipeline_stage_order_includes_quality_closure -q
```

Expected: fail because the stage order does not include scenario stages.

- [ ] **Step 3: Update `STAGE_ORDER`**

In `src/pjm_forecast/pipeline.py`:

```python
STAGE_ORDER = [
    "prepare_data",
    "tune_model",
    "backtest_all_models",
    "evaluate_and_plot",
    "generate_scenarios",
    "reduce_scenarios",
    "diagnose_scenarios",
    "audit_event_risk_overlay",
    "finalize_quality_flow",
    "export_report_assets",
    "export_model_snapshot",
]
```

- [ ] **Step 4: Add scenario stage helpers**

Add these functions:

```python
def _scenario_generation_enabled(workspace: Workspace) -> bool:
    generation = workspace.config.raw.get("report", {}).get("scenario_generation", {})
    return isinstance(generation, Mapping) and bool(generation.get("enabled", False))


def _run_generate_scenarios(workspace: Workspace, split: str) -> None:
    if _scenario_generation_enabled(workspace):
        workspace.generate_scenarios(split=split)


def _run_reduce_scenarios(workspace: Workspace, split: str) -> None:
    if _scenario_generation_enabled(workspace):
        workspace.reduce_scenarios(split=split)


def _run_diagnose_scenarios(workspace: Workspace, split: str) -> None:
    if _scenario_generation_enabled(workspace):
        workspace.diagnose_scenarios(split=split)
```

- [ ] **Step 5: Register stages**

Add to `STAGE_FUNCTIONS`:

```python
"generate_scenarios": _run_generate_scenarios,
"reduce_scenarios": _run_reduce_scenarios,
"diagnose_scenarios": _run_diagnose_scenarios,
```

- [ ] **Step 6: Run stage-order test**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_pipeline_stage_order_includes_quality_closure -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit**

```powershell
git add src\pjm_forecast\pipeline.py tests\test_workspace.py
git commit -m "Wire scenario generation into pipeline"
```

---

### Task 9: Include Scenario Artifacts In Final Manifest

**Files:**
- Modify: `src/pjm_forecast/workspace.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Extend finalization test**

In `test_workspace_finalize_quality_flow_writes_summary_and_manifest`, create tiny scenario artifacts before calling `finalize_quality_flow`:

```python
workspace.artifacts.raw_scenarios("test").parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"value": [1]}).to_parquet(workspace.artifacts.raw_scenarios("test"), index=False)
pd.DataFrame({"value": [1]}).to_parquet(workspace.artifacts.reduced_scenarios("test"), index=False)
pd.DataFrame({"metric": ["daily_max_abs_error"], "value": [0.0]}).to_csv(
    workspace.artifacts.scenario_reduction_diagnostics("test"),
    index=False,
)
```

Add manifest assertions:

```python
assert str(workspace.artifacts.raw_scenarios("test")) in artifact_paths
assert str(workspace.artifacts.reduced_scenarios("test")) in artifact_paths
assert str(workspace.artifacts.scenario_reduction_diagnostics("test")) in artifact_paths
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_finalize_quality_flow_writes_summary_and_manifest -q
```

Expected: fail because the manifest does not include scenario artifacts.

- [ ] **Step 3: Add scenario artifacts to manifest**

In `Workspace.finalize_quality_flow`, add these paths to `artifact_paths`:

```python
self.artifacts.raw_scenarios(split),
self.artifacts.reduced_scenarios(split),
self.artifacts.scenario_reduction_diagnostics(split),
```

- [ ] **Step 4: Run test**

Run:

```powershell
uv run python -m pytest tests\test_workspace.py::test_workspace_finalize_quality_flow_writes_summary_and_manifest -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src\pjm_forecast\workspace.py tests\test_workspace.py
git commit -m "Include scenario artifacts in run manifest"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/protocol/canonical_release_checklist.md`

- [ ] **Step 1: Update README workflow output list**

Add these bullets to the canonical pipeline output list in `README.md`:

```markdown
- raw scenario paths under `artifacts_current/scenarios/{split}_raw_scenarios.parquet`
- reduced scenario paths under `artifacts_current/scenarios/{split}_reduced_scenarios.parquet`
- scenario reduction diagnostics under `artifacts_current/metrics/{split}_scenario_reduction_diagnostics.csv`
```

- [ ] **Step 2: Add plain-language explanation**

Add this short section to `README.md` near the scenario evaluation text:

```markdown
### Scenario Generation

The scenario generation step converts quantile predictions into possible 24-hour price paths. It first learns the hour-to-hour movement pattern from validation predictions, then applies that pattern to the requested split. The raw file keeps many generated paths. The reduced file keeps fewer representative paths and assigns each one a probability.

The test split is not used to choose the sampling pattern or reduction settings.
```

- [ ] **Step 3: Update release checklist artifacts**

Add these lines to `docs/protocol/canonical_release_checklist.md`:

```markdown
- `artifacts_current/scenarios/test_raw_scenarios.parquet`
- `artifacts_current/scenarios/test_reduced_scenarios.parquet`
- `artifacts_current/metrics/test_scenario_reduction_diagnostics.csv`
```

- [ ] **Step 4: Commit**

```powershell
git add README.md docs\protocol\canonical_release_checklist.md
git commit -m "Document scenario generation flow"
```

---

### Task 11: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
uv run python -m pytest tests\test_copula.py tests\test_scenario_generation.py tests\test_scenario_reduction.py tests\test_scenario_diagnostics.py tests\test_workspace.py tests\test_config_contracts.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
uv run python -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Run scenario-only pipeline smoke using existing predictions**

Run:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test --start-from generate_scenarios --stop-after diagnose_scenarios
```

Expected files:

```text
artifacts_current/scenarios/test_raw_scenarios.parquet
artifacts_current/scenarios/test_reduced_scenarios.parquet
artifacts_current/metrics/test_scenario_reduction_diagnostics.csv
```

- [ ] **Step 4: Run closure smoke**

Run:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test --start-from generate_scenarios --stop-after export_report_assets
```

Expected:

- Scenario artifacts exist.
- Quality manifest exists.
- Report export includes `test_scenario_reduction_diagnostics.csv`.

- [ ] **Step 5: Inspect artifact sizes**

Run:

```powershell
Get-Item artifacts_current\scenarios\test_raw_scenarios.parquet
Get-Item artifacts_current\scenarios\test_reduced_scenarios.parquet
```

Expected:

- Raw scenario file is larger than reduced scenario file.
- Both files are generated artifacts and are not committed.

- [ ] **Step 6: Check git status**

Run:

```powershell
git status --short
```

Expected: only source, tests, docs, and config changes are tracked for commit; generated scenario artifacts are not staged.

---

## Self-Review

Spec coverage:

- LHS trajectory sampling is covered by Task 1.
- Raw scenario generation is covered by Task 2.
- Scenario reduction is covered by Task 3.
- Raw-vs-reduced checks are covered by Task 4.
- Config validation is covered by Task 5.
- Workspace and artifact integration are covered by Task 6.
- CLI commands are covered by Task 7.
- Pipeline wiring is covered by Task 8.
- Manifest/report closure is covered by Task 9.
- Documentation is covered by Task 10.
- Verification is covered by Task 11.

Placeholder scan:

- No empty placeholders.
- No empty “add tests” instruction without concrete test code.
- No undefined public function is used before a task defines it.

Type consistency:

- Scenario DataFrame uses `forecast_day`, `scenario_id`, `scenario_kind`, `source_scenario_id`, `ds`, `horizon`, `y_scenario`, and `probability` consistently.
- `generate_scenarios`, `reduce_scenarios`, and `diagnose_scenarios` are Workspace methods and pipeline stage names.
- Artifact names are consistent across Workspace, manifest, README, and verification.
