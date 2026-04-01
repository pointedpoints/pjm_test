# AGENTS.md

## Mission and scope
- This repo is a reproducible PJM/COMED day-ahead forecasting pipeline: data prep -> feature store -> rolling backtest -> evaluation/report export (`README.md`).
- Keep changes aligned with the existing time protocol: **no UTC remapping** in v1 (`README.md`, `src/pjm_forecast/data/epftoolbox.py`).

## Architecture map (what depends on what)
- `src/pjm_forecast/workspace.py` is the workflow boundary. `Workspace.open(...)` owns config loading, directory resolution, artifact paths, and stage orchestration.
- `scripts/*.py` are CLI shims over `Workspace`; benchmark scripts stay at the top level, optional branches live under `scripts/experiments/` and `scripts/ops/`.
- `src/pjm_forecast/config.py` + `src/pjm_forecast/paths.py` remain low-level wiring helpers behind `Workspace`.
- Data layer: `src/pjm_forecast/data/epftoolbox.py` downloads CSV and normalizes to panel columns `unique_id, ds, y, system_load_forecast, zonal_load_forecast`.
- Feature layer: `src/pjm_forecast/features/engineering.py` adds calendar/cyclical features and lag features; preserves `ds` hourly ordering.
- Backtest layer: `src/pjm_forecast/backtest/engine.py` applies rolling windows + weekly retrain policy and enforces exact horizon row count.
- Model layer: `src/pjm_forecast/models/registry.py` maps config model types to adapters (`seasonal_naive`, `lear`, `dnn`, `nbeatsx`).
- Evaluation layer: `src/pjm_forecast/evaluation/*` computes scalar metrics, DM tests, and plots; `Evaluator` owns run discovery/alignment and report export remains a derived artifact copy step.
- Retrieval layer: `src/pjm_forecast/retrieval/*` keeps retrieval math in `residual_memory.py`; `RetrievalRunner` owns warmup/tuning/apply orchestration.
- Spike correction layer: `src/pjm_forecast/spike_correction/*` is an explicit experiment branch for GBM-based spike gating/residual correction; `SpikeCorrectorRunner` owns warmup/tuning/apply orchestration.

## Critical data and prediction contracts
- Feature frames are expected to contain `ds`, `y`, future exogenous columns, and lag columns (see `tests/test_data_pipeline.py`).
- `run_rolling_backtest(...)` expects model objects with `fit(train_df)` + `predict(history_df, future_df)`; prediction output must include `ds` and `y_pred` (`src/pjm_forecast/models/base.py`).
- Backtest output contract includes `ds, y, y_pred, model, split, seed, quantile, metadata` (`tests/test_backtest_protocol.py`).
- Split boundaries are serialized JSON timestamps (`train_end`, `validation_*`, `test_*`) and consumed by `get_daily_split_days(...)`.

## Developer workflows (canonical commands)
- Environment (Python 3.12):
  - `python -m pip install -e .[dev]` for data/features/tests only.
  - `python -m pip install -e .[dev,ml]` for NBEATSx/epftoolbox models (`pyproject.toml`).
- End-to-end pipeline (run in order):
  - `python scripts\prepare_data.py --config configs\pjm_day_ahead_v1.yaml`
  - `python scripts\tune_nbeatsx.py --config configs\pjm_day_ahead_v1.yaml`
  - `python scripts\backtest_all_models.py --config configs\pjm_day_ahead_v1.yaml --split test`
  - `python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_v1.yaml --split test`
  - `python scripts\export_report_assets.py --config configs\pjm_day_ahead_v1.yaml --split test`
- Optional experiment:
  - `python scripts\experiments\retrieve_nbeatsx.py --config configs\pjm_day_ahead_v1.yaml --split test`
  - `python scripts\experiments\run_spike_corrector.py --config configs\pjm_day_ahead_v1.yaml --split test`
  - `python scripts\experiments\run_lgbm_stacker.py --config configs\pjm_day_ahead_v1.yaml --split test`
- Test baseline:
  - `pytest`

## Project-specific conventions to preserve
- Paths in config are project-relative and resolved from config location (`ProjectConfig.resolve_path`).
- `tune_nbeatsx.py` mutates `config.models["nbeatsx"]` inside the Optuna objective; avoid assuming immutable config objects.
- `backtest_all_models.py` runs multi-seed only for `nbeatsx`; other models use `benchmark_seed`.
- EPF wrappers rename columns to `Price`, `Exogenous 1`, `Exogenous 2`; this mapping is required for `epftoolbox` adapters.
- Optional ML dependencies are lazily imported in model `__post_init__`; missing packages should raise clear `ImportError`, not fail silently.

## Integration points and artifacts
- External data source is configured in YAML (`dataset.source_url`) and downloaded only if missing.
- Key artifacts are under `artifacts/`: `hyperparameters/`, `predictions/`, `metrics/`, `plots/`, `report/`.
- `ArtifactStore.prediction_runs(...)` is the source for evaluation discovery; do not rebuild run identity from filename regex in callers.
- Spike correction writes structured parameter and diagnostics artifacts; corrected predictions must still preserve the canonical prediction contract and may add debug columns.
- LightGBM stacking is another explicit experiment branch; it consumes aligned base-model prediction runs plus feature-store columns and writes its own params/diagnostics artifacts without entering the default pipeline.
- `artifacts/report/` is a derived export view, not the source of truth for metrics or plots.
- `evaluate_and_plot.py` pairs runs for DM tests only when `ds` timestamps align exactly.

