# PJM Day-Ahead Forecasting

Reproducible PJM/COMED day-ahead forecasting pipeline:

- data ingress and normalization
- feature store generation
- rolling backtest
- evaluation and report export

The current canonical configuration is:

- [`configs/pjm_day_ahead_current_processed.yaml`](configs/pjm_day_ahead_current_processed.yaml)

This config uses:

- `official_weather_ready` as the dataset source
- `2021-01-01` to `2026-03-31` hourly data
- Open-Meteo historical forecast weather features
- `NBEATSx` as the active benchmark model

Current probabilistic experiments are tracked separately from the canonical
config. As of the latest experiment notes, `NHITS 600` is the strongest
probabilistic structure candidate, with `NBEATSx 600` retained as the main
baseline for comparison.

## Environment

Use `uv` with Python 3.12:

```powershell
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --extra ml
```

If you only need data/features/tests:

```powershell
uv sync --extra dev
```

`LEAR` and `DNN` require `epftoolbox`, which is not installed by default:

```powershell
uv pip install git+https://github.com/jeslago/epftoolbox.git
```

## Canonical Workflow

Prepare processed data:

```powershell
uv run python scripts\prepare_data.py --config configs\pjm_day_ahead_current_processed.yaml
```

Tune `NBEATSx`:

```powershell
uv run python scripts\tune_nbeatsx.py --config configs\pjm_day_ahead_current_processed.yaml
```

Run backtest:

```powershell
uv run python scripts\backtest_all_models.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

Evaluate and export assets:

```powershell
uv run python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_current_processed.yaml --split test
uv run python scripts\export_report_assets.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

Or run the pipeline wrapper:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

## Data Contracts

- Timestamps stay in timezone-naive local time. Do not remap to UTC in v1.
- Calendar features are derived from the `ds` local hourly sequence. Do not mix
  UTC-remapped timestamps into feature, split, lag, or forecast-window logic.
- Canonical panel columns include `unique_id`, `ds`, `y`, and configured future exogenous signals.
- `NBEATSx` uses:
  - future exogenous signals plus calendar columns as `futr_exog`
  - price lags plus configured lagged signal columns as `hist_exog`

## Experiment Layout

- `configs/pjm_day_ahead_current_processed.yaml` is the canonical runnable
  workflow.
- `configs/experiments/` contains reproducible experiment branches. Keep these
  config-driven and avoid changing canonical behavior unless an experiment has
  been promoted.
- `docs/experiments/` records small human-readable summaries for experiment
  decisions. Generated prediction, metrics, plot, and scenario artifacts remain
  under `artifacts*` directories and should not be treated as source of truth.

## Splits

The current canonical split policy is defined by config and materialized to:

- `data/processed_current/split_boundaries.json`

Current defaults:

- validation: `182` days
- test: `1` year
- rolling window: `728` days

## Legacy Configs

These configs remain in the repo for baseline or compatibility scenarios, but they are not the canonical weather-enabled path:

- [`configs/pjm_day_ahead_v1.yaml`](configs/pjm_day_ahead_v1.yaml)
- [`configs/pjm_day_ahead_kaggle.yaml`](configs/pjm_day_ahead_kaggle.yaml)
