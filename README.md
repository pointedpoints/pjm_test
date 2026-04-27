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
- `NHITS` dense upper-tail quantile training as the active benchmark model

The promoted mainline model is `nhits_tail_grid_weighted_main`. It uses a dense
upper-tail quantile grid through `q0.995`, weighted Huber multi-quantile loss,
and `hour_x_regime` CQR calibration using `spike_score` as postprocess context.

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

Tune the model named by `tuning.model_name`:

```powershell
uv run python scripts\tune_model.py --config configs\pjm_day_ahead_current_processed.yaml
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
- NeuralForecast models (`NBEATSx`/`NHITS`) use:
  - future exogenous signals plus calendar columns as `futr_exog`
  - price lags plus configured lagged signal columns as `hist_exog`
- Canonical `spike_score` is derived into the feature store for calibration
  context. It is not included in the promoted NHITS model's `futr_exog`.
- Postprocessing supports validation-fitted q50 bias correction before CQR.
  It remains disabled in the canonical config until it beats CQR-only metrics
  on validation/test.

## Experiment Layout

- `configs/pjm_day_ahead_current_processed.yaml` is the canonical runnable
  workflow.
- `configs/experiments/` contains reproducible experiment branches. Keep these
  config-driven and avoid changing canonical behavior unless an experiment has
  been promoted.
- `docs/experiments/2026-04-26-execution-plan.md` is the locked execution plan
  for the next prediction-quality phases. Follow its experiment order and
  promotion gates before expanding scope.
- `scripts/inject_prediction_context.py` can copy existing prediction parquet
  files into a new prediction directory while joining context columns from the
  configured feature store by `ds`. Use it for postprocess-only branches that
  reuse a baseline model body but need calibration context such as
  `spike_score`; the canonical backtest writes required `hour_x_regime`
  context directly.
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
