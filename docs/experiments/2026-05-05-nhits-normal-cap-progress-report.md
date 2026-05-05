# NHITS Normal-Cap Progress Report

Date: 2026-05-05
Branch: `codex/comed-scoreboard-and-baselines`
Pushed head: `4a796c9`

## Scope

This work implements the non-training infrastructure for the first NHITS
normal-day causal target-capping experiment.

The current question remains narrow:

- main model family: NHITS only
- training change: causal hourly target cap through existing `target_filter`
- evaluation focus: normal-day and forecast low-risk-day relative error
- primary metrics: WAPE, median APE, p75 APE, p90 APE, sMAPE
- secondary diagnostics: MAE and q50 bias

No baseline model work was added. The canonical config remains unchanged.
No UTC remapping was introduced.

## Completed

### Normal-Day Diagnostics

Added `src/pjm_forecast/evaluation/normal_day.py`.

It computes q50/point diagnostics for these segments:

- `all`
- `actual_normal_day`
- `actual_spike_day`
- `forecast_low_risk_day`
- `forecast_high_risk_day`

Actual normal/spike labels are evaluation-only and use realized daily max `y`.
Forecast low/high-risk labels use forecast-available `spike_score`, threshold,
and mean/max aggregation.

### Evaluation Wiring

Updated:

- `src/pjm_forecast/evaluation/evaluator.py`
- `src/pjm_forecast/workspace.py`
- `src/pjm_forecast/evaluation/scorecard.py`

The evaluation flow now writes:

- `{split}_normal_day_diagnostics.csv`

The experiment scorecard now includes:

- `actual_normal_day_q50_wape`
- `actual_normal_day_median_ape`
- `actual_normal_day_p75_ape`
- `actual_normal_day_p90_ape`
- `actual_normal_day_smape`
- `forecast_low_risk_day_q50_wape`
- `forecast_low_risk_day_median_ape`
- `forecast_low_risk_day_p75_ape`
- `forecast_low_risk_day_p90_ape`
- `forecast_low_risk_day_smape`

The scorecard joins diagnostics by `run`, so multi-run metrics do not leak
across models or seeds.

### Experiment Config

Added:

- `configs/experiments/pjm_current_validation_nhits_normal_cap.yaml`

Key config choices:

- `project.name`: `pjm_current_validation_nhits_normal_cap`
- `backtest.benchmark_models`: `["nhits_normal_cap"]`
- `tuning.model_name`: `nhits_normal_cap`
- artifact root: `../artifacts_phase2/nhits_normal_cap`
- model type: `nhits`
- `target_filter.enabled`: `true`
- `target_filter.window_observations`: `365`
- `target_filter.min_history`: `60`
- `target_filter.quantile`: `0.95`
- `target_filter.fallback_quantile`: `0.975`
- `target_filter.iqr_multiplier`: `3.0`

Project-relative experiment paths were checked so the config resolves data under
`D:\pjm_remaster\data\...`, not `D:\pjm_remaster\configs\data\...`.

## Review Fixes Applied

Several subagent review issues were found and fixed before stopping:

- `q50_bias_mean` now uses arithmetic mean bias, not median residual.
- normal-day tests explicitly cover `median_ape`, `p75_ape`, `p90_ape`, and
  `smape`, not only WAPE.
- `Workspace.evaluate` now passes computed normal-day diagnostics into
  experiment scorecard construction.
- the normal-cap config now uses correct experiment-relative data paths.
- the registry test monkeypatches the NHITS adapter so dev-only test
  environments do not need NeuralForecast/Torch just to test wrapper behavior.

Final code review found no blocking issues. Remaining minor notes:

- normal-day diagnostics are sorted lexicographically by `segment`, which is
  cosmetic.
- `low_risk_threshold` is not range-validated; the current config uses `0.50`.
- normal-day sMAPE is fraction-scale, matching relative-error diagnostics;
  top-level `smape` remains percent-scale.

## Verification

No training or backtesting commands were run.

Fresh non-training verification:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest tests\test_normal_day.py tests\test_scorecard.py tests\test_workspace.py::test_workspace_evaluate_passes_normal_day_diagnostics_to_scorecard tests\test_model_registry_target_filter.py -q --basetemp=.tmp_pytest_nhits_normal_focused
```

Result:

```text
14 passed, 1 warning
```

Full test suite:

```powershell
$env:TMP='D:\pjm_remaster\.tmp'
$env:TEMP='D:\pjm_remaster\.tmp'
& D:\pjm_remaster\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp_pytest_nhits_normal_full
```

Result:

```text
220 passed, 1 warning
```

The warning is the existing `pkg_resources` deprecation warning from `hyperopt`.

Config load check:

```text
pjm_current_validation_nhits_normal_cap
['nhits_normal_cap']
nhits_normal_cap
{'enabled': True, 'window_observations': 365, 'min_history': 60, 'quantile': 0.95, 'fallback_quantile': 0.975, 'iqr_multiplier': 3.0}
{'actual_daily_max_quantile': 0.95, 'low_risk_score_column': 'spike_score', 'low_risk_threshold': 0.5, 'low_risk_aggregation': 'mean'}
```

## Not Run

Paused per instruction because these can involve model training or require
trained predictions:

```powershell
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
& D:\pjm_remaster\.venv\Scripts\python.exe scripts\experiments\spike_filter_diagnostics.py --config configs\experiments\pjm_current_validation_nhits_normal_cap.yaml --split validation
```

## Current Git State

Committed and pushed implementation range:

- `90f6a7b feat: add normal-day relative error diagnostics`
- `ab70eb9 fix: compute normal-day mean bias correctly`
- `3a92e7e test: cover normal-day relative error quantiles`
- `0179a3f feat: write normal-day diagnostics during evaluation`
- `3c120c4 chore: drop unused normal-day diagnostics assignment`
- `32adf78 feat: add normal-day fields to experiment scorecards`
- `2889442 fix: pass normal-day diagnostics into scorecards`
- `d96d0ca exp: add NHITS normal-day cap validation config`
- `4a796c9 fix: keep normal-cap config paths project-relative`

Known untracked items after verification:

- `.tmp_pytest_nhits_normal_full/`
- `.tmp_pytest_review/`
- `.tmp_pytest_review_task3/`
- `configs/experiments/pjm_current_validation_nhits_spike_filtered_target.yaml`

The untracked spike-filtered placeholder config was not reused or staged.

## Next Step When Training Is Allowed

Run validation backtest for `nhits_normal_cap`, then evaluate and inspect the
normal-day scorecard fields against the current validation scorecard.

Interpretation priority after artifacts exist:

1. `actual_normal_day_q50_wape`
2. `forecast_low_risk_day_q50_wape`
3. normal/low-risk median and p75 APE
4. `q50_wape_20_30` and `q50_wape_30_50`
5. guardrails: global pinball, high-price tail coverage, crossing rate, and
   normal-hour interval width
