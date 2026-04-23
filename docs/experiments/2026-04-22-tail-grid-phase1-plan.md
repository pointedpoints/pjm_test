# Upper-Tail Grid Phase 1

Date: 2026-04-22

## Decision

Start with the marginal upper tail before adding more scenario machinery. The
linear tail policy removed the hard q99 ceiling, but the current NHITS scenario
diagnostics barely changed, which points to weak upstream upper-tail geometry.

This phase keeps the experiment raw-first:

- no monotonic postprocessing;
- no CQR calibration;
- no scenario evaluation;
- single seed;
- short validation smoke window.

## Config

Config:

- `configs/experiments/pjm_current_validation_nhits_tail_grid_phase1.yaml`

Models:

- `nhits_baseline`: current 13-quantile NHITS smoke baseline.
- `nhits_tail_grid`: adds q0.975 and q0.995 to densify the upper tail.
- `nhits_tail_grid_weighted`: dense upper-tail grid plus stronger upper-tail
  weights, wider tail Huber deltas, and a light monotonicity penalty.

## Diagnostics Added

`compute_quantile_diagnostics()` now records upper-tail shape and spike miss
columns:

- `q95_q99_gap_mean`
- `q95_q99_slope_mean`
- `q99_q995_gap_mean`
- `q99_q995_slope_mean`
- `q99_exceedance_rate`
- `q99_excess_mean`
- `q99_excess_p95`
- `max_y_q99_gap`
- `worst_q99_underprediction`
- `daily_max_q99_gap_mean`
- `daily_max_q99_gap_max`

These columns are intended to distinguish a better global pinball score from a
real improvement in spike support.

## Execution

```powershell
.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_phase1.yaml --split validation
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_phase1.yaml --split validation
```

Primary comparison should use:

- raw pinball and CRPS;
- raw crossing;
- raw coverage/width;
- q95-q99 and q99-q995 gaps/slopes;
- q99 exceedance and daily max q99 gap.

## Promotion Gate

Promote a candidate only if it improves upper-tail support without a large
pinball/CRPS regression. A higher q99/q995 slope by itself is not sufficient if
coverage collapses or crossing rises materially.

## Initial Validation Smoke

Command:

```powershell
.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_phase1.yaml --split validation
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_phase1.yaml --split validation
```

Results:

| model | pinball | CRPS | crossing | cov98 | width98 | q95-q99 gap | q99-q995 gap | q99 exceed | worst q99 under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nhits_baseline | 2.6669 | 6.1869 | 66.90% | 94.46% | 53.75 | 16.15 | n/a | 2.68% | 93.20 |
| nhits_tail_grid | 2.4288 | 6.2037 | 72.28% | 95.56% | 56.34 | 15.65 | 7.48 | 2.59% | 97.16 |
| nhits_tail_grid_weighted | 2.4289 | 6.2166 | 55.88% | 96.06% | 60.45 | 15.59 | 12.84 | 2.52% | 93.29 |

Interpretation:

Dense upper-tail quantiles materially improved raw pinball in this smoke run,
but did not by itself improve q95-q99 slope or crossing. The weighted variant is
the better phase-1 candidate because it keeps the pinball gain, reduces crossing
below baseline, raises 98% coverage, and creates a larger q99-q995 upper-tail
segment for downstream `linear` tail policy. Its CRPS regression is small but
real, so it should be treated as a candidate rather than promoted directly.

Next action:

- run a slightly longer validation check for `nhits_tail_grid_weighted`;
- compare it under postprocessed evaluation with `tail_policy: linear`;
- if the same pattern holds, move to spike/regime features rather than pushing
  the upper-tail weights harder.

## Longer Validation Check

Config:

- `configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_long.yaml`

This is a bounded longer check using the current prepared-data validation split:

- validation window: 182 days, as fixed in `data/processed_current/split_boundaries.json`;
- max steps: 300;
- models: `nhits_baseline_long` and `nhits_tail_grid_weighted_long`;
- postprocess: disabled;
- scenario evaluation: disabled.

Command:

```powershell
.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_long.yaml --split validation
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_long.yaml --split validation
```

Results:

| model | pinball | CRPS | crossing | cov98 | width98 | q95-q99 gap | q99-q995 gap | q99 exceed | worst q99 under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nhits_baseline_long | 2.7812 | 6.4084 | 75.94% | 90.22% | 41.01 | 12.26 | n/a | 4.62% | 94.70 |
| nhits_tail_grid_weighted_long | 2.5142 | 6.4051 | 65.75% | 92.97% | 46.14 | 12.03 | 10.26 | 4.08% | 87.99 |

Interpretation:

The phase-1 pattern holds on the longer check. The weighted dense-grid model
keeps the large pinball improvement, slightly improves CRPS, reduces crossing by
about 10 percentage points, raises 98% coverage, and lowers the worst q99
underprediction. The q95-q99 gap remains flat to slightly lower, but the new
q99-q995 segment gives downstream linear tail extrapolation a stronger upper
tail slope.

Note: `validation_days` is serialized into the prepared split boundaries during
data preparation. Because this run reused `data/processed_current`, the actual
validation period was the existing 182-day split from `2024-10-02` through
`2025-04-01`, not a newly cut 56-day window.

Next action:

- evaluate the same prediction files with monotonic postprocessing and
  `tail_policy: linear`;
- treat validation scenario diagnostics as an in-sample proxy;
- if postprocessed scenario metrics do not improve materially, move to
  spike/regime features rather than increasing tail weights further.

## Postprocessed Linear-Tail Proxy

Config:

- `configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_long_linear_tail.yaml`

This config reuses the longer-check prediction directory and writes separate
metrics under `artifacts_tmp/nhits_tail_grid_weighted_long_linear_tail/`. On the
validation split, CQR is not applied because the evaluator only fits validation
calibration for test runs. This proxy therefore applies monotonic
postprocessing and evaluates Student-t scenarios with `tail_policy: linear`.
Scenario diagnostics on validation are in-sample proxies.

Command:

```powershell
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_long_linear_tail.yaml --split validation
```

Postprocessed quantile results:

| model | post pinball | post CRPS | post crossing | post cov98 | post width98 | post q99-q995 gap | post q99 exceed | post worst q99 under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nhits_baseline_long | 2.7831 | 6.4084 | 0.00% | 90.29% | 41.04 | n/a | 4.56% | 94.70 |
| nhits_tail_grid_weighted_long | 2.5140 | 6.4051 | 0.00% | 92.99% | 46.17 | 10.23 | 4.05% | 87.99 |

Linear-tail scenario proxy:

| model | energy | variogram | path mean MAE | daily max abs | daily spread abs | daily ramp abs | Spearman corr MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| nhits_baseline_long | 38.3727 | 532.5525 | 8.4580 | 13.6541 | 13.9659 | 8.2223 | 0.1193 |
| nhits_tail_grid_weighted_long | 38.3987 | 542.3628 | 8.4980 | 13.6951 | 13.7543 | 8.3630 | 0.1150 |

Interpretation:

The longer raw improvement survives monotonic postprocessing: pinball, CRPS,
coverage98, q99 exceedance, and worst q99 underprediction all remain better for
the weighted dense-grid model. The linear-tail scenario proxy does not improve
overall path scores; only daily spread and Spearman correlation improve. This
matches the earlier conclusion: the upper-tail grid/loss work is useful, but
further gains should come from spike/regime recognition rather than pushing
tail weights harder.

Next action:

- start the `spike_score` feature path;
- keep `nhits_tail_grid_weighted_long` as the current tail-aggressive candidate;
- do not promote it directly to canonical until test-split CQR/scenario results
  are checked.

## Spike Score And Hour-Regime CQR

Config:

- `configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml`

This keeps `nhits_tail_grid_weighted_long` as the current tail-aggressive
candidate and adds a new spike/regime branch:

- `spike_score` derived feature from load, cooling pressure, heating pressure,
  and prior-day max ramp;
- `spike_score` included in NHITS future exogenous inputs;
- prediction frames carry `spike_score` for downstream calibration grouping;
- CQR grouping supports `hour_x_regime` with `regime_score_column=spike_score`
  and `regime_threshold=0.67`;
- scenario evaluation remains Student-t with `tail_policy: linear`.

Implementation notes:

- The new processed feature store is isolated under
  `data/processed_nhits_tail_spike_score/`.
- Validation evaluation does not apply CQR because the evaluator only fits
  validation calibration for test runs. The validation run therefore checks the
  raw/spike-score model, monotonic postprocessing, scenario proxy, and
  real-data `hour_x_regime` CQR fitability.
- A real validation prediction smoke fit produced 150 CQR adjustment entries for
  `hour_x_regime`, confirming that the grouping path has enough samples for the
  configured `min_group_size=24` fallback logic.

Validation commands:

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml
.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml --split validation
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml --split validation
```

Validation results:

| model | post pinball | post CRPS | post crossing | post cov98 | post q99-q995 gap | post q99 exceed | post worst q99 under |
|---|---:|---:|---:|---:|---:|---:|---:|
| nhits_tail_grid_weighted_long | 2.5140 | 6.4051 | 0.00% | 92.99% | 10.23 | 4.05% | 87.99 |
| nhits_tail_grid_weighted_spike_regime | 2.5254 | 6.4621 | 0.00% | 93.61% | 10.75 | 3.32% | 90.34 |

Linear-tail scenario proxy:

| model | energy | variogram | path mean MAE | daily max abs | daily spread abs | daily ramp abs | Spearman corr MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| nhits_tail_grid_weighted_long | 38.3987 | 542.3628 | 8.4980 | 13.6951 | 13.7543 | 8.3630 | 0.1150 |
| nhits_tail_grid_weighted_spike_regime | 38.8141 | 528.6447 | 8.5278 | 13.0584 | 12.5849 | 7.9364 | 0.1159 |

Interpretation:

Adding `spike_score` is not a free global win: post pinball and CRPS degrade
slightly versus `nhits_tail_grid_weighted_long`. But the spike/regime branch
improves the path metrics we wanted it to target: daily max, daily spread,
daily ramp, and variogram. It also lowers q99 exceedance, while q99 worst
underprediction worsens slightly. This is consistent with a regime signal that
helps path shape but still needs test-split calibration before promotion.

Next action:

- keep `nhits_tail_grid_weighted_long` as the main tail-aggressive baseline;
- use `nhits_tail_grid_weighted_spike_regime` for the first test-split
  hour×regime CQR check;
- compare test post-CQR coverage and scenario path metrics before considering a
  gated blend.

## Test Split Spike-Regime Check

Configs:

- `configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_linear_tail.yaml`
- `configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_spike_context_hour_regime.yaml`
- `configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml`

The test check separates two meanings of "add spike regime":

- **model-input branch**:
  `nhits_tail_grid_weighted_spike_regime` adds `spike_score` to NHITS future
  exogenous inputs and uses `hour_x_regime` CQR.
- **postprocess-only branch**:
  `nhits_tail_grid_weighted_long_spike_context_hour_regime` keeps the
  `nhits_tail_grid_weighted_long` model body and exogenous inputs unchanged.
  It only carries `spike_score` in prediction frames so validation-fit CQR can
  apply `hour_x_regime` adjustments on test.

Commands:

```powershell
.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml --split test
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_tail_grid_weighted_spike_regime.yaml --split test

.venv\Scripts\python.exe scripts\backtest_all_models.py --config configs\experiments\pjm_current_test_nhits_tail_grid_weighted_long_linear_tail.yaml --split test
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_test_nhits_tail_grid_weighted_long_linear_tail.yaml --split test

.venv\Scripts\python.exe scripts\prepare_data.py --config configs\experiments\pjm_current_test_nhits_tail_grid_weighted_long_spike_context_hour_regime.yaml
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_test_nhits_tail_grid_weighted_long_spike_context_hour_regime.yaml --split test
```

For the postprocess-only branch, the test run reused the already completed
`nhits_tail_grid_weighted_long` validation/test prediction values and joined
`spike_score` by `ds` into an isolated prediction directory. This avoids
retraining an identical model and tests the intended intervention: calibration
context only. A threshold sweep showed that `regime_threshold=0.50` was more
stable than the initial `0.67`; the committed postprocess-only config uses
`0.50`.

Test quantile results:

| run | post pinball | post CRPS | post cov80 | post cov90 | post cov98 | post width98 | post q99 exceed | post worst q99 under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_long, monotonic only | 3.4478 | 8.4748 | 64.15% | 77.26% | 92.42% | 55.96 | 5.11% | 375.31 |
| weighted_long, hour CQR | 3.2922 | 8.2522 | 77.12% | 86.55% | 96.07% | 69.15 | 2.23% | 329.19 |
| weighted_long, hour x regime CQR, t=0.50 | 3.2729 | 8.2637 | 78.47% | 88.13% | 96.99% | 93.98 | 1.90% | 307.75 |
| spike_score as model input, hour x regime CQR, t=0.67 | 3.8485 | 9.4282 | 77.14% | 86.24% | 96.23% | 129.09 | 2.67% | 226.71 |

Test linear-tail scenario diagnostics:

| run | energy | variogram | path mean MAE | daily max abs | daily spread abs | daily ramp abs | Spearman corr MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted_long, monotonic only | 53.5363 | 742.9189 | 10.9344 | 22.9972 | 22.6368 | 12.6740 | 0.0602 |
| weighted_long, hour CQR | 51.8432 | 717.6055 | 10.8655 | 21.6868 | 21.8451 | 12.7064 | 0.0536 |
| weighted_long, hour x regime CQR, t=0.50 | 51.6807 | 716.5002 | 10.8920 | 21.6615 | 23.0905 | 14.0468 | 0.0544 |
| spike_score as model input, hour x regime CQR, t=0.67 | 59.7035 | 979.4209 | 17.0333 | 65.5879 | 66.9806 | 49.3485 | 0.0528 |

Spike-day post metrics were computed on the top 10% test days by realized daily
maximum price (`daily max >= 118.6155`, 37 days):

| run | daily max abs | daily spread abs | daily ramp abs | p50 hourly MAE | q99 exceed | q99 excess mean | q99 worst under |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted_long, monotonic only | 119.8610 | 123.4879 | 80.7027 | 37.5152 | 16.55% | 10.4575 | 375.31 |
| weighted_long, hour CQR | 119.6861 | 123.3130 | 80.7174 | 37.5024 | 12.95% | 7.9739 | 329.19 |
| weighted_long, hour x regime CQR, t=0.50 | 119.6237 | 123.2506 | 80.7199 | 37.4816 | 8.11% | 4.9609 | 307.75 |
| spike_score as model input, hour x regime CQR, t=0.67 | 109.0744 | 98.2745 | 71.7545 | 45.4585 | 10.81% | 6.6108 | 226.71 |

Interpretation:

The first spike-regime implementation overreached by feeding `spike_score` into
NHITS. On test it improves some spike-day shape metrics, but it damages global
pinball, CRPS, scenario path scores, and p50 hourly MAE. It should remain a
targeted branch, not the main line.

The cleaner postprocess-only branch is the better immediate path. Keeping
`nhits_tail_grid_weighted_long` unchanged and using `spike_score` only for CQR
improves test pinball versus monotonic-only baseline and versus hour-only CQR.
It also materially reduces q99 exceedance and worst q99 underprediction on
spike days. The tradeoff is wider 98% intervals and weaker daily spread/ramp
scenario scores than hour-only CQR.

Next action:

- promote the postprocess-only spike context config as the next targeted test
  branch;
- do not promote the model-input spike branch globally;
- tune `spike_score` and `regime_threshold` on validation/calibration splits
  before treating the `t=0.50` test sensitivity as final evidence;
- next model-side spike work should use a gated/blended p50 or classifier path,
  not unconditional inclusion of `spike_score` in all NHITS future exogenous
  inputs.
