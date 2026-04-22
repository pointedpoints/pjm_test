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
