# Phase 2 E14 QRA Ensemble

## Scope

- Plan item: `E14` heterogeneous ensemble / QRA on approved inputs
- Goal: keep the `E11` spike-aware tail gains while improving overall
  probabilistic quality through a meta-ensemble layer.

## Config And Script

- Config:
  `configs/experiments/pjm_current_phase2_qra_ensemble.yaml`
- Script:
  `scripts/experiments/evaluate_phase2_qra_ensemble.py`

## Members

This QRA run blends three model families / objectives:

1. `nbeatsx_current`
2. `nhits_q50w150`
3. `nhits_tail_spike_context`

Method:

- interpolate each member onto the dense target quantile grid
- fit per-quantile linear QRA on validation
- compare postprocessing variants on a validation holdout
- refit on full validation and evaluate on test

## Validation Holdout

The validation set is split into:

- first `91` forecast days for QRA fitting
- last `91` forecast days for holdout evaluation

### Results

| variant | mae | pinball | crps | q99 exceed |
| --- | ---: | ---: | ---: | ---: |
| `hour_cqr` | `10.1462` | `2.7818` | `7.2429` | `2.5641%` |
| `hour_regime_cqr_t50` | `10.0799` | `2.8106` | `7.2557` | `3.3883%` |
| `hour_regime_cqr_t67` | `10.1394` | `2.8150` | `7.3044` | `2.7473%` |
| `raw_monotonic` | `10.2517` | `2.9035` | `7.5095` | `2.7473%` |

Interpretation:

- `0.67` is again dominated.
- `hour_cqr` is best on holdout pinball / CRPS.
- `hour_regime_cqr_t50` is best on holdout `MAE`.

## Test Results

### Scalar metrics

| variant | mae | pinball | crps |
| --- | ---: | ---: | ---: |
| `hour_regime_cqr_t50` | `11.2424` | `3.1890` | `8.2046` |
| `hour_cqr` | `11.2699` | `3.1952` | `8.2239` |
| `hour_regime_cqr_t67` | `11.2753` | `3.2419` | `8.3168` |
| `raw_monotonic` | `11.3026` | `3.2838` | `8.4247` |

### Tail diagnostics

| variant | q99 exceed | q99 excess mean | daily max q99 gap mean | post crossing |
| --- | ---: | ---: | ---: | ---: |
| `hour_regime_cqr_t50` | `1.2477%` | `0.3021` | `-82.9572` | `0` |
| `hour_cqr` | `0.7898%` | `0.2635` | `-93.8228` | `0` |
| `hour_regime_cqr_t67` | `1.1103%` | `0.4177` | `-79.4320` | `0` |
| `raw_monotonic` | `1.7285%` | `0.4812` | `-38.3504` | `0` |

### Scenario diagnostics

| variant | energy score | variogram score | path mean mae | daily max abs error |
| --- | ---: | ---: | ---: | ---: |
| `hour_regime_cqr_t50` | `51.6290` | `708.7435` | `10.9113` | `20.8853` |
| `hour_cqr` | `51.8616` | `707.6933` | `10.9236` | `21.1285` |
| `hour_regime_cqr_t67` | `52.6919` | `728.4019` | `11.0423` | `21.4033` |
| `raw_monotonic` | `53.6331` | `752.7540` | `11.1879` | `21.5131` |

## Comparison To E11

Reference `E11` winner:

- `nhits_tail_grid_weighted_long` + `hour_x_regime@0.50`
- `MAE 10.9801`
- `pinball 3.2719`
- `CRPS 8.2680`
- `q99 exceed 1.6941%`
- `energy score 51.6810`

`E14` best candidate (`hour_regime_cqr_t50`) changes that to:

- `MAE 11.2424`
- `pinball 3.1890`
- `CRPS 8.2046`
- `q99 exceed 1.2477%`
- `energy score 51.6290`

Interpretation:

- `E14` improves the main probabilistic metrics and upper-tail miss metrics.
- `E14` gives back center accuracy relative to `E11`.
- The tradeoff is real, not noise-level.

## Decision

`E14` is accepted as the best **probabilistic** candidate produced so far.

More precisely:

- keep `hour_regime_cqr_t50` as the working QRA ensemble candidate
- keep `hour_cqr` as a near-tied calibration benchmark
- do not call the center-side regression free: `MAE` worsens relative to the
  best `NHITS` specialist branches

## Consequence

The repo now has two credible Phase 2 endpoints:

- `E11` spike-aware `NHITS` specialist: better center quality
- `E14` QRA ensemble: better global probabilistic quality

This is enough to stop broad exploration and move to promotion judgment:

- if objective priority is pinball / CRPS / tail control, prefer `E14`
- if objective priority is preserving `MAE`, prefer `E11`
