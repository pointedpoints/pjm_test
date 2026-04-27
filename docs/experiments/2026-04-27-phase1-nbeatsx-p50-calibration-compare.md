# Phase 1 NBEATSx P50 Calibration Compare

## Scope

- Plan items: `hour_x_regime` calibration comparison, Phase 1 promotion gate
- Candidate: `nbeatsx_p50_friendly`
- Test objective: check whether postprocessing can preserve the validation
  `MAE` gain while recovering enough probabilistic quality for promotion.

## Configs

- Raw monotonic:
  `configs/experiments/pjm_current_test_phase1_nbeatsx_p50_raw.yaml`
- Hour calibration:
  `configs/experiments/pjm_current_test_phase1_nbeatsx_p50_hour.yaml`
- Hour-by-regime calibration:
  `configs/experiments/pjm_current_test_phase1_nbeatsx_p50_hour_x_regime.yaml`

All three configs share the same regenerated prediction context and carry
`spike_score` in both validation and test prediction files.

## Test Results

| variant | mae | pinball | crps | q50_mae | q99 exceed | post crossing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_monotonic` | `12.2631` | `4.6559` | `9.5483` | `12.2631` | `5.7120%` | `0` |
| `hour` | `12.2600` | `4.4910` | `9.1948` | `12.2600` | `5.1053%` | `0` |
| `hour_x_regime` | `12.2555` | `4.4943` | `9.2012` | `12.2555` | `4.7619%` | `0` |

Interpretation:

- `hour` is the best pinball / CRPS option in this branch.
- `hour_x_regime` is slightly better on `MAE` and `q99 exceed`.
- The two calibrated variants are effectively tied and both remain far below
  the test quality bar needed for promotion.

## Reference Test Baselines

| model | mae | pinball | crps | q99 exceed | post crossing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nhits_q50w150` | `10.9673` | `3.2883` | `8.2926` | `1.8544%` | `0` |
| canonical `nbeatsx` | `11.3003` | `3.6463` | `8.4238` | n/a | `0` |

## Decision

Phase 1 declares **no promotion** for the `P50`-friendly `nbeatsx` branch.

Reasoning:

- Calibration improved this branch relative to its raw test output, but only
  from `pinball 4.6559` down to roughly `4.49`.
- Even after calibration, the branch is still materially worse than both:
  - the current canonical `nbeatsx` test result
  - the existing `NHITS` test candidates
- The branch fails the Phase 1 gate: it wins neither `MAE` nor global
  probabilistic quality on test.

## Consequence

- Close Phase 1 with a no-promotion decision for the `P50`-friendly neural
  path.
- Keep the approved/current `NHITS` probabilistic line as the working mainline.
- Move to the next locked queue item: `E10` tail expert work.
