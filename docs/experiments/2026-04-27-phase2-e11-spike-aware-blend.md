# Phase 2 E11 NHITS Spike-Aware Blend

## Scope

- Plan item: `E11` spike-aware two-stage blend
- Base model: `nhits_tail_grid_weighted_long`
- Goal: improve spike and upper-tail behavior without giving back the `E10`
  center-quality gains.

## Artifacts Used

- Working test config:
  `configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_spike_context_hour_regime.yaml`
- Reproducible comparison script:
  `scripts/experiments/evaluate_nhits_spike_context_calibration.py`
- Generated comparison tables:
  - `artifacts_tmp/nhits_tail_grid_weighted_long_spike_context/analysis/validation_holdout_summary.csv`
  - `artifacts_tmp/nhits_tail_grid_weighted_long_spike_context/analysis/test_summary.csv`

The comparison keeps the same raw `NHITS` tail-expert predictions and varies
only postprocessing:

- `hour_cqr`
- `hour_x_regime` with threshold `0.50`
- `hour_x_regime` with threshold `0.67`

## Validation Holdout

Method:

- Use the existing validation prediction parquet from the spike-context branch.
- Fit calibration on the first `91` validation forecast days.
- Evaluate on the last `91` validation forecast days.

### Results

| variant | mae | pinball | crps | q99 exceed | q99 excess mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hour_cqr` | `10.0626` | `2.9405` | `7.5042` | `3.3425%` | `0.5720` |
| `hour_regime_cqr_t50` | `10.0427` | `2.9985` | `7.5914` | `3.3425%` | `0.6731` |
| `hour_regime_cqr_t67` | `10.0707` | `2.9742` | `7.5541` | `3.3425%` | `0.6203` |
| `raw_monotonic` | `10.0731` | `3.1202` | `7.8100` | `5.8150%` | `0.8900` |

Interpretation:

- `0.67` is dominated and can be dropped.
- `hour_cqr` is best on holdout pinball and CRPS.
- `hour_x_regime@0.50` is best on holdout `MAE`, and all calibrated variants
  are much better than raw monotonic.

## Test Comparison

### Results

| variant | mae | pinball | crps | q99 exceed | q99 excess mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hour_cqr` | `10.9858` | `3.2702` | `8.2408` | `1.1103%` | `0.5212` |
| `hour_regime_cqr_t50` | `10.9801` | `3.2719` | `8.2680` | `1.6941%` | `0.5663` |
| `hour_regime_cqr_t67` | `10.9889` | `3.3253` | `8.3077` | `1.5911%` | `0.8157` |
| `raw_monotonic` | `10.9987` | `3.4478` | `8.4748` | `5.1053%` | `1.3469` |

Interpretation:

- Both calibrated variants are materially better than raw monotonic.
- `hour_cqr` and `hour_x_regime@0.50` are effectively tied on overall pinball.
- `hour_x_regime@0.50` keeps the best `MAE`.
- `hour_x_regime@0.67` is not competitive.

## Supporting Spike-Day View

The pre-existing derived spike-day analysis in
`artifacts_tmp/nhits_tail_grid_weighted_long_spike_context/spike_day_post_metrics.csv`
shows why `hour_x_regime@0.50` remains interesting even though its full-test
pinball is almost tied with `hour_cqr`:

| variant | spike-day q99 exceed | spike-day q99 excess mean | spike-day worst q99 under |
| --- | ---: | ---: | ---: |
| `hour_cqr` | `12.9505%` | `7.9739` | `329.1898` |
| `hour_regime_cqr_t50` | `8.1081%` | `4.9609` | `307.7543` |

This supporting view is not the primary decision source, but it is directionally
consistent with the Phase 2 objective.

## Decision

`E11` is accepted with `hour_x_regime` threshold `0.50` as the working
spike-aware candidate.

Reasoning:

- `0.50` is the best compromise once the objective shifts from generic
  calibration to spike-aware tail protection.
- It preserves the `E10` center-quality level:
  - `MAE 10.9987 -> 10.9801`
  - `pinball 3.4478 -> 3.2719`
- It sharply improves upper-tail misses versus raw monotonic:
  - `q99 exceed 5.1053% -> 1.6941%`
  - `q99 excess mean 1.3469 -> 0.5663`
- The `hour` variant remains a strong sanity benchmark, but `0.50` is the
  better Phase 2 candidate because it targets spike-day failures more directly.

## Next Step

Move to `E14`.

The next task is to test whether a heterogeneous ensemble / QRA layer can keep
the `E11` spike-aware tail gains while pulling overall pinball and center-side
metrics down again.
