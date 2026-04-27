# Phase 2 E10 NHITS Tail Expert

## Scope

- Plan item: `E10` weighted long-grid tail expert
- Model family: `NHITS`
- Objective: establish a dedicated upper-tail expert before any spike-aware
  blend logic is introduced.

## Configs Used

- Validation:
  `configs/experiments/pjm_current_validation_nhits_tail_grid_weighted_long_linear_tail.yaml`
- Test:
  `configs/experiments/pjm_current_test_nhits_tail_grid_weighted_long_linear_tail.yaml`

The validation config compares:

- `nhits_baseline_long`
- `nhits_tail_grid_weighted_long`

The test config evaluates the selected tail-expert model only.

## Validation Evidence

### Scalar metrics

| model | mae | pinball |
| --- | ---: | ---: |
| `nhits_baseline_long` | `8.3599` | `2.7831` |
| `nhits_tail_grid_weighted_long` | `8.4362` | `2.5140` |

### Tail diagnostics

| model | post q99 exceed | post q99 excess mean | post daily max q99 gap mean | post crossing |
| --- | ---: | ---: | ---: | ---: |
| `nhits_baseline_long` | `4.5559%` | `0.5724` | `-9.6254` | `0` |
| `nhits_tail_grid_weighted_long` | `4.0522%` | `0.5399` | `-9.9915` | `0` |

Interpretation:

- The weighted long-grid model clearly improves probabilistic quality on
  validation.
- `MAE` regresses slightly, but this is acceptable for `E10` because the
  branch is explicitly a tail expert rather than a `P50` mainline.
- The tail-expert variant improves both exceedance rate and exceedance size.

## Test Evidence

### Scalar metrics

| model | mae | pinball | crps |
| --- | ---: | ---: | ---: |
| `nhits_tail_grid_weighted_long` | `10.9987` | `3.4478` | `8.4748` |

### Tail diagnostics

| metric | value |
| --- | ---: |
| post crossing | `0` |
| post q99 exceed | `5.1053%` |
| post q99 excess mean | `1.3469` |
| post q99 excess p95 | `0.2545` |
| post worst q99 underprediction | `375.3112` |
| post daily max q99 gap mean | `-6.6044` |

## Reference Test Baselines

| model | mae | pinball |
| --- | ---: | ---: |
| canonical `nbeatsx` | `11.3003` | `3.6463` |
| `nhits_q50w150` | `10.9673` | `3.2883` |

Interpretation:

- `E10` improves materially over the old canonical `nbeatsx` on test pinball.
- `E10` does not beat the best existing `NHITS` test candidate on center-side
  quality.
- `E10` is good enough to serve as the Phase 2 tail-expert base, but not good
  enough to claim the spike problem is solved.

## Decision

`E10` is accepted as the working `NHITS` tail-expert baseline for Phase 2.

That means:

- keep `nhits_tail_grid_weighted_long` as the dedicated tail expert
- do not promote it as a new universal mainline by itself
- carry it forward into `E11` for spike-aware gating and upper-tail blending

## Next Step

Move to `E11`.

The target for `E11` is not another generic reweighting pass. It is a
spike-aware blend that uses this `NHITS` tail expert only when the spike gate
is active, while preserving the center of the distribution elsewhere.
