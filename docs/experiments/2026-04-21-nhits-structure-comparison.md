# NHITS vs NBEATSx Probabilistic Structure Comparison

Date: 2026-04-21

## Scope

This experiment compares the current probabilistic NBEATSx pipeline against an NHITS candidate using the same PJM/COMED data protocol, rolling retrain policy, quantile grid, postprocessing, CDF reconstruction, and Student-t copula scenario evaluation.

The comparison intentionally uses fresh artifact directories rather than `artifacts_current` to avoid reusing previous prediction or resume chunks.

## Configurations

- `configs/experiments/pjm_current_full_compare_nbeatsx.yaml`
- `configs/experiments/pjm_current_full_compare_nhits.yaml`
- `configs/experiments/pjm_current_full_compare_nhits_steps1000.yaml`
- `configs/experiments/pjm_current_full_compare_nhits_steps1200.yaml`

Common settings:

- Rolling validation and test backtests with the project weekly retrain protocol.
- Quantiles: `0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99`.
- Loss: `HuberMQLoss`, `delta=0.75`.
- Ensemble: two members with seed offsets `0` and `11`.
- Quantile postprocess: monotonic enforcement plus asymmetric hourly CQR.
- Scenario evaluation: Student-t copula fitted from validation PIT pseudo-observations.

Fresh-run sanity checks:

- NBEATSx 600: validation `182` chunks, test `364` chunks.
- NHITS 600: validation `182` chunks, test `364` chunks.
- NHITS 1000: validation `182` chunks, test `364` chunks.
- NHITS 1200: validation `182` chunks, test `364` chunks.

## Test Results

| model | post pinball | raw crossing | post CRPS | post width 80 | post width 90 | post width 98 | energy score |
|---|---:|---:|---:|---:|---:|---:|---:|
| NBEATSx 600 | 3.6156 | 59.65% | 8.3295 | 25.67 | 33.99 | 62.16 | 52.11 |
| NHITS 600 | 3.5970 | 51.48% | 8.2775 | 24.58 | 33.22 | 61.00 | 51.85 |
| NHITS 1000 | 3.6177 | 48.37% | 8.3122 | 24.24 | 32.76 | 60.02 | 52.04 |
| NHITS 1200 | 3.6161 | 47.34% | 8.3058 | 24.06 | 32.46 | 59.62 | 52.06 |

## Interpretation

NHITS at 600 steps is the strongest current backbone candidate. It improves post-calibrated pinball, CRPS, interval width, raw crossing, and scenario energy score relative to the current NBEATSx full compare.

Increasing NHITS to 1000 or 1200 steps does not improve the probabilistic objective. Longer training reduces raw crossing and interval width, but it also reduces raw coverage and does not improve post-calibrated pinball or scenario scores.

The spike-day failure mode remains. On the worst test day, 2026-01-27, the observed maximum was about `927`, while raw `q99` maxima remained far below it:

- NBEATSx 600: `q99 max = 495.98`
- NHITS 600: `q99 max = 458.17`
- NHITS 1000: `q99 max = 431.84`
- NHITS 1200: `q99 max = 426.70`

This suggests that longer training makes NHITS more conservative rather than better at representing extreme upward spikes.

## Decision

- Keep NBEATSx 600 as the strong incumbent baseline.
- Promote NHITS 600 to the main candidate for further probabilistic experiments.
- Do not pursue longer NHITS training budget as the next optimization path.
- Next useful direction: condition spike behavior through forecast-available regime features or regime-aware calibration rather than global training-budget increases.
