# Bounded Median-Bias Grid

Date: 2026-04-28

## Goal

Test whether a bounded postprocess median-bias layer can improve P50/MAE while
preserving the current canonical hourly CQR distribution behavior.

This is a no-retraining experiment. It reuses the current canonical NHITS
predictions and applies median-bias correction before hourly CQR calibration.

## Inputs

- validation prediction:
  `artifacts_current/predictions/nhits_tail_grid_weighted_main_validation_seed7.parquet`
- test prediction:
  `artifacts_current/predictions/nhits_tail_grid_weighted_main_test_seed7.parquet`
- output:
  `artifacts_phase3/median_bias_grid_canonical_floors/`

The grid uses the same interval coverage floors as the canonical config:

- `0.10-0.90 = 0.76`
- `0.05-0.95 = 0.86`
- `0.01-0.99 = 0.95`

## Command

```powershell
uv run python scripts\experiments\evaluate_median_bias_grid.py `
  --validation-prediction artifacts_current\predictions\nhits_tail_grid_weighted_main_validation_seed7.parquet `
  --test-prediction artifacts_current\predictions\nhits_tail_grid_weighted_main_test_seed7.parquet `
  --output-dir artifacts_phase3\median_bias_grid_canonical_floors `
  --max-abs-adjustments 5 10 20 `
  --validation-holdout-days 91 `
  --min-group-size 24 `
  --group-by hour `
  --interval-coverage-floor 0.10-0.90=0.76 0.05-0.95=0.86 0.01-0.99=0.95
```

## Validation Holdout

| Variant | MAE | Pinball | q99 exceed | q99 excess | width98 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| `hour_cqr` | 10.0626 | 2.9614 | 3.43% | 0.5818 | 51.1386 | 0 |
| `hour_median_bias_cap5` | 9.9639 | 2.9619 | 3.48% | 0.5913 | 51.0256 | 0 |
| `hour_median_bias_cap10` | 9.9669 | 2.9619 | 3.48% | 0.5913 | 51.0256 | 0 |
| `hour_median_bias_cap20` | 9.9669 | 2.9619 | 3.48% | 0.5913 | 51.0256 | 0 |
| `raw_monotonic` | 10.0731 | 3.1202 | 5.82% | 0.8900 | 46.0657 | 0 |

## Test

| Variant | MAE | Pinball | q99 exceed | q99 excess | width98 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| `hour_cqr` | 10.9858 | 3.2922 | 2.23% | 0.8696 | 69.1474 | 0 |
| `hour_median_bias_cap5` | 10.9628 | 3.2948 | 2.24% | 0.8700 | 69.1157 | 0 |
| `hour_median_bias_cap10` | 10.9628 | 3.2948 | 2.24% | 0.8700 | 69.1157 | 0 |
| `hour_median_bias_cap20` | 10.9628 | 3.2948 | 2.24% | 0.8700 | 69.1157 | 0 |
| `raw_monotonic` | 10.9987 | 3.4478 | 5.11% | 1.3469 | 55.9650 | 0 |

## Decision

Do not promote bounded median-bias correction into canonical.

It gives a small test MAE/P50 gain (`10.9858 -> 10.9628`) but slightly worsens
test pinball (`3.2922 -> 3.2948`) and does not improve q99 risk. The validation
holdout shows the same pattern: lower MAE, but slightly worse pinball and upper
tail diagnostics.

The experiment is useful as a repeatable diagnostic tool. It confirms that the
current P50 error is not just a simple hourly location bias that can be corrected
for free after training. Further P50 work should prefer model-side changes that
do not perturb the calibrated tail, such as a small `loss_delta` grid or
separate median-focused experiments kept out of canonical until they beat the
full distribution gate.
