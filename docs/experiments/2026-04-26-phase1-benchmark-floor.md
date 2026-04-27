# Phase 1 Benchmark Floor

## Purpose

Run the first locked Phase 1 benchmark-floor set from the execution plan:

- `Naive-168`
- `LEAR`
- `LightGBM quantile`
- `XGBoost quantile`

Config:

- `configs/experiments/pjm_current_validation_phase1_benchmark_floor.yaml`

## Scope

This branch restores the locked `P50` feature contract for benchmark work:

- `system_load_forecast`
- `price_lag_168`
- `source_lag_168`
- `prior_day_price_max`
- `prior_day_price_max_ramp`

Calibration and scenario evaluation stay off for this benchmark-floor pass.

## Validation Result

| model | MAE | RMSE | sMAPE | pinball |
| --- | ---: | ---: | ---: | ---: |
| `lightgbm_q` | 8.1895 | 15.9728 | 27.6545 | 2.9180 |
| `xgboost_q` | 8.2069 | 15.8954 | 27.8571 | 2.9976 |
| `seasonal_naive` | 15.7900 | 30.1226 | 47.1349 | n/a |
| `lear` | 635.0032 | 7499.4748 | 30.2256 | n/a |

DM highlights:

- `lightgbm_q` vs `seasonal_naive`: strong win on validation
- `xgboost_q` vs `seasonal_naive`: strong win on validation
- `lightgbm_q` vs `xgboost_q`: no significant difference on validation

Quantile diagnostics:

- `lightgbm_q` post pinball: `2.9180`
- `xgboost_q` post pinball: `2.9976`
- Both tree baselines kept `post_crossing_rate = 0`

## Decision

- Keep `lightgbm_q` and `xgboost_q` as active Phase 1 benchmark-floor models.
- Keep `seasonal_naive` as the lower-bound baseline.
- Do not treat the current `LEAR` result as a trustworthy benchmark. The wrapper
  now has a linear fallback for numerical stability, but its validation result is
  far outside a credible range and needs separate debugging before reuse.

## Next Step

Continue the locked queue with:

- `E8`: current neural mainline comparison
- `E9`: `P50`-friendly neural variant
- then `hour_x_regime` calibration comparison
