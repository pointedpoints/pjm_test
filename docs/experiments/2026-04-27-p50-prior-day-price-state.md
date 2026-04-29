# P50 Prior-Day Price State

This is the second evidence loop from
`D:/BaiduNetdiskDownload/comed_da_lmp_improvement_process.md` flow 5.

## Purpose

Test whether previous complete-day price state helps the NHITS dense quantile
model understand recent market conditions better than a single lag value.

## Config

- `configs/experiments/pjm_current_p50_price_state.yaml`

The config adds these model `futr_exog` features:

- `prior_day_price_max`
- `prior_day_price_spread`
- `prior_day_price_max_ramp`

It keeps CQR calibration and scenario evaluation disabled, with monotonic
postprocess enabled, matching the future-lag ablation setup.

## Commands

```powershell
uv run python scripts\prepare_data.py --config configs\experiments\pjm_current_p50_price_state.yaml
uv run python scripts\backtest_all_models.py --config configs\experiments\pjm_current_p50_price_state.yaml --split validation
uv run python scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_p50_price_state.yaml --split validation
uv run python scripts\export_report_assets.py --config configs\experiments\pjm_current_p50_price_state.yaml --split validation
```

## Gate

Promote only if validation improves P50 MAE or tail metrics without materially
hurting normal-regime P50:

- compare against `nhits_tail_grid_weighted_long`,
- check `validation_regime_metrics.csv` for normal/extreme/daily_max,
- do not combine with future price lags unless this feature family wins alone.

## Validation Results

Executed on 2026-04-27 with single seed `7`, no CQR calibration, monotonic
postprocess enabled.

| Config | MAE | Pinball | q99 exceed | q99 excess mean | worst q99 under | width 98 |
|---|---:|---:|---:|---:|---:|---:|
| `nhits_tail_grid_weighted_long` baseline | 8.4213 | 2.5142 | 4.6245% | 0.5805 | 94.7047 | 41.0050 |
| `future_price_lag_168` | 8.3691 | 2.5188 | 4.0522% | 0.5742 | 106.5220 | 43.4284 |
| `future_price_lag_168 + 336` | 8.6927 | 2.6464 | 4.7161% | 0.6226 | 86.8223 | 42.7157 |
| `prior_day_price_state` | 8.1618 | 2.4108 | 3.4799% | 0.4900 | 98.1887 | 47.2279 |

Regime metrics:

| Regime | P50 MAE | P99 pinball | q99 excess mean |
|---|---:|---:|---:|
| normal | 6.1217 | 0.2537 | 0.0064 |
| extreme | 34.6993 | 8.6536 | 8.2735 |
| daily_max | 17.6531 | 4.1665 | 4.0941 |

## Validation Decision

`prior_day_price_state` passes validation strongly enough to run test. It
improves MAE, pinball, q99 exceedance, q99 excess, and extreme/daily-max tail
metrics versus the no-state NHITS tail baseline. The main cost is wider q98
intervals, so test promotion must check whether the gain is robust or mostly
width inflation.

## Test Results

Executed on 2026-04-27 with the same config.

| Config | MAE | Pinball | q99 exceed | q99 excess mean | worst q99 under | width 98 |
|---|---:|---:|---:|---:|---:|---:|
| `nhits_tail_grid_weighted_long` + linear tail | 10.9987 | 3.4478 | 5.1053% | 1.3469 | 375.3112 | 55.9650 |
| `nhits_tail_grid_weighted_long` + spike context | 10.9801 | 3.2729 | 1.9002% | 0.5773 | 307.7543 | 93.9779 |
| `nhits_q50w150` | 10.9673 | 3.2883 | 1.8544% | 0.5712 | 314.4821 | 95.0639 |
| `prior_day_price_state` | 12.2918 | 3.7841 | 3.9034% | 0.8016 | 233.3980 | 113.2470 |

Regime metrics for `prior_day_price_state` on test:

| Regime | P50 MAE | P99 pinball | q99 excess mean |
|---|---:|---:|---:|
| normal | 7.8055 | 0.4109 | 0.0573 |
| extreme | 75.2757 | 21.5882 | 13.0222 |
| daily_max | 28.2339 | 6.7674 | 6.2068 |

## Test Decision

Do not promote `prior_day_price_state` into the main model. The validation gain
does not generalize to test: MAE and pinball are worse than the existing NHITS
tail candidates, and width inflation is large. The lower worst q99
underprediction is not enough to justify the broad degradation.

Keep prior-day price state available for `spike_score` or later gated-tail
diagnostics, but not as a default NHITS `futr_exog` feature.
