# P50 Future Price Lag Ablation

This is the first evidence loop from
`D:/BaiduNetdiskDownload/comed_da_lmp_improvement_process.md` flow 4.

## Purpose

Retest weekly price lag as a horizon-aligned future-known feature:

```text
future_price_lag_k(ds) = y(ds - k hours)
```

This is different from `price_lag_168` as a history-window exogenous signal.
For day-ahead forecasting, the horizon-aligned lag gives each future hour the
same hour from the previous week.

## Configs

- `configs/experiments/pjm_current_p50_futr_lag168.yaml`
- `configs/experiments/pjm_current_p50_futr_lag168_336.yaml`

Both configs keep the NHITS dense upper-tail setup but disable CQR calibration
and scenario evaluation. Evaluation still applies monotonic postprocess and
writes scalar metrics, quantile diagnostics, and regime metrics.

## Commands

Run validation first:

```powershell
uv run python scripts\prepare_data.py --config configs\experiments\pjm_current_p50_futr_lag168.yaml
uv run python scripts\run_pipeline.py --config configs\experiments\pjm_current_p50_futr_lag168.yaml --split validation

uv run python scripts\prepare_data.py --config configs\experiments\pjm_current_p50_futr_lag168_336.yaml
uv run python scripts\run_pipeline.py --config configs\experiments\pjm_current_p50_futr_lag168_336.yaml --split validation
```

Only run test after validation has a clear direction.

## Promotion Gate

Keep a future lag only if validation improves P50 or tail behavior without
moving normal-regime P50 materially backward:

- `p50_mae` improves, or does not worsen while tail metrics improve,
- `normal` regime `p50_mae` does not materially degrade,
- `p95_pinball` or `p99_pinball` improves,
- `q99_excess_mean` or `worst_q99_underprediction` declines,
- improvement is not only width inflation.

If `future_price_lag_168 + future_price_lag_336` does not beat
`future_price_lag_168`, do not add more price lag variants.

## Validation Results

Executed on 2026-04-27 with single seed `7`, no CQR calibration, monotonic
postprocess enabled.

| Config | MAE | Pinball | q99 exceed | q99 excess mean | worst q99 under | width 98 |
|---|---:|---:|---:|---:|---:|---:|
| `nhits_tail_grid_weighted_long` baseline | 8.4213 | 2.5142 | 4.6245% | 0.5805 | 94.7047 | 41.0050 |
| `future_price_lag_168` | 8.3691 | 2.5188 | 4.0522% | 0.5742 | 106.5220 | 43.4284 |
| `future_price_lag_168 + 336` | 8.6927 | 2.6464 | 4.7161% | 0.6226 | 86.8223 | 42.7157 |

Regime metrics for `future_price_lag_168`:

| Regime | P50 MAE | P99 pinball | q99 excess mean |
|---|---:|---:|---:|
| normal | 6.2568 | 0.2506 | 0.0190 |
| extreme | 35.9524 | 8.9975 | 8.6920 |
| daily_max | 17.0332 | 4.5936 | 4.5309 |

## Decision

`future_price_lag_168` is a mixed candidate: it slightly improves MAE and q99
exceed/excess versus the no-lag NHITS tail baseline, but it worsens overall
pinball and worst q99 underprediction. It should not be promoted directly.

`future_price_lag_168 + future_price_lag_336` fails the validation gate. It
worsens MAE, pinball, q99 exceed, q99 excess, and extreme-regime P50 MAE. Do
not add more weekly price lag variants until a stronger gating or P50-only use
case is justified.

Next step: test prior-day price state as a separate feature family, not combined
with the failed `168 + 336` lag expansion.
