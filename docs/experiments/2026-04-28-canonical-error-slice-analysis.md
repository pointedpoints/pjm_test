# Canonical NHITS Error Slice Analysis

Date: 2026-04-28

## Scope

This note analyzes the current canonical NHITS hourly CQR result:

- prediction file:
  `artifacts_current/predictions/nhits_tail_grid_weighted_main_test_seed7.parquet`
- calibration source:
  `artifacts_current/predictions/nhits_tail_grid_weighted_main_validation_seed7.parquet`
- feature context:
  `data/processed_current/feature_store.parquet`

The analysis reconstructs the current postprocess contract:

- monotonic quantile enforcement
- hourly asymmetric CQR
- interval coverage floors:
  - `0.10-0.90 = 0.76`
  - `0.05-0.95 = 0.86`
  - `0.01-0.99 = 0.95`

## Summary

The main error is not the ordinary-hour median forecast. It is concentrated in
high-price regimes, daily maxima, winter stress days, summer spike days, and
evening peak hours.

The q99 miss pattern is event-driven rather than uniformly low. A small number
of extreme days accounts for most q99 excess. CQR substantially improves raw
upper-tail coverage, but it cannot fully recover missing conditional signal on
rare event days.

Daily-max failures are partially identifiable ahead of time from available
context. Single load or weather variables are not enough; the useful signal is a
combined risk context using `spike_score`, recent price/ramp state, and load or
weather stress.

## P50 Error Concentration

`bias = y - q50`, so positive values mean the median forecast is too low.

### By Hour

The worst P50 hours are evening peak hours, with hour 7 also elevated:

| Hour | P50 MAE | Bias | q99 Exceed |
|---:|---:|---:|---:|
| 19 | 21.06 | +13.32 | 4.40% |
| 18 | 20.57 | +11.00 | 4.12% |
| 17 | 17.66 | +7.67 | 4.12% |
| 20 | 16.25 | +8.73 | 4.40% |
| 7 | 15.90 | +9.67 | 1.65% |
| 16 | 13.30 | +2.04 | 3.57% |
| 21 | 13.01 | +5.56 | 1.92% |

This suggests P50 work should be peak-hour aware instead of applying a uniform
median correction.

### By Month

The largest monthly errors are in January 2026, then February 2026 and July
2025:

| Month | P50 MAE | Bias | q99 Exceed |
|---|---:|---:|---:|
| 2026-01 | 26.19 | +13.61 | 5.65% |
| 2026-02 | 14.10 | +3.25 | 4.76% |
| 2025-07 | 12.10 | +5.62 | 3.23% |
| 2025-04 | 11.48 | +0.43 | 5.46% |
| 2025-11 | 10.46 | +0.88 | 3.33% |

January is the dominant stress period in the current test split.

### By Regime

The median problem is sharply concentrated in high-price regimes:

| Regime | Count | P50 MAE | Bias | q99 Exceed | q99 Excess |
|---|---:|---:|---:|---:|---:|
| normal | 6988 | 6.97 | -1.45 | 0.67% | 0.031 |
| high | 874 | 12.02 | +5.41 | 2.40% | 0.150 |
| spike | 437 | 18.68 | +12.81 | 4.12% | 0.378 |
| extreme | 437 | 65.47 | +54.49 | 24.94% | 16.206 |
| daily max | 364 | 28.17 | +26.21 | 7.42% | 5.309 |

The normal regime is not the core problem. In normal hours, q50 is slightly high
on average. Extreme and daily-max points dominate the median error.

## q99 Miss Concentration

Post-CQR q99 misses are concentrated in a small set of days:

- total test hours: `8736`
- q99 miss hours: `195`
- q99 miss rate: `2.23%`
- days with at least one q99 miss: `40 / 364`

q99 excess concentration:

| Top Days | Share of q99 Excess |
|---:|---:|
| 1 | 20.6% |
| 3 | 48.3% |
| 5 | 67.1% |
| 10 | 86.5% |
| 20 | 96.7% |

The q99 miss problem is therefore not a broad calibration drift. It is driven by
a small number of event days.

### Worst q99 Excess Days

| Day | y Max | Post q99 Max | Daily Max Gap | Miss Hours | q99 Excess Sum | Spike Score Max | Load Max | Temp Max | Cooling Max | Heating Max | Prior Ramp Max | Price Lag24 Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-24 | 333.05 | 239.32 | 93.73 | 14 | 1563.62 | 0.875 | 14207 | -13.78 | 0.00 | 46.51 | 32.87 | 144.68 |
| 2025-07-29 | 497.18 | 216.54 | 280.64 | 7 | 1302.72 | 0.896 | 19950 | 33.04 | 16.54 | 0.00 | 128.91 | 346.46 |
| 2026-01-26 | 638.14 | 524.98 | 113.15 | 10 | 800.31 | 0.877 | 14546 | -12.88 | 0.00 | 42.28 | 31.77 | 285.10 |
| 2026-02-02 | 294.59 | 119.42 | 175.17 | 14 | 778.64 | 0.858 | 13226 | -2.41 | 0.00 | 29.52 | 55.12 | 112.20 |
| 2025-07-28 | 346.46 | 159.83 | 186.62 | 9 | 648.99 | 0.858 | 20900 | 31.82 | 16.98 | 0.00 | 17.44 | 103.33 |
| 2026-01-27 | 927.38 | 598.19 | 329.19 | 6 | 571.71 | 0.884 | 14342 | -9.89 | 0.00 | 38.64 | 252.51 | 638.14 |

These are event days with either cold stress, heat stress, recent high prices,
or ramp state. The largest miss is still substantial after CQR.

## Raw Model vs CQR

The raw NHITS upper tail is too low; CQR improves it materially:

| Metric | Raw | Post-CQR |
|---|---:|---:|
| q99 exceed | 5.19% | 2.23% |
| q99 excess mean | 1.356 | 0.870 |
| q50 MAE | 11.017 | 10.986 |
| pinball | 3.452 | 3.292 |
| crossing rate | 68.34% | 0.00% |

The post-CQR q99 exceed rate should be interpreted against the configured
`0.01-0.99` coverage floor of `0.95`, whose implied outside-interval rate is
`5%` and roughly `2.5%` per side under symmetric tail allocation. A q99 exceed
rate of `2.23%` is therefore close to the current postprocess target rather
than obviously too high.

However, regime-level metrics show that CQR is not enough for extreme events:

| Regime | Raw q99 Exceed | Post q99 Exceed | Raw q99 Excess | Post q99 Excess |
|---|---:|---:|---:|---:|
| normal | 1.30% | 0.67% | 0.077 | 0.031 |
| high | 8.70% | 2.40% | 0.567 | 0.150 |
| spike | 23.11% | 4.12% | 1.927 | 0.378 |
| extreme | 40.73% | 24.94% | 22.637 | 16.206 |

CQR repairs broad calibration and crossing, but extreme days still need
conditional event information.

## Daily Max Failure Predictability

Post-CQR daily-max failures:

- daily-max fail days: `22 / 364`
- failed days average `y_max`: `256.69`
- non-failed days average `y_max`: `70.22`
- failed days average daily-max gap: `+71.56`
- non-failed days average daily-max gap: `-51.92`

Context comparison:

| Context | Non-Fail Days | Fail Days |
|---|---:|---:|
| spike_score_max | 0.737 | 0.821 |
| load_max | 12371 | 14683 |
| cooling_max | 2.68 | 4.61 |
| heating_max | 14.52 | 17.72 |
| prior_day_price_max_ramp | 26.15 | 58.69 |
| price_lag24_max | 76.91 | 151.91 |

Single variables are imperfect, but high-risk buckets capture a large share of
the q99 excess:

| Top 10% Context Feature | Fail Capture | q99 Excess Share |
|---|---:|---:|
| spike_score_mean | 50.0% | 86.1% |
| spike_score_max | 40.9% | 67.5% |
| price_lag24_max | 36.4% | 61.7% |
| prior_day_price_max_ramp | 31.8% | 41.3% |
| load_max | 22.7% | 34.5% |
| temp_max | 22.7% | 34.6% |

This supports a combined event-risk approach. Load/weather alone are not strong
enough, but `spike_score`, prior-day price level, prior-day ramp, and weather or
load stress together can identify many high-loss days before evaluation.

## Interpretation

1. P50 improvement should not be global.
   The normal regime is not the main issue; global median shifts risk damaging
   ordinary-hour calibration. This matches the bounded median-bias experiment,
   where MAE improved slightly but pinball worsened.

2. q99 miss is event-driven.
   Most q99 excess is concentrated in a few stress days. Wider global CQR would
   make normal days worse and still may not solve the largest events.

3. CQR is doing useful work.
   Raw q99 exceed is too high, and post-CQR brings the aggregate rate close to
   the configured target. The remaining failures are conditional missing-signal
   failures, not a simple postprocess bug.

4. Daily max can be partially recognized.
   The best signals are combined event-risk indicators, especially
   `spike_score_mean`, `spike_score_max`, `price_lag24_max`, and
   `prior_day_price_max_ramp`.

## Recommended Next Experiment

Build a constrained event-risk tail overlay:

- keep q50 unchanged
- keep normal hours unchanged
- activate only for high-risk days or high-risk hours
- use validation to select thresholds and uplift caps
- apply uplift only to q95+ or q99+
- evaluate with the full gate:
  - pinball
  - q99 exceed
  - q99 excess
  - width98
  - daily-max q99 gap
  - crossing rate

Candidate risk features:

- daily `spike_score_mean`
- daily `spike_score_max`
- daily `price_lag24_max`
- daily `prior_day_price_max_ramp`
- `zonal_load_forecast` stress
- heating/cooling stress

The goal should be to reduce top-event q99 excess and daily-max q99 gap without
materially increasing width or degrading overall pinball.
