# Event-Risk Tail Overlay

Date: 2026-04-28

## Goal

Reduce event-day q99 misses without changing the median forecast or retraining
NHITS.

The overlay is intentionally narrow:

- it only activates on high-risk days
- it only shifts q99/q995
- it does not change q50 or lower quantiles
- it fits all parameters from validation data only
- test is used only once for final reporting

## Leakage Control

The validation/test protocol is:

1. Split validation into an earlier calibration segment and a validation
   holdout.
2. Use the earlier validation segment to fit hourly CQR and event-risk overlay
   parameters.
3. Select the overlay candidate by validation holdout metrics.
4. Apply the selected configuration to test using full validation as the source
   split.

The event-risk score uses prediction-time context already written into the
prediction parquet: `spike_score`. The selected daily risk signal is the mean
`spike_score` for the forecast day.

An initial attempt to fit overlay uplift from in-sample post-CQR residuals
learned zero uplift. That is expected: CQR calibrated and evaluated on the same
calibration segment suppresses q99 positive residuals. The final overlay
therefore fits uplift from raw/monotonic calibration residuals, then applies the
bounded uplift after canonical hourly CQR. Validation holdout decides whether
that raw-residual uplift is too aggressive.

## Selected Configuration

Canonical config section:

```yaml
event_risk_tail_overlay:
  enabled: true
  source_split: "validation"
  risk_score_column: "spike_score"
  risk_aggregation: "mean"
  risk_threshold_quantile: 0.90
  residual_quantile: 1.00
  max_uplift: 50.0
  target_quantiles: [0.99, 0.995]
```

Interpretation:

- activate on daily `spike_score_mean` above the validation 90th percentile
- compute uplift from high-risk raw q99 positive residuals
- use the maximum positive residual, capped at 50
- add that uplift only to q99 and q995

## Experiment Command

```powershell
uv run python scripts\experiments\evaluate_event_risk_tail_overlay.py `
  --validation-prediction artifacts_current\predictions\nhits_tail_grid_weighted_main_validation_seed7.parquet `
  --test-prediction artifacts_current\predictions\nhits_tail_grid_weighted_main_test_seed7.parquet `
  --output-dir artifacts_phase3\event_risk_tail_overlay_sparse `
  --risk-score-column spike_score `
  --risk-aggregations mean `
  --risk-threshold-quantiles 0.80 0.90 0.95 `
  --residual-quantiles 0.99 1.00 `
  --max-uplifts 25 50 `
  --target-quantiles 0.99 0.995 `
  --validation-holdout-days 91 `
  --calibration-min-group-size 24 `
  --interval-coverage-floor 0.10-0.90=0.76 0.05-0.95=0.86 0.01-0.99=0.95
```

## Validation Holdout

Selection rule: prioritize lower pinball without increasing P50 error, then
check q99 excess, worst q99 miss, and width98.

| Variant | Pinball | q99 Exceed | q99 Excess | Worst q99 Under | Daily Max Gap Max | width98 | Active Days | Uplift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hourly CQR | 2.9614 | 3.43% | 0.5818 | 79.63 | 47.36 | 51.14 | 0.0% | 0.00 |
| `overlay_mean_p90_r100_cap50` | 2.9302 | 1.83% | 0.1733 | 33.39 | 1.12 | 69.43 | 39.6% | 46.24 |
| `overlay_mean_p80_r100_cap50` | 2.9378 | 1.24% | 0.1262 | 33.39 | 1.12 | 82.14 | 67.0% | 46.24 |

`overlay_mean_p90_r100_cap50` is the validation winner by pinball. The p80
candidate reduces q99 excess slightly more, but it activates on too many days
and widens the 98% interval more.

## Test Result

After promoting the validation-selected p90/r100/cap50 rule into canonical:

| Variant | Pinball | q50 MAE | q99 Exceed | q99 Excess | Worst q99 Under | Daily Max Gap Mean | Daily Max Gap Max | width98 | Crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hourly CQR baseline | 3.2922 | 10.9858 | 2.23% | 0.8696 | 329.19 | -44.46 | 329.19 | 69.15 | 0 |
| event-risk overlay | 3.2483 | 10.9858 | 1.61% | 0.4707 | 279.19 | -50.37 | 279.19 | 75.05 | 0 |

The overlay improves the distribution objective and upper-tail misses while
leaving P50 unchanged.

## Decision

Promote the event-risk tail overlay into canonical.

The tradeoff is acceptable:

- pinball improves by about `0.044`
- q99 excess mean drops by about `46%`
- worst q99 underprediction improves by `50`
- P50 is unchanged
- crossing remains `0`
- width98 increases from `69.15` to `75.05`

The remaining miss on the largest event day is still large, so this is not the
end of tail work. The next step should enrich risk context beyond `spike_score`
alone, especially `price_lag24_max` and `prior_day_price_max_ramp`, but those
columns must be explicitly written into prediction context before use.
