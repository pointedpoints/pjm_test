# NHITS Normal-Day Causal Cap Design

## Purpose

Build a focused NHITS experiment for improving the large majority of non-spike
COMED day-ahead hours and days. The model should learn the normal price curve
without being dominated by rare spike magnitudes, while all prediction artifacts
and final evaluation continue to use the original ground-truth `y`.

The first question is narrow: does causal hourly target capping during NHITS
training improve normal-day relative error versus the current NHITS mainline?

## Scope

This design prioritizes:

- NHITS as the only model family in the first experiment.
- Causal hourly cap replacement of spike-like training targets.
- Normal-day and low-risk-day evaluation focused on relative error against
  ground truth.

This design does not include LEAR, LightGBM, XGBoost, frequency features, RAG,
or a separate spike prediction model.

## Normal-Day Definitions

Use two complementary normal-day labels.

### Actual Normal Day

This is an evaluation-only label derived from ground truth after the fact.

A day is an actual normal day when its realized price path does not contain an
extreme price event. The initial rule should be configurable and use a robust
threshold such as:

- daily max `y` below a rolling historical quantile threshold, or
- no hourly `y` above a configured absolute or rolling threshold.

This label is not allowed in training, prediction, or calibration decisions.
It exists to answer whether normal realized days improved.

### Forecast Low-Risk Day

This is a prediction-time label derived only from forecast-available context.

The first rule should use existing `spike_score` context because it is already
constructed with rolling-safe historical percentile scoring. A day is forecast
low-risk when a daily aggregation of `spike_score`, initially mean or max, is
below a configured threshold.

This label can be used for reporting and later operational gating because it
does not require future realized prices.

## Training Approach

Use causal hourly capping during NHITS training.

For each training window:

1. Apply the existing causal spike filter by hour-of-day.
2. Compute `y_train_clean = min(y, spike_threshold)` for detected spike hours.
3. Keep non-spike rows unchanged.
4. Fit NHITS on a copy of the training frame where `y` is replaced by
   `y_train_clean`.
5. Keep `history_df`, `future_df`, prediction parquet output, and evaluation
   targets on original `y`.

This preserves the contiguous hourly sequence required by NHITS and avoids row
deletion. It also keeps raw lag columns unchanged in the first pass so the
experiment isolates the target-capping effect.

## Model Configuration

Create a real NHITS normal-cap experiment config rather than using the current
untracked placeholder file.

The config should:

- have a distinct project name and artifact directory;
- use `benchmark_models: ["nhits_normal_cap"]` or another explicit NHITS model
  name;
- copy the current NHITS mainline settings as the starting point;
- enable `target_filter` on that NHITS model;
- preserve the current time protocol with timezone-naive local timestamps.

The current canonical config must remain unchanged until validation and test
evidence supports promotion.

## Evaluation

Relative error is the primary evaluation surface. MAE is only a secondary
diagnostic because a 5 USD/MWh miss is much more severe when the realized price
is near 20 than when it is near 200.

The normal-day scorecard should report at least:

- `actual_normal_day_q50_wape`
- `actual_normal_day_median_ape`
- `actual_normal_day_p75_ape`
- `actual_normal_day_p90_ape`
- `actual_normal_day_smape`
- `forecast_low_risk_day_q50_wape`
- `forecast_low_risk_day_median_ape`
- `forecast_low_risk_day_p75_ape`
- `forecast_low_risk_day_p90_ape`
- `forecast_low_risk_day_smape`

The existing price-bin relative-error slices remain important, especially:

- `10-20`
- `20-30`
- `30-50`
- `50-100`

For these bins, compare WAPE and APE quantiles before looking at MAE.

## Guardrails

The model should not be promoted from normal-day relative error alone. Guardrail
metrics should still be checked:

- global pinball should not degrade materially;
- quantile crossing rate should remain zero after postprocessing;
- q99 and q995 coverage should not collapse in high actual-price regimes;
- normal-hour interval width should not inflate materially;
- spike-filter diagnostics should show that the cap removes plausible spike
  magnitudes rather than broad normal-price movement.

## Success Criteria

A validation candidate is worth testing only if it improves or clearly
directionally improves:

- actual normal-day q50 WAPE;
- forecast low-risk-day q50 WAPE;
- median and p75 APE on normal/low-risk days;
- WAPE in the `20-30` and `30-50` price bins.

It should not materially worsen:

- global pinball;
- normal-day p90 APE;
- high-regime tail coverage;
- crossing rate.

## Risks

Over-capping may teach NHITS a biased-low price curve and make postprocessing
work harder.

Raw price lags may still contain spike magnitudes while the training target is
cleaned. This is acceptable in the first pass because it isolates the target cap,
but a later experiment may need cleaned lag variants if diagnostics point there.

Actual normal-day metrics are evaluation-only. They are useful for diagnosis,
but they must not leak into training, calibration, or operational gating.

## Expected Files

Likely implementation files:

- `src/pjm_forecast/evaluation/normal_day.py`
- `src/pjm_forecast/evaluation/evaluator.py`
- `src/pjm_forecast/evaluation/scorecard.py`
- `src/pjm_forecast/workspace.py`
- `configs/experiments/pjm_current_validation_nhits_normal_cap.yaml`
- `tests/test_normal_day.py`
- `tests/test_scorecard.py`
- `tests/test_model_registry_target_filter.py`

The existing files below should be reused rather than rebuilt:

- `src/pjm_forecast/spike_filter.py`
- `src/pjm_forecast/models/target_filter.py`
- `src/pjm_forecast/models/registry.py`

## Spec Self-Review

- Placeholder scan: no placeholder sections or unresolved decisions remain.
- Consistency check: training uses causal hourly cap; evaluation uses both
  actual normal-day and forecast low-risk-day labels.
- Scope check: this is one implementation unit focused on NHITS normal-day
  relative-error evaluation and target capping.
- Ambiguity check: relative error is primary; MAE is explicitly secondary.
