# Spike-Filtered Normal Training Design

## Purpose

The next experiment separates normal-price learning from spike behavior without deleting timestamps from the hourly series. The normal model should train against a cleaned target that reduces the influence of extreme price spikes, while final validation and test evaluation still use the original `y`.

This is intended to answer one concrete question first: does removing spike magnitude from the normal-model training target improve ordinary-regime q50 and probabilistic quality without making tail diagnostics worse?

## Selected Approach

Use a causal spike-replacement target:

- Keep the canonical panel target column `y` unchanged.
- Build an additional cleaned target column, initially named `y_train_clean`.
- Build an `is_training_spike` indicator.
- Build `spike_residual = y - y_train_clean`.
- Train selected experiment models on `y_train_clean` by temporarily mapping it to model input column `y` inside training windows.
- Preserve original `y` in prediction artifacts and all evaluation metrics.

This avoids deleting rows. Deleting rows is acceptable for simple tabular learners but unsafe as the first shared design because NHITS/NBEATSx training depends on contiguous hourly windows and price lag semantics.

## Spike Detection

The detector must be causal. A timestamp can only be labeled using information available before or at that timestamp in a training window; it must not use validation or test future outcomes.

The initial detector should be simple and reproducible:

- Group by hour-of-day, because price scale and spike behavior differ by hour.
- For each hour group, use a rolling historical window over prior observations.
- Estimate a high reference threshold from historical values.
- Label a point as a training spike when `y` exceeds the historical threshold.
- Do not label the first insufficient-history rows as spikes.

The first detector default is a 365-observation rolling window within each hour group, a minimum of 60 prior observations, and a threshold of rolling q95 plus `3 * IQR`. If `IQR` is too small, use rolling q975 as the fallback threshold. These values should be configurable, but the first validation run should use them unchanged.

## Replacement Rule

For each detected spike hour:

- Replace `y` with the causal cap value used by the detector, or with a causal hour-group rolling median plus capped spread.
- Store the removed magnitude in `spike_residual`.
- Leave non-spike rows unchanged.

The replacement value should be smooth enough for normal-price training but still realistic for the same hour and season. It must be computed from history only.

## Model Integration

The first implementation should introduce this as an optional model-training wrapper rather than changing the base feature store contract for every model.

Recommended flow:

1. Prepare the normal feature frame as today.
2. At model fit time, if the model config enables spike-filtered target training, create a copy of the training frame.
3. Compute `y_train_clean`, `is_training_spike`, and `spike_residual` on that training frame.
4. Replace the copied frame's `y` with `y_train_clean` before passing it into the wrapped model's `fit`.
5. Keep `history_df` and `future_df` prediction calls compatible with the existing `ForecastModel` interface.
6. Prediction artifacts continue to merge future original `y`, not `y_train_clean`.

This keeps `Workspace`, `run_rolling_backtest`, and evaluator output protocols mostly intact.

## First Experiment Scope

Start with quick, low-cost baselines:

- `lightgbm_q` with spike-filtered target.
- `xgboost_q` with spike-filtered target.
- Optional native or epftoolbox LEAR only after the wrapper behavior is stable.

Then, if the tabular experiment is directionally useful, add an NHITS filtered-target variant. NHITS should use replacement, not row deletion.

## Evaluation

Evaluate on validation before test. Do not judge by a few selected days.

Required validation outputs:

- Existing scalar metrics: MAE, RMSE, sMAPE, pinball where available.
- Existing relative-error scorecard slices: all, `10-20`, `20-30`, `30-50`, `50-100`.
- Existing tail regime diagnostics, especially actual `>p99` q99/q995 coverage for models that expose high quantiles.
- A small diagnostic summary for the training filter: spike count, spike share, mean/max removed residual, and count by hour.

Promotion criteria:

- Ordinary-regime q50 WAPE and sMAPE improve or stay close to the unfiltered baseline.
- Pinball does not degrade materially.
- Tail diagnostics do not collapse.
- The filter removes plausible spike magnitudes rather than broad normal-price movement.

## Risks

The main risk is over-cleaning. If the detector marks normal high-price regimes as spikes, the model will learn a biased low median and later postprocessing will have to compensate too much.

The second risk is inconsistent history. If price lags still contain raw spikes while the target is cleaned, the model may learn confusing relationships. The first version should keep raw lag features so the experiment is minimal and comparable. A later version can test cleaned lag variants if diagnostics show this is the bottleneck.

The third risk is tail under-support. Filtering improves normal behavior but may make the learned upper quantiles even weaker. This is why `spike_residual` is kept explicitly for later tail/RAG/overlay work instead of discarded.

## Non-Goals

This design does not implement the full two-stage spike occurrence and residual model.

This design does not replace the existing event-risk tail overlay.

This design does not change the repository time protocol or remap timestamps.

This design does not promote filtered training to the canonical current config until validation evidence supports it.

## Expected Files

Likely implementation files:

- `src/pjm_forecast/spike_filter.py` for causal spike labeling and target replacement.
- `src/pjm_forecast/models/target_filter.py` for a small `ForecastModel` wrapper.
- `src/pjm_forecast/models/registry.py` to wrap configured experiment models.
- `src/pjm_forecast/evaluation/spike_filter_diagnostics.py` for filter diagnostics.
- `src/pjm_forecast/workspace.py` and `ArtifactStore` for writing diagnostics if needed.
- `configs/experiments/pjm_current_validation_spike_filtered_tree.yaml` for the first validation experiment.
- Focused tests under `tests/`.

## Implementation Defaults

- Window length: 365 prior days per hour group when available.
- Minimum history: 60 observations per hour group.
- Threshold: rolling hour-group q95 plus `3 * IQR`, with a fallback to q975 when IQR is too small.
- Replacement: threshold cap, so `y_train_clean = min(y, threshold)`.
- Diagnostics: a separate CSV writer attached to evaluation or a lightweight experiment CLI, to avoid modifying prediction artifacts.
