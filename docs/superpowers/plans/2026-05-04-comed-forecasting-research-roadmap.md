# COMED Day-Ahead Probabilistic Forecasting Research Roadmap

Date: 2026-05-04

## 1. Purpose

This document summarizes the current PJM/COMED day-ahead probabilistic forecasting project, the observed model weaknesses, the literature-motivated improvement directions, and a staged research plan.

The central conclusion is that the project should not treat the current issue as a single tail-calibration problem. The evidence points to two coupled but distinct problems:

1. Normal-price periods have non-trivial q50 level and shape error when evaluated by relative metrics.
2. Extreme-price periods have severe upper-tail undercoverage, even after current event-risk tail overlay.

The next research phase should therefore separate:

```text
normal-period level / shape / relative-error modeling
extreme-period spike / tail / scenario-risk modeling
```

Frequency-domain and multi-scale methods are especially relevant because they provide a principled way to separate slow seasonal structure, daily shape, short-term oscillation, and spike residuals.

## 2. Current Project Context

The repository implements a reproducible PJM/COMED day-ahead forecasting pipeline:

```text
data preparation
-> feature engineering
-> rolling backtest
-> probabilistic model prediction
-> quantile postprocessing
-> evaluation and plots
-> scenario generation and reduction
```

Important project boundaries:

- The workflow boundary is `Workspace`.
- Backtests use rolling windows and a weekly retrain protocol.
- Prediction outputs follow the contract:

```text
ds, y, y_pred, model, split, seed, quantile, metadata
```

- Current main model family is `nhits_tail_grid_weighted_main`.
- Current model produces quantile forecasts, not only point forecasts.
- Current postprocessing includes:
  - monotonic quantile repair;
  - hourly asymmetric CQR;
  - event-risk tail overlay;
  - scenario diagnostics and copula-based scenario generation.

The current active test period used in recent analysis is:

```text
2025-04-02 00:00:00 -> 2026-03-31 23:00:00
n = 8736 hourly observations
```

## 3. Current Model Performance Summary

### 3.1 Absolute-error view

The current postprocessed model has the following full-test performance:

| Metric | Value |
|---|---:|
| MAE | 10.99 |
| RMSE | 24.98 |
| sMAPE | 28.68% |
| pinball | 3.25 |
| q95 upper coverage | 0.913 |
| q99 upper coverage | 0.984 |
| q995 upper coverage | 0.990 |
| mean q99-q50 width | 44.69 |
| mean q995-q50 width | 55.20 |

The absolute MAE can make the model appear acceptable. However, COMED prices often sit around 20-50 USD/MWh in normal periods, so an absolute error of 5-10 is a large relative error.

### 3.2 Relative-error view by actual price

The q50 relative-error breakdown is more concerning:

| Actual price bin | n hours | q50 MAE | median APE | p75 APE | p90 APE | WAPE |
|---|---:|---:|---:|---:|---:|---:|
| 10-20 | 1283 | 5.71 | 24.9% | 46.3% | 83.6% | 34.8% |
| 20-30 | 2455 | 6.17 | 17.5% | 33.3% | 54.0% | 24.8% |
| 30-50 | 2923 | 7.47 | 15.2% | 26.6% | 42.0% | 19.7% |
| 50-100 | 1419 | 15.49 | 19.6% | 32.9% | 44.6% | 23.7% |
| 100-200 | 200 | 45.39 | 30.8% | 46.4% | 63.1% | 35.4% |
| >200 | 130 | 124.75 | 37.3% | 58.0% | 66.5% | 37.6% |

Interpretation:

- The model is not only failing on rare spikes.
- In normal-price regions, q50 relative error is materially high.
- For 20-30 prices, the median percentage error is about 17.5%, and the upper quartile is about 33.3%.
- This matters because many market decisions are made in these normal operating ranges.

### 3.3 Tail behavior by actual price regime

The upper-tail coverage looks reasonable globally but breaks down sharply in the highest actual-price regimes:

| Actual price regime | n hours | q99 coverage | q995 coverage |
|---|---:|---:|---:|
| <= p50 | 4368 | 0.994 | 0.998 |
| p50-p80 | 2621 | 0.993 | 0.998 |
| p80-p90 | 873 | 0.978 | 0.993 |
| p90-p95 | 437 | 0.979 | 0.982 |
| p95-p99 | 349 | 0.903 | 0.905 |
| > p99 | 88 | 0.636 | 0.682 |

Interpretation:

- Normal hours are often overcovered.
- Extreme hours are severely undercovering.
- Global q99 coverage near 0.984 is misleading because ordinary hours dominate the sample.
- A single global tail-thickening rule can improve global pinball while still being structurally crude.

### 3.4 Month-level behavior

The worst months by pinball and q50 MAE include:

| Month | pinball | q50 MAE | q99 coverage | q995 coverage | q99 excess mean |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 8.736 | 26.19 | 0.965 | 0.976 | 2.245 |
| 2026-02 | 4.273 | 14.10 | 0.984 | 0.985 | 0.556 |
| 2025-07 | 4.013 | 12.10 | 0.984 | 0.984 | 1.780 |
| 2025-04 | 3.253 | 11.48 | 0.945 | 0.967 | 0.376 |

Interpretation:

- January 2026 is the dominant failure period.
- July 2025 and February 2026 are also material.
- The system likely needs regime-aware evaluation and modeling rather than only global score minimization.

## 4. Literature-Motivated Baseline Expectations

### 4.1 Public benchmark context

Open EPF benchmark literature reports much lower sMAPE for several benchmark markets and model families. In the NBEATSx benchmark, PJM point-forecast sMAPE values for strong models are roughly in the 11-12% range, while our current q50 full-test sMAPE is 28.68%.

This comparison is not one-to-one because:

- our period is 2025-2026;
- our zone is COMED;
- market volatility may differ materially;
- we optimize probabilistic pinball rather than point sMAPE;
- postprocessing affects tail quantiles more than q50.

Still, it suggests that q50 quality deserves serious attention.

Relevant source:

- [Olivares et al., NBEATSx for electricity price forecasting](https://www.sciencedirect.com/science/article/pii/S0169207022000413)

### 4.2 Benchmarking discipline

The EPF benchmark literature warns against evaluating new methods without strong simple baselines, long test sets, consistent metrics, and statistical tests.

Relevant source:

- [Lago et al., Forecasting day-ahead electricity prices: review and open-access benchmark](https://www.sciencedirect.com/science/article/pii/S0306261921004529)

Implication for this project:

- NHITS should be compared against LEAR/LASSO under the same rolling protocol.
- Evaluation should include sMAPE, WAPE, q50 bias, pinball, coverage, interval width, and tail exceedance.
- Metrics should be reported by month, hour, price bin, and high-risk regime.

### 4.3 LASSO / LEAR relevance

LASSO-based electricity price models remain strong in EPF because they exploit:

- hourly autoregressive structure;
- lagged prices across many hours;
- lagged load and forecast load;
- calendar and holiday effects;
- shrinkage to prevent overfitting.

The LASSO + variance-stabilizing transformation literature is directly relevant because one study uses PJM COMED, system load forecast, and zonal load forecast.

Relevant source:

- [Uniejewski and Weron, LASSO and variance-stabilizing transformations for EPF](https://www.mdpi.com/1996-1073/11/8/2039)

Implication:

- A LEAR/LASSO q50 baseline should be added.
- It should not be treated as a weak benchmark.
- It may also become an ensemble member or residual-correction baseline.

### 4.4 Variance-stabilizing transformations

Electricity prices are spiky, skewed, and sometimes close to zero or negative. Log transforms are not always valid. The VST literature studies alternatives such as:

- asinh;
- mirror log;
- Box-Cox-like transforms;
- polynomial transforms;
- PIT-Gaussian transforms;
- t-PIT transforms.

Relevant source:

- [Uniejewski, Weron and Ziel, variance-stabilizing transformations](https://ideas.repec.org/p/wuu/wpaper/hsc1701.html)

Implication:

- The current fixed `asinh_q95` transform may be too rigid.
- Target transform scale should be tuned or adapted.
- VST should be evaluated by q50 sMAPE/WAPE, not only pinball.

### 4.5 Spike filtering

Recent filtering work argues that historical spikes can distort model estimation. A rolling robust filtering approach can identify and replace extreme spikes in the input used for model training, improving out-of-sample accuracy for several models and markets.

Relevant source:

- [Cerasa and Zani, filtering strategy for out-of-sample EPF](https://www.sciencedirect.com/science/article/pii/S030626192500087X)

Implication:

- A filtered-target backbone is promising.
- Filtering should not remove spike risk from the whole system.
- The architecture should separate:

```text
filtered target for normal curve modeling
spike residual / tail module for extreme risk modeling
```

### 4.6 Modular probabilistic forecasting

GEFCom2014 probabilistic electricity price forecasting used a modular hybrid structure:

```text
point forecasting
-> pre-filtering
-> quantile regression
-> post-processing
```

Relevant source:

- [Maciejowska and Nowotarski, GEFCom2014 probabilistic EPF hybrid model](https://www.sciencedirect.com/science/article/pii/S0169207015001430)

Implication:

- It is reasonable to improve q50/point quality first.
- Quantile calibration and scenario generation should build on a credible marginal center.
- End-to-end quantile training alone may not be sufficient.

## 5. Frequency-Domain and Multi-Scale Motivation

Frequency-domain methods are relevant because electricity prices contain several different time-scale components:

```text
slow trend / long-term seasonal component
daily shape
weekly behavior
weather/load-driven medium-frequency changes
short transient oscillations
spike residuals
```

Trying to force one model to learn all components simultaneously can create two issues:

1. The model overfits or underfits normal shape.
2. Spike behavior contaminates normal-period level estimates.

Frequency-domain and decomposition methods provide a way to separate these effects.

## 6. Frequency-Domain Literature Context

### 6.1 Long-term seasonal component

Nowotarski and Weron show that decomposing prices into a long-term seasonal component and a stochastic component can improve day-ahead EPF.

Relevant source:

- [Nowotarski and Weron, LTSC in day-ahead EPF](https://www.sciencedirect.com/science/article/pii/S014098831630127X)

Core idea:

```text
price = LTSC + stochastic component
forecast price = forecast LTSC + forecast stochastic component
```

### 6.2 Wavelet-based LTSC

Wavelet-based seasonal component models have been shown to outperform simple monthly dummy or sine-based models in LTSC estimation and forecasting.

Relevant source:

- [Janczura et al., robust estimation and forecasting of LTSC](https://www.sciencedirect.com/science/article/abs/pii/S0140988313000686)

Implication:

- A wavelet or multi-scale smoother is a strong candidate for COMED.
- Simple calendar sin/cos features may not be enough.

### 6.3 LASSO with LTSC

The LTSC approach also benefits parameter-rich LASSO models.

Relevant source:

- [Jedrzejewski, Marcjasz and Weron, LTSC with LASSO](https://www.mdpi.com/1996-1073/14/11/3249)

Implication:

- Decomposition should be tested with both NHITS and LEAR/LASSO.
- It should not be limited to a neural model experiment.

### 6.4 Probabilistic SCAR-type models

Probabilistic extensions of seasonal component models show that decomposition can improve distributional forecasts, not only point forecasts.

Relevant source:

- [LTSC Part II: probabilistic forecasting](https://www.sciencedirect.com/science/article/pii/S0140988318300653)

Implication:

- Frequency decomposition can be part of the probabilistic pipeline.
- It does not have to be only a q50 preprocessing trick.

## 7. Frequency-Domain Research Ideas

This section lists concrete frequency-domain ideas grouped by use case.

### 7.1 Use frequency decomposition to improve the target

Core design:

```text
y_t = low_frequency_t + high_frequency_residual_t
```

Where:

- `low_frequency_t` captures slow seasonal and level structure.
- `high_frequency_residual_t` captures daily deviations, transient dynamics, and spikes.

Candidate decomposition methods:

1. Wavelet smoothing.
2. Hodrick-Prescott filter.
3. STL-like decomposition.
4. DCT low-pass smoothing.
5. FFT low-pass smoothing.
6. Rolling robust spline or local regression.

Research question:

```text
Does predicting residual_t and adding back low_frequency_t reduce q50 sMAPE/WAPE?
```

Expected benefit:

- Backbone model sees a more stationary residual target.
- Spike contamination of the normal curve is reduced.
- q50 shape may improve in 20-50 price ranges.

Risks:

- Decomposition can leak future information if not rolling/causal.
- Low-pass filters must be implemented with strict forecast-time availability.
- If the low-frequency component is poorly extrapolated, it can introduce level bias.

Safe protocol:

```text
For each forecast day:
  fit decomposition only on history up to day-1
  estimate low-frequency history
  forecast/extrapolate next-day low-frequency component
  model residual component
  add components back
```

### 7.2 Use frequency features as model inputs

Frequency features can summarize recent shape and volatility more robustly than raw lags.

Candidate rolling windows:

- 24 hours;
- 48 hours;
- 72 hours;
- 168 hours;
- 336 hours.

Candidate features:

| Feature | Meaning |
|---|---|
| low-frequency energy | strength of slow movement |
| high-frequency energy | short-term volatility |
| high/low energy ratio | relative noisiness |
| spectral entropy | whether recent prices are structured or chaotic |
| dominant frequency strength | whether daily/weekly rhythm is strong |
| wavelet detail energy | transient/spike-like behavior |
| wavelet approximation level | smoothed recent level |
| phase of daily component | where the current curve sits in daily cycle |
| DCT coefficients | compressed recent shape |
| rolling shape PCA scores | empirical daily-shape representation |

Possible feature names:

```text
price_fft_energy_24_low
price_fft_energy_24_high
price_fft_high_low_ratio_168
price_spectral_entropy_168
price_wavelet_detail_energy_24
price_wavelet_detail_energy_168
price_dct_coef_24_1
price_dct_coef_24_2
price_dct_coef_24_3
price_daily_shape_pca_1
price_daily_shape_pca_2
```

Expected benefit:

- Better regime recognition.
- Better distinction between smooth normal days and unstable days.
- Better q50 residual correction.
- Better RAG retrieval.

Risks:

- Too many spectral features can overfit.
- Some spectral features are hard to interpret.
- Features based on raw price windows must be carefully aligned to avoid horizon leakage.

### 7.3 Use frequency-domain similarity for RAG

Current RAG thinking uses structured context:

```text
hour
calendar
load forecast
weather
price lag
spike_score
prior-day ramp
```

Frequency-domain RAG would add recent shape descriptors:

```text
past 24/48/168h price curve
-> spectral or wavelet embedding
-> kNN retrieval
```

Possible embedding options:

1. DCT coefficients of the last 24h and 168h.
2. FFT amplitude spectrum of recent residuals.
3. Wavelet energy vector across scales.
4. PCA of normalized daily price curves.
5. Shapelet-style descriptors for peak timing and recovery.
6. Hybrid embedding:

```text
[structured features, spectral features, residual-shape features]
```

Candidate retrieval targets:

```text
q50 residual
q90 residual
q99 positive residual
PIT values
daily path residual vector
```

Recommended first RAG correction:

```text
q50_corrected = q50_model + weighted_median(neighbor_q50_residual)
```

Then:

```text
q99_corrected = q99_model + weighted_quantile(neighbor_positive_q99_residual)
```

Expected benefit:

- Similar days are selected by recent behavior, not only by calendar/load/weather.
- Frequency embedding is more robust to small timing shifts than raw Euclidean distance.
- It may improve ordinary shape errors and spike-tail errors with separate residual memories.

Risks:

- If the same history window is heavily autocorrelated, retrieval may overfit recent regimes.
- If validation memory is too small, nearest neighbors may not include rare spike analogs.
- Distance weighting must be tuned and audited.

Required audit outputs:

```text
neighbor dates
neighbor distances
neighbor actual prices
neighbor q50 residual distribution
neighbor q99 residual distribution
retrieval strength
correction size
before/after q50 sMAPE
before/after q99 coverage
```

### 7.4 Use frequency-domain loss or regularization

A model can be evaluated or lightly regularized on curve shape:

```text
loss = time_domain_loss + lambda * frequency_domain_shape_loss
```

Candidate frequency losses:

- DCT coefficient error for the 24h predicted curve;
- low-frequency reconstruction error;
- high-frequency energy mismatch penalty;
- daily-shape phase/timing penalty;
- spectral entropy mismatch.

Potential use:

- Not as the first training change.
- More useful after we know which frequencies the current model misses.

Risks:

- Frequency loss may smooth true spikes.
- It may improve visual shape while hurting price-bin sMAPE or pinball.
- It complicates quantile training.

Recommendation:

- Use frequency-domain diagnostics first.
- Consider frequency loss only if diagnostics show systematic shape-frequency errors.

### 7.5 Frequency-domain postprocessing

Postprocessing idea:

```text
predicted 24h q50 path
-> decompose
-> smooth implausible high-frequency component
-> reconstruct path
```

Alternative:

```text
quantile paths
-> smooth only q50/central quantiles
-> leave q99/q995 tail untouched or corrected separately
```

Potential benefit:

- Reduces noisy day-ahead shapes.
- May improve sMAPE in normal periods.

Risks:

- Easy to erase true peaks.
- Can make probabilistic paths inconsistent.
- Can hide model weakness rather than solve it.

Recommendation:

- Treat as a diagnostic or fallback.
- Do not start with this as the primary method.

## 8. Spike Filtering Research Design

Spike filtering should be tested as a training-target strategy, not as a final forecast censoring strategy.

### 8.1 Concept

```text
raw_y = normal_component + spike_residual
filtered_y ~= normal_component
spike_residual = raw_y - filtered_y
```

Backbone model:

```text
features -> filtered_y quantiles or q50
```

Tail module:

```text
features + residual memory -> spike residual / upper tail correction
```

### 8.2 Candidate filters

1. Rolling median/MAD filter by hour.
2. Rolling robust regression residual filter.
3. Wavelet detail thresholding.
4. Daily-shape outlier filter.
5. Similar-day residual filter.
6. Hybrid:

```text
low-frequency smoother + robust residual threshold
```

### 8.3 Important constraints

- Filtering must be causal in rolling backtests.
- Filtering should only use data available before the forecast date.
- Filtered target should not erase spike information from the entire pipeline.
- Raw target remains necessary for tail evaluation and residual memory.

### 8.4 Evaluation

Compare:

```text
raw target NHITS
filtered target NHITS
filtered target + q50 residual RAG
filtered target + tail residual RAG
filtered target + current event overlay
```

Metrics:

- q50 sMAPE by price bin;
- WAPE by price bin;
- q50 bias by hour;
- q99/q995 coverage in p95-p99 and >p99 actual regimes;
- pinball;
- CRPS;
- scenario path diagnostics.

## 9. RAG Research Design

RAG should be treated as retrieval-conditioned calibration, not as a generic language-model-style component.

### 9.1 RAG memory source

Preferred memory:

```text
rolling out-of-sample validation predictions
```

Reason:

- Training fitted residuals are too optimistic.
- Calibration memory should be produced in the same environment as test predictions.
- Validation predictions are closer to the real deployment situation.

Future extension:

```text
generate rolling OOS predictions for earlier training periods
```

This can enlarge memory without using in-sample fitted residuals.

### 9.2 RAG correction types

#### q50 residual correction

Purpose:

- reduce ordinary level/shape error;
- improve sMAPE and WAPE.

Formula:

```text
neighbors = kNN(test_context, memory_context)
delta_q50 = weighted_median(memory_y - memory_q50)
q50_new = q50_old + shrinkage * delta_q50
```

#### distribution shift

Purpose:

- shift all central quantiles coherently when local residuals show bias.

Formula:

```text
q_tau_new = q_tau_old + shrinkage * local_residual_shift
```

#### q99 positive residual correction

Purpose:

- improve upper-tail coverage in high-risk regimes.

Formula:

```text
positive_residual = max(y - q99, 0)
tail_uplift = weighted_quantile(neighbor_positive_residual, target_level)
q99_new = q99_old + shrinkage * tail_uplift
q995_new = q995_old + shrinkage * tail_uplift
```

#### PIT remap

Purpose:

- correct local probability calibration when local PIT distribution is reliably non-uniform.

Warning:

- Recent probes suggest naive PIT shrinkage can reduce night overcoverage but hurt spike coverage.
- PIT remap should not be used blindly.

### 9.3 RAG distance design

Candidate context groups:

```text
calendar and hour features
load forecast features
weather features
lagged price features
prior-day ramp features
spike_score
frequency-domain shape features
recent residual features
```

Distance options:

1. Weighted Euclidean distance over standardized features.
2. Cosine distance over normalized embeddings.
3. Mahalanobis distance with shrinkage covariance.
4. Two-stage retrieval:

```text
filter by broad regime
-> rank by weighted distance
```

5. Hybrid structured + frequency distance:

```text
D = alpha * D_structured + beta * D_frequency + gamma * D_recent_residual
```

Recommended first design:

```text
weighted Euclidean structured features
+ DCT/wavelet energy features
+ transparent audit of neighbors
```

Avoid initially:

- double-tower neural retrieval;
- opaque learned embeddings;
- high-dimensional embeddings without auditability.

## 10. Scenario Generation and Copula Implications

Scenario generation quality depends on marginal quantiles and path dependence.

Current issue:

- If q50 and marginal tails are flawed, copula improvements cannot fully solve scenario realism.
- Vine/copula can improve dependence structure, but cannot create credible marginal tail support by itself.

Recommended order:

```text
fix q50 / normal-period relative error
-> fix marginal upper-tail calibration
-> then improve path dependence with copula/vine
```

Possible future path:

1. Generate calibrated marginal distributions after q50/RAG/tail corrections.
2. Fit copula or vine on OOS residual/PIT paths.
3. Generate raw scenarios.
4. Reduce scenarios with Wasserstein-aware forward selection.
5. Evaluate path-level metrics:
   - daily max distribution;
   - ramp distribution;
   - night/day shape realism;
   - spike timing;
   - scenario coverage of actual path.

## 11. Proposed Research Phases

### Phase 1: Evaluation hardening

Goal:

Make relative-error and regime diagnostics first-class outputs.

Add evaluation outputs:

```text
overall q50 sMAPE / WAPE
price-bin q50 sMAPE / WAPE
monthly q50 sMAPE / WAPE
hourly q50 bias
actual-regime q99/q995 coverage
tail exceedance mean and max
daily peak gap diagnostics
```

Success criteria:

- Every experiment reports normal-period and tail-period metrics separately.
- No experiment is judged by global pinball alone.

### Phase 2: Strong point baselines

Goal:

Determine whether NHITS q50 is underperforming simple strong baselines.

Experiments:

```text
LEAR/LASSO q50
ElasticNet q50
seasonal naive
current NHITS q50
```

Feature candidates:

- lagged prices: 24, 48, 72, 168;
- prior-day min/max/mean/ramp;
- all-hour previous-day profile;
- load forecasts;
- weather;
- calendar;
- holidays;
- spectral features.

Success criteria:

- Identify whether q50 weakness is a neural model issue, feature issue, target issue, or market-regime issue.

### Phase 3: VST and filtered target probes

Goal:

Reduce normal-period relative error.

Experiments:

```text
raw target
asinh scale grid
rolling asinh scale
median/MAD normalized asinh
N-PIT-like transform
filtered target
filtered target + VST
```

Success metrics:

- q50 sMAPE in 20-30 and 30-50 price bins;
- WAPE in 10-50 price range;
- monthly q50 bias;
- no unacceptable degradation of q99/q995 tail after tail correction.

### Phase 4: Frequency decomposition probes

Goal:

Test whether decomposition improves normal curve shape and q50 relative error.

Experiments:

```text
wavelet LTSC + residual model
HP-filter LTSC + residual model
DCT low-pass + residual model
rolling spectral features only
frequency RAG only
frequency features + RAG
```

Diagnostics:

```text
q50 error decomposition into low-frequency and high-frequency components
spectral energy of residual error
hourly shape error
daily path shape error
```

Success metrics:

- lower q50 sMAPE without hiding spike risk;
- improved daily shape alignment;
- lower residual bias in normal-price regimes.

### Phase 5: q50 residual RAG

Goal:

Use local analogs to correct q50 level and shape.

Experiment:

```text
memory = validation OOS predictions
target = test predictions
correction = weighted median neighbor q50 residual
```

Variants:

- structured distance only;
- frequency distance only;
- structured + frequency distance;
- different k values;
- different shrinkage rules.

Success metrics:

- q50 sMAPE reduction by price bin;
- WAPE reduction;
- stable month-level improvement;
- no large degradation in tail calibration after distribution shift.

### Phase 6: Tail residual RAG

Goal:

Replace coarse global event overlay with local residual-aware tail correction.

Experiment:

```text
neighbor_positive_q99_residual
-> local uplift
-> apply only when local exceedance evidence is strong
```

Success metrics:

- improve p95-p99 and >p99 q99/q995 coverage;
- avoid excessive widening in ordinary hours;
- improve pinball or keep pinball stable;
- reduce daily peak q99 gaps.

### Phase 7: Scenario and dependence refinement

Goal:

Improve path realism after marginal distributions are credible.

Experiments:

```text
Gaussian copula
Student-t copula
vine copula
latent-factor copula
frequency-conditioned copula features
```

Success metrics:

- daily max coverage;
- ramp distribution;
- scenario path energy/smoothness;
- spike timing coverage;
- reduced-scenario representativeness.

## 12. Key Design Decisions to Resolve

### 12.1 Should the backbone target be raw price or filtered price?

Options:

1. Raw price target.
2. Filtered normal-component target.
3. Residual target after LTSC decomposition.
4. Multi-head target:

```text
normal component
spike residual
quantiles
```

Current recommendation:

Start with filtered or decomposed target probes before changing the production model.

### 12.2 Should frequency decomposition happen before or after VST?

Options:

```text
decompose raw y -> transform residual
transform y -> decompose transformed y
decompose and transform both target and exogenous variables
```

Literature suggests this ordering matters. It should be tested explicitly.

Initial candidate:

```text
rolling decomposition on raw y
-> residual normalization
-> asinh transform on residual
```

### 12.3 Should RAG correct q50 only or the whole distribution?

Options:

1. q50 only.
2. central quantile shift.
3. upper-tail uplift only.
4. full PIT remap.
5. combined but gated correction.

Current recommendation:

Start with q50 residual correction. Then add local tail uplift. Use PIT remap only after local reliability is established.

### 12.4 Should LEAR/LASSO be a benchmark or an ensemble member?

Options:

1. Benchmark only.
2. q50 ensemble member.
3. quantile-regression input.
4. RAG feature source.

Current recommendation:

Run it first as a benchmark. If it beats NHITS q50 in normal regimes, include it as an ensemble or residual target.

## 13. Risks and Failure Modes

### 13.1 Leakage risk

Decomposition, filtering, and RAG can easily leak future information.

Mitigation:

- all filters must be rolling and causal;
- all memory residuals must be out-of-sample;
- all feature generation must use only forecast-time available data;
- audit timestamps for every derived feature.

### 13.2 Over-smoothing risk

Frequency filtering can erase legitimate peaks.

Mitigation:

- separate normal component from spike residual;
- evaluate top-price regimes separately;
- preserve raw target for tail residual modeling.

### 13.3 Metric tradeoff risk

Improving sMAPE can hurt tail risk.

Mitigation:

- maintain separate acceptance criteria for q50 and q99/q995;
- never select solely on sMAPE or pinball;
- use multi-objective experiment summaries.

### 13.4 Memory-size risk for RAG

Validation-only memory may be too small for rare regimes.

Mitigation:

- generate additional rolling OOS training-period memory;
- use regime-aware fallback;
- audit neighbor support for each target day.

### 13.5 Complexity risk

Frequency decomposition, filtering, RAG, and copula can create a complex pipeline.

Mitigation:

- introduce one module at a time;
- require ablation evidence;
- keep postprocess-only probes before retraining;
- formalize only methods that show robust benefit.

## 14. Recommended Near-Term Experiments

### Experiment 1: Relative-error evaluation module

Output:

```text
relative_error_by_price_bin.csv
relative_error_by_month.csv
relative_error_by_hour.csv
actual_regime_tail_coverage.csv
```

Purpose:

Provide a stable scoreboard for all future experiments.

### Experiment 2: LEAR/LASSO q50 baseline

Purpose:

Measure whether current NHITS q50 is competitive in normal-price regimes.

Expected outcome:

- If LEAR is better: use it as q50 ensemble/residual baseline.
- If NHITS is better: focus on transform/filter/RAG rather than model family.

### Experiment 3: Spike-filtered q50 probe

Purpose:

Test whether historical spikes are distorting q50 learning.

Variants:

```text
rolling median/MAD filter
rolling robust regression filter
wavelet detail threshold filter
```

### Experiment 4: Frequency error decomposition

Purpose:

Determine whether q50 errors are mostly low-frequency level errors or high-frequency shape errors.

Output:

```text
error_low_frequency_component
error_high_frequency_component
monthly spectral error
hourly spectral error
```

### Experiment 5: Frequency-RAG q50 correction

Purpose:

Use similar recent shapes to correct q50 residuals.

Variants:

```text
structured RAG
frequency RAG
structured + frequency RAG
```

### Experiment 6: Local tail residual RAG

Purpose:

Replace global tail uplift with local residual-based uplift.

Decision rule:

```text
apply uplift only when neighbor q99 exceedance or positive residual evidence is strong
```

## 15. Proposed Acceptance Criteria

No single metric should decide success. A candidate method should satisfy:

### Normal-period criteria

- q50 sMAPE improves in 20-30 and 30-50 bins.
- WAPE improves in 10-50 range.
- q50 bias by hour does not worsen materially.
- monthly sMAPE is more stable.

### Tail criteria

- q99/q995 coverage improves in p95-p99 and >p99 actual regimes.
- daily peak q99 gap decreases.
- ordinary-hour interval width does not inflate excessively.

### Probabilistic criteria

- pinball improves or remains stable.
- crossing rate remains near zero.
- CRPS improves or remains stable.

### Scenario criteria

- scenario daily max distribution improves.
- ramp and daily path metrics improve.
- reduced scenarios remain representative.

## 16. Overall Recommendation

The next phase should follow this order:

```text
1. harden relative-error and regime evaluation
2. add LEAR/LASSO q50 baseline
3. test spike-filtered and VST targets
4. test frequency decomposition and frequency features
5. test q50 residual RAG with frequency-aware retrieval
6. test local tail residual RAG
7. revisit copula/vine scenario generation after marginal forecasts improve
```

The key architectural direction is:

```text
normal curve model
+ local q50 residual correction
+ local tail residual correction
+ calibrated dependence/scenario model
```

This direction is consistent with the evidence from recent diagnostics and with the main EPF literature themes:

- strong simple baselines matter;
- transformations and filtering matter;
- decomposition can improve both point and probabilistic forecasts;
- similar-day retrieval is a valid EPF idea;
- probabilistic forecasting benefits from modular design.

## 17. Discussion Questions for Research Review

1. Should COMED normal-period q50 forecasting be modeled on raw prices, filtered prices, or decomposition residuals?
2. Which frequency decomposition is most appropriate for rolling day-ahead use: wavelet, HP filter, STL-like, DCT, or FFT low-pass?
3. Should VST be applied before decomposition, after decomposition, or separately to residual components?
4. Is frequency-domain retrieval likely to improve similar-day selection beyond load/weather/calendar features?
5. What is the safest way to prevent frequency filters from erasing meaningful spike information?
6. Should LEAR/LASSO become an ensemble member if it improves q50 sMAPE?
7. Should RAG first correct q50 residuals, tail residuals, or both with separate gates?
8. What metric tradeoff should be accepted between q50 sMAPE and q99/q995 coverage?
9. How much OOS memory is needed for reliable retrieval in rare spike regimes?
10. Should scenario generation wait until marginal q50 and tail calibration improve?

