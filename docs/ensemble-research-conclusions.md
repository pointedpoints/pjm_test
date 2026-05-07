# Ensemble Strategy Research — Key Conclusions

> Sources: M5 competition code, 18 papers/repos, GPT-5.5 literature review (2026-05-06)
> Full report: `.hermes/runs/20260506-192406-ensemble-research/report.md`

## Core finding

NHITS and NBEATSx share the same NeuralForecast/N-BEATS block family. Combining them
averages two variants of the same bias — residual covariance is high, diversity gain is low.

## Recommended ensemble architecture (v1)

Family-balanced point ensemble, 6–8 members across 4–5 families:

| Family | Weight | Members | Rationale |
|--------|--------|---------|-----------|
| Linear (LEAR) | 25% | LEAR-long (+ LEAR-short if validated) | Sparse EPF baseline, cheap, strong |
| Tree | 35% | LightGBM-global + XGBoost/LightGBM-short | Nonlinear interactions, regime splits |
| Neural | 30% | NHITS-main (NBEATSx no separate vote) | Global sequence learner |
| Naive/Anchor | 10% | Seasonal naive / day-hour median | Robustness floor |

## Weighting strategy

1. **Level 0**: Equal mean (baseline — always report)
2. **Level 1**: Family-balanced average (no family dominates by member count)
3. **Level 2**: Regularized rolling stacking with ridge + family caps (only if validated)

Per the "forecast combination puzzle" (Smith & Wallis 2009), equal weights often beat
estimated weights due to sampling error in small validation sets.

## Where CQR/post-processing fits

```
member point forecasts → family-balanced ensemble → residual collection → CQR calibration → final intervals
```

Calibrate AFTER ensemble, not per-member. Ensemble residuals are lower variance.

## Staged implementation

- **Stage A** (zero new training): Combine existing NHITS + LightGBM + LEAR + seasonal_naive predictions
- **Stage B**: Add LEAR-short and LightGBM-short (window diversity)
- **Stage C**: Ensemble-level CQR by hour/block + spike specialist if needed
- **Stage D**: Optional ARIMA/ETS per hour (only if cheap and empirically complementary)

## Evaluation protocol

For every candidate ensemble, report:
1. Overall MAE/RMSE
2. P50/P90/P95 APE in $10–30 price bucket (the real pain point)
3. Spike bucket metrics (separate, so spike improvement doesn't hide normal-price failure)
4. Residual correlation matrix by model family
5. Diebold-Mariano tests vs individual models
6. Stability across rolling folds

## Acceptance bar for adding a member

Add model m if it (a) improves family-balanced ensemble on rolling folds, OR
(b) improves a target bucket (e.g. $10–30 P90 APE) without harming overall MAE, OR
(c) improves interval calibration after CQR.
