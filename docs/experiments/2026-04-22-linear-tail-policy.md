# Linear Tail Policy Evaluation

Date: 2026-04-22

## Scope

This change adds an explicit tail policy for `QuantileSurface.ppf()` and the
scenario evaluation path. The immediate goal is to remove the hard cap imposed
by flat extrapolation at the highest quantile knot while preserving the old
behavior as the default.

Implemented policies:

- `flat`: existing conservative behavior. Values outside the observed quantile
  range are clamped to the outer knots.
- `linear`: extrapolates the lower and upper tails using the first two and last
  two quantile knots.

The policy is passed through quantile surface construction, copula fitting,
scenario diagnostics, and `report.scenario_evaluation.tail_policy`. Scenario
diagnostic output records the active policy.

## Experiment Config

Config:

- `configs/experiments/pjm_current_full_compare_nhits_linear_tail.yaml`

This config reuses the existing full-compare NHITS prediction directory and
writes derived metrics/plots/report artifacts to:

- `artifacts_experiments/full_compare_nhits_linear_tail/`

## Results

Test scenario diagnostics, flat baseline vs linear tail:

| metric | flat | linear | delta |
|---|---:|---:|---:|
| energy_score | 51.846748 | 51.849549 | +0.002801 |
| variogram_score | 707.388439 | 706.447411 | -0.941028 |
| path_mean_mae | 10.824443 | 10.829301 | +0.004858 |
| daily_max_abs_error | 21.841067 | 21.850578 | +0.009511 |
| daily_spread_abs_error | 22.300662 | 22.297694 | -0.002968 |
| daily_ramp_abs_error | 13.042366 | 13.035078 | -0.007288 |
| spearman_corr_mae | 0.049880 | 0.050233 | +0.000353 |

## Interpretation

The infrastructure works and removes the structural q99 ceiling for sampled
scenario paths. On the current NHITS predictions, however, scenario diagnostics
barely move. This indicates that the main bottleneck is not only the flat tail
cap in `QuantileSurface.ppf()`, but the upstream marginal geometry: the model's
upper quantile grid and q95-q99 slope are still too weak for spike regimes.

The next high-value experiments should therefore prioritize:

- denser upper-tail quantiles such as q0.975/q0.995;
- stronger upper-tail loss weights and tuned Huber deltas;
- explicit spike/regime features or a gated spike-score blend;
- regime-aware calibration before more complex copula changes.

## Validation

Unit/contract tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_quantile_surface.py tests/test_copula.py tests/test_scenarios.py tests/test_config_contracts.py tests/test_evaluator.py tests/test_evaluate_script.py -q
```

Result: `40 passed, 1 warning`.

Scenario sanity check:

```powershell
.venv\Scripts\python.exe scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_full_compare_nhits_linear_tail.yaml --split test
```
