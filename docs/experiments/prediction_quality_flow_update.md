# Prediction Quality Flow Update

Input feedback: `D:/BaiduNetdiskDownload/prediction_quality_flow_update_direction.md`.

## Review

The feedback is technically aligned with the current evidence:

- direct NHITS feature expansion has shown validation/test instability,
- `future_price_lag_168 + 336` should stay rejected,
- `prior_day_price_state` should not enter default `futr_exog`,
- `future_price_lag_168` and prior-day state are better treated as context
  candidates,
- tail improvements need width-adjusted scoring before promotion.

## Implemented In This Pass

Added quality-flow infrastructure that does not require retraining:

- `src/pjm_forecast/evaluation/quality_gate.py`
- `src/pjm_forecast/evaluation/tail_tradeoff.py`
- `scripts/experiments/build_quality_flow_registry.py`
- unit tests for gate decisions and registry generation

The registry builder reads existing scalar metrics, quantile diagnostics, and
regime metrics, then writes:

```text
artifacts_phase2/quality_flow/benchmark_registry.csv
```

The generated registry is a local artifact view and remains ignored by git.

## Deferred

The following feedback items remain valuable but require a separate evidence
loop:

- `spike_score_diagnostics.csv`,
- `hour_x_regime` CQR grid,
- gated tail blend,
- final spike-basis ablation.

They should build on the benchmark registry and quality gate rather than
expanding NHITS `futr_exog`.

## Follow-up Implementation

The canonical `spike_score` feature now uses historical percentile scoring
instead of full-sample rank normalization. Scores for an existing prefix are
stable when future rows are appended, so the context is safe to use for
postprocess grouping without peeking at the future sample distribution.

Evaluation now writes `{split}_spike_score_diagnostics.csv`, and
`scripts/experiments/evaluate_hour_x_regime_grid.py` compares `hour_cqr`
against `hour_x_regime` thresholds such as `0.50` and `0.67` on an existing
prediction pair without retraining.

## Hour x Regime Grid Check

Command:

```powershell
uv run python scripts\experiments\evaluate_hour_x_regime_grid.py --validation-prediction artifacts_tmp\nhits_tail_grid_weighted_long_spike_context\predictions\nhits_tail_grid_weighted_long_validation_seed7.parquet --test-prediction artifacts_tmp\nhits_tail_grid_weighted_long_spike_context\predictions\nhits_tail_grid_weighted_long_test_seed7.parquet --output-dir artifacts_phase2\hour_x_regime_grid --thresholds 0.50 0.67 --validation-holdout-days 91 --min-group-size 24
```

Summary on the existing spike-context NHITS prediction artifacts:

| Mode | Variant | MAE | Pinball | q99 exceed | q99 excess | width98 |
|---|---|---:|---:|---:|---:|---:|
| validation_holdout | `hour_cqr` | 10.0626 | 2.9405 | 3.34% | 0.5720 | 51.2235 |
| validation_holdout | `hour_regime_cqr_t50` | 10.0427 | 2.9985 | 3.34% | 0.6731 | 48.4921 |
| validation_holdout | `hour_regime_cqr_t67` | 10.0707 | 2.9742 | 3.34% | 0.6203 | 50.2123 |
| test | `hour_cqr` | 10.9858 | 3.2702 | 1.11% | 0.5212 | 120.2417 |
| test | `hour_regime_cqr_t50` | 10.9801 | 3.2719 | 1.69% | 0.5663 | 95.5317 |
| test | `hour_regime_cqr_t67` | 10.9889 | 3.3253 | 1.59% | 0.8157 | 88.9312 |

Decision: do not promote `hour_x_regime` over `hour_cqr` from this grid. The
`0.50` threshold only improves test MAE by a negligible amount while validation
pinball, test pinball, q99 exceedance, and q99 excess are worse. Keep it as a
diagnostic branch until rolling-safe predictions are regenerated and pass the
quality gate.

Canonical config is therefore set back to hourly CQR (`group_by: "hour"`).
`spike_score` remains configured as a prediction context column so evaluation
can still write spike diagnostics from fresh canonical backtests.
