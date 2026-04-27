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

- rolling-safe `spike_score`,
- `spike_score_diagnostics.csv`,
- `hour_x_regime` CQR grid,
- gated tail blend,
- final spike-basis ablation.

They should build on the benchmark registry and quality gate rather than
expanding NHITS `futr_exog`.
