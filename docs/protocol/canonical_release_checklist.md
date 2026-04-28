# Canonical Release Checklist

## Before Running

- Confirm the working tree is clean except intended changes: `git status --short`.
- Confirm the canonical config is `configs/pjm_day_ahead_current_processed.yaml`.
- Confirm the v1 time protocol remains timezone-naive local time; do not remap timestamps to UTC.
- Confirm generated heavyweight artifacts are not staged.

## Run

Run validation first, then test:

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split validation
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test
```

The closure wrapper runs evaluation, event-risk audit, quality finalization,
report export, and model snapshot export after backtesting.

## Gate

- `artifacts_current/metrics/test_quality_gate_summary.csv` has `decision` equal to `CANONICAL_CANDIDATE`.
- `post_crossing_rate == 0`.
- `post_q99_exceedance_rate <= 0.025`.
- `spike_score_audit_status == PASS`.
- `normal_width_status` is `PASS`, or `WARN` is explicitly documented. Current policy sets `WARN` when normal-hour width ratio is missing or greater than `1.05`.

## Artifacts To Inspect

- `artifacts_current/predictions/nhits_tail_grid_weighted_main_validation_seed7.parquet`
- `artifacts_current/predictions/nhits_tail_grid_weighted_main_test_seed7.parquet`
- `artifacts_current/metrics/validation_metrics.csv`
- `artifacts_current/metrics/test_metrics.csv`
- `artifacts_current/metrics/test_quantile_diagnostics.csv`
- `artifacts_current/metrics/test_event_risk_tail_overlay/overlay_implementation_audit.json`
- `artifacts_current/metrics/test_event_risk_tail_overlay/spike_score_audit.json`
- `artifacts_current/metrics/test_event_risk_tail_overlay/width_by_regime.csv`
- `artifacts_current/metrics/test_quality_gate_summary.csv`
- `artifacts_current/metrics/test_run_manifest.json`
- `artifacts_current/report/`
- `artifacts_current/models/nhits_tail_grid_weighted_main_snapshot/manifest.json`

## Before Push

- Run `uv run python -m pytest` unless intentionally doing a docs-only push.
- Confirm generated heavyweight artifacts are not staged.
- Commit only source, tests, docs, and small intentional configs.
