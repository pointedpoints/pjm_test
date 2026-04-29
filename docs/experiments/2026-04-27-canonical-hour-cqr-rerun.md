# Canonical NHITS Hourly CQR Rerun

Date: 2026-04-27

## Commands

```powershell
uv run python scripts\prepare_data.py --config configs\pjm_day_ahead_current_processed.yaml
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split validation --start-from backtest_all_models
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_current_processed.yaml --split test --start-from backtest_all_models
```

`tune_model` was intentionally skipped. This run validates the fixed canonical
NHITS configuration with rolling-safe `spike_score` and hourly CQR.

## Scalar Metrics

| Split | Model | MAE | Pinball | q99 exceed | q99 excess | width98 | post crossing |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | `nbeatsx` | 7.9302 | 2.6597 | 7.58% | 0.8027 | 34.2443 | 0 |
| validation | `nhits_tail_grid_weighted_main` | 8.4362 | 2.5140 | 4.05% | 0.5399 | 46.1740 | 0 |
| test | `nbeatsx` | 11.3003 | 3.6463 | 2.71% | 0.9814 | 65.2001 | 0 |
| test | `nhits_tail_grid_weighted_main` | 10.9858 | 3.2922 | 2.23% | 0.8696 | 69.1474 | 0 |

## Gate Read

Relative to the old NBEATSx artifacts in `artifacts_current`:

- validation: `TAIL_ONLY`; pinball and q99 excess improve, but MAE worsens by
  about `0.5060`.
- test: `PROMOTE`; MAE, pinball, and q99 excess improve without width inflation.

The test target is met (`pinball < 3.40`, q99 exceed below `2.5%`, crossing
zero). The validation MAE regression means the next quality work should focus
on P50/MAE stability rather than adding more tail widening.

## Spike Context

Fresh NHITS prediction parquet files now include `spike_score`:

- `artifacts_current/predictions/nhits_tail_grid_weighted_main_validation_seed7.parquet`
- `artifacts_current/predictions/nhits_tail_grid_weighted_main_test_seed7.parquet`

Both have non-null `spike_score` coverage of `100%`, enabling
`{split}_spike_score_diagnostics.csv` even though canonical CQR grouping is
hourly rather than `hour_x_regime`.
