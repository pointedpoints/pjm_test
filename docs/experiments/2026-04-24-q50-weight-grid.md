# NHITS q50 Weight Grid

## Purpose

The current postprocess-only q50 bias correction is available but not promoted:
it improved raw pinball in a local replay and slightly worsened post-CQR test
metrics. The next safer path is model-side median learning.

This experiment holds the promoted NHITS dense upper-tail setup fixed and varies
only the q0.50 loss weight:

- `nhits_q50w100`: q50 weight `1.00`
- `nhits_q50w125`: q50 weight `1.25`
- `nhits_q50w150`: q50 weight `1.50`

All other quantile weights, quantile deltas, monotonicity penalty, architecture,
and exogenous contracts match the promoted tail-grid setup.

## Config

- `configs/experiments/pjm_current_validation_nhits_q50_weight_grid.yaml`
- Processed data: `data/processed_nhits_q50_weight_grid/`
- Artifacts: `artifacts_tmp/nhits_q50_weight_grid/`
- Split target: `validation`
- Scenario diagnostics: disabled
- CQR calibration: disabled
- Monotonic quantile postprocess: enabled

## Run

```powershell
uv run python scripts\prepare_data.py --config configs\experiments\pjm_current_validation_nhits_q50_weight_grid.yaml
uv run python scripts\backtest_all_models.py --config configs\experiments\pjm_current_validation_nhits_q50_weight_grid.yaml --split validation
uv run python scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_validation_nhits_q50_weight_grid.yaml --split validation
```

## Promotion Rule

Prefer a q50-weight variant only if it improves validation `q50_mae` without
materially worsening validation pinball, CRPS, q99 exceedance, or tail width.
After selecting a candidate, rerun test with the canonical CQR settings before
promotion.

## Validation Result

Completed on the validation split with monotonic postprocess and no CQR:

| model | pinball | CRPS | q50 MAE | q99 exceed | width 98 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nhits_q50w100` | 2.5140 | 6.4051 | 8.4362 | 4.05% | 46.17 |
| `nhits_q50w125` | 2.5181 | 6.4117 | 8.4339 | 4.05% | 46.16 |
| `nhits_q50w150` | 2.5081 | 6.3786 | 8.3819 | 4.21% | 46.06 |

`nhits_q50w150` is the only useful candidate: it improves q50 MAE and global
pinball/CRPS, while q99 exceedance rises slightly. DM vs `q50w100` is
significant on validation (`p ~= 0.0066`).

Test candidate config:

```powershell
uv run python scripts\backtest_all_models.py --config configs\experiments\pjm_current_test_nhits_q50w150.yaml --split test
uv run python scripts\evaluate_and_plot.py --config configs\experiments\pjm_current_test_nhits_q50w150.yaml --split test
```

## Test Result

Completed on the test split with canonical CQR and scenario diagnostics:

| model | pinball | CRPS | q50 MAE | q99 exceed | width 98 | scenario energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current canonical q50w100 | 3.2729 | 8.2637 | 10.9801 | 1.90% | 93.98 | 51.68 |
| `nhits_q50w150` | 3.2883 | 8.2926 | 10.9673 | 1.85% | 95.06 | 51.97 |

Decision: do not promote `nhits_q50w150`. It gives a small P50 MAE gain on
test, but worsens global probabilistic quality, widens the central 98%
interval, increases worst q99 underprediction, and degrades scenario scores.
Keep canonical q50 weight at `1.00` until a candidate improves P50 without
giving back pinball/CRPS.
