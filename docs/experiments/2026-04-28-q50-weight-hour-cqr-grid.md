# Q50 Weight Hourly CQR Grid

Date: 2026-04-28

## Goal

Test whether increasing only the median quantile weight can reduce P50/MAE
without changing the dense upper-tail grid, model architecture, or hourly CQR
postprocess contract.

## Config

`configs/experiments/pjm_current_validation_nhits_q50_weight_hour_cqr.yaml`

The grid keeps canonical NHITS settings fixed and changes only the q0.50 loss
weight:

- `nhits_q50w100_hour_cqr`: q50 weight `1.00`
- `nhits_q50w125_hour_cqr`: q50 weight `1.25`
- `nhits_q50w150_hour_cqr`: q50 weight `1.50`

Validation artifacts are under:

```text
artifacts_phase3/q50_weight_hour_cqr/
```

## Commands

```powershell
uv run python scripts\prepare_data.py --config configs\experiments\pjm_current_validation_nhits_q50_weight_hour_cqr.yaml
uv run python scripts\run_pipeline.py --config configs\experiments\pjm_current_validation_nhits_q50_weight_hour_cqr.yaml --split validation --start-from backtest_all_models
```

Only the validation winner was run on test:

```powershell
uv run python - <<'PY'
from pathlib import Path
from pjm_forecast.workspace import Workspace

workspace = Workspace.open(Path("configs/experiments/pjm_current_validation_nhits_q50_weight_hour_cqr.yaml"))
workspace.config.raw["backtest"]["benchmark_models"] = ["nhits_q50w150_hour_cqr"]
workspace.backtest(split="test")
workspace.evaluate(split="test")
workspace.export_report(split="test")
PY
```

## Validation Results

| Model | MAE | Pinball | q99 exceed | q99 excess | width98 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| `q50w100` | 8.4362 | 2.5140 | 4.05% | 0.5399 | 46.1740 | 0 |
| `q50w125` | 8.4339 | 2.5181 | 4.05% | 0.5488 | 46.1648 | 0 |
| `q50w150` | 8.3819 | 2.5081 | 4.21% | 0.5554 | 46.0573 | 0 |

`q50w150` is the validation winner on MAE and pinball, but it gives back some
upper-tail quality.

## Test Result for q50w150

| Model | MAE | Pinball | q99 exceed | q99 excess | width98 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| current canonical | 10.9858 | 3.2922 | 2.23% | 0.8696 | 69.1474 | 0 |
| `q50w150` | 10.9706 | 3.3123 | 2.36% | 0.9105 | 68.2648 | 0 |

## Decision

Do not promote `q50w150` into canonical. It slightly improves test MAE, but
test pinball and q99 excess regress. Keep the current canonical
`nhits_tail_grid_weighted_main` as the main reference.

The useful signal is that higher q50 weight can help validation P50, but the
tradeoff is not clean enough. Next P50 work should test less tail-disruptive
knobs, especially `loss_delta` or a bounded postprocess median-bias layer,
before changing canonical training weights.
