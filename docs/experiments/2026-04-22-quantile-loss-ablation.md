# Quantile Loss Ablation

Date: 2026-04-22

## Scope

This experiment tested training-time changes for raw probabilistic quantile
quality on the current PJM/COMED pipeline. The goal was to reduce quantile
crossing and improve tail support without relying on evaluation-time
postprocessing.

All runs used the same lightweight validation smoke protocol:

- split: validation
- horizon: 24 hours
- validation window: 28 days
- model family: NHITS
- training budget: 180 steps
- ensemble: disabled, single seed 7
- postprocess: disabled

Configs:

- `configs/experiments/pjm_current_validation_nhits_tail_loss_smoke.yaml`
- `configs/experiments/pjm_current_validation_nhits_loss_ablation.yaml`
- `configs/experiments/pjm_current_validation_nhits_weighted_light_mono.yaml`

## Implemented Loss Controls

- `quantile_weights`: per-quantile weighting for MQLoss and HuberMQLoss.
- `quantile_deltas`: per-quantile Huber deltas for HuberMQLoss.
- `monotonicity_penalty`: soft squared penalty on crossed adjacent quantiles.

These controls are available to both `NBEATSxModel` and `NHITSModel`.

## Results

Raw validation metrics:

| model | crossing | pinball | CRPS | cov90 | width90 | worst-day q99 error |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 66.90% | 2.6669 | 6.1869 | 81.20% | 26.78 | 134.44 |
| weights only | 64.70% | 2.6789 | 6.2122 | 81.32% | 26.86 | 96.69 |
| weights + mono 0.02 | 56.34% | 2.6704 | 6.1998 | 82.10% | 27.29 | 100.45 |
| weights + mono 0.03 | 56.94% | 2.6661 | 6.1887 | 82.49% | 27.45 | 111.01 |
| mono 0.05 | 57.17% | 2.6845 | 6.2359 | 82.74% | 27.63 | 159.15 |
| mono 0.10 | 48.33% | 2.6918 | 6.2588 | 84.02% | 28.10 | 150.67 |
| deltas only | 65.59% | 2.6716 | 6.2000 | 81.41% | 26.92 | 129.47 |
| weights + deltas + mono 0.05 | 52.34% | 2.6907 | 6.2524 | 82.97% | 27.74 | 117.69 |

## Interpretation

Soft monotonicity is the strongest control for reducing raw crossing, but high
penalties hurt pinball and CRPS. Tail weights improve the worst spike-day q99
error, but do not materially solve crossing on their own. Per-quantile Huber
deltas were weak in this smoke test.

The most balanced lightweight candidate is `weights + mono 0.03`: it lowers raw
crossing materially while keeping pinball and CRPS close to baseline and
improving 90% coverage. This remains an experimental candidate, not a canonical
configuration.

## Validation

Unit/contract tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_quantile_losses.py tests/test_models.py tests/test_config_contracts.py -q
```

Result: `42 passed`.
