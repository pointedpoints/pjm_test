# Phase 1 P50-Friendly Neural Validation

## Scope

- Plan items: `E8`, `E9`
- Goal: check whether restored feature inputs plus a `P50`-leaning quantile grid
  can beat the current neural validation baselines on `MAE` without giving back
  too much probabilistic quality.

## Config

- Validation compare config:
  `configs/experiments/pjm_current_validation_phase1_p50_friendly_neural.yaml`
- Models:
  - `nbeatsx_p50_friendly`
  - `nhits_p50_friendly`

## Validation Results

| model | mae | rmse | pinball | raw crossing |
| --- | ---: | ---: | ---: | ---: |
| `nbeatsx_p50_friendly` | `8.0973` | `13.3415` | `2.9920` | `0.5096` |
| `nhits_p50_friendly` | `8.1012` | `13.6483` | `3.0907` | `0.5563` |
| `lightgbm_q` | `8.1895` | `15.9728` | `2.9180` | `0.0000` |
| `nhits_struct` | `8.2217` | `13.6928` | `2.8866` | `0.5348` |
| `nbeatsx_current_struct` | `8.2667` | `13.7378` | `2.8951` | `0.6101` |

DM test inside the `E9` pair:

- `nbeatsx_p50_friendly` vs `nhits_p50_friendly`: statistic `-0.0932`, `p=0.9257`

## Decision

- Keep `nbeatsx_p50_friendly` as the only surviving Phase 1 candidate.
- Drop `nhits_p50_friendly` from the Phase 1 path.
- Do **not** promote either model from validation alone.

Reasoning:

- `nbeatsx_p50_friendly` materially improved `MAE` versus the current neural
  baselines and beat the tree benchmark floor on median-side quality.
- The same model regressed probabilistic quality:
  - `pinball 2.9920` vs `2.8866` for `nhits_struct`
  - `pinball 2.9920` vs `2.9180` for `lightgbm_q`
- `nhits_p50_friendly` was worse than `nbeatsx_p50_friendly` on both `MAE` and
  `pinball`, so it is not worth further Phase 1 runtime.

## Next Step

Run the required Phase 1 calibration comparison on test for
`nbeatsx_p50_friendly` only:

1. raw monotonic
2. `hour`
3. `hour_x_regime`

No promotion is allowed unless postprocessed test metrics recover enough
probabilistic quality while retaining the `MAE` gain.
